"""
highlight_clipper.py

Fully local pipeline to turn a long-form video into ranked short clips,
with face-tracking vertical crop and burned-in auto captions.

Pipeline:
  1. Transcribe with faster-whisper (GPU accelerated, word-level timestamps)
  2. Rank highlight-worthy segments with a local Ollama model.
     For long videos, this runs as a map-reduce: the transcript is chunked
     into ~15-min windows, each is ranked independently, then a final pass
     re-ranks the combined candidates against each other for a globally
     consistent, deduplicated top N -- this is what "sharper reasoning"
     means here: the model isn't trying to reason over a 2-hour transcript
     in one shot, and weak/duplicate picks get filtered by a dedicated
     cross-video review pass.
  3. Cut each segment with ffmpeg (full frame)
  4. If --vertical: sample faces with OpenCV, smooth the horizontal
     position over time, and render a 9:16 crop that follows the speaker.
  5. If --captions: build short caption chunks from Whisper's word-level
     timestamps and burn them in as styled subtitles.

Requirements:
  pip install faster-whisper ffmpeg-python requests opencv-python numpy

  Ollama running locally with a model pulled, e.g.:
      ollama pull qwen2.5:32b-instruct
  (32b gives noticeably better ranking judgment than 14b and fits
   comfortably on a 24GB 4090; drop to 14b with --model if it's too slow.)

Usage:
  python highlight_clipper.py input.mp4 --clips 8 --vertical --captions
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests
from faster_whisper import WhisperModel

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:32b-instruct"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".webm", ".m4v", ".flv"}


# ---------------------------------------------------------------------------
# Transcription (with word-level timestamps for captions)
# ---------------------------------------------------------------------------

def transcribe(video_path: Path, model_size: str = "large-v3"):
    """Transcribe video with faster-whisper on GPU. Returns segments with
    start/end/text plus a per-word timestamp list for caption generation."""
    print(f"[1/6] Loading Whisper model ({model_size}) on GPU...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print("[1/6] Transcribing (word-level timestamps enabled)...")
    segments, info = model.transcribe(str(video_path), vad_filter=True, word_timestamps=True)

    transcript = []
    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})
        transcript.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(), "words": words})

    print(f"[1/6] Done. Language: {info.language}, {len(transcript)} segments.")
    return transcript


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, temperature: float = 0.3) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_json_array(raw_output: str):
    cleaned = re.sub(r"```json|```", "", raw_output).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            print("!! Could not parse LLM output as JSON. Raw output was:")
            print(raw_output)
            sys.exit(1)
        return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Highlight ranking: single pass + chunked map-reduce for long videos
# ---------------------------------------------------------------------------

def format_transcript_for_llm(segments):
    lines = []
    for seg in segments:
        lines.append(f"({seg['start']:.1f}-{seg['end']:.1f}) {seg['text']}")
    return "\n".join(lines)


RANK_PROMPT_TEMPLATE = """You are an expert short-form video editor. Below is a timestamped
transcript excerpt of a long-form video (interview, debate, podcast, or livestream).

Identify up to {num_clips} standalone moments in THIS excerpt that would work as
short-form clips ({min_len}-{max_len} seconds each) for YouTube Shorts, Instagram
Reels, or TikTok. Look for: strong hooks, emotional peaks, quotable lines,
conflict/disagreement, surprising reveals, and practical/useful takeaways.
Only pick moments that are fully self-contained (a viewer with no other context
should understand them). Use the exact timestamps shown.

Respond with ONLY a JSON array, no other text, in this exact format:
[
  {{"start": 123.4, "end": 178.2, "score": 87, "hook": "short quote or paraphrase of the opening line", "reason": "one sentence on why this works"}}
]

TRANSCRIPT EXCERPT:
{transcript}
"""

REDUCE_PROMPT_TEMPLATE = """You are doing a final editorial pass on short-form clip candidates
pulled from different sections of a single longer video. Each candidate was scored
in isolation, only relative to its own section -- your job is to judge them against
EACH OTHER, across the whole video.

Select the best {num_clips} moments overall. Re-score each on a single consistent
0-100 scale. Drop candidates that are thematically repetitive (make the same point
as a stronger candidate elsewhere), even if their timestamps don't overlap. Favor
variety (different topics/moments) over several near-identical picks.

Respond with ONLY a JSON array, no other text, in this exact format:
[
  {{"start": 123.4, "end": 178.2, "score": 87, "hook": "...", "reason": "..."}}
]

CANDIDATES:
{candidates_json}
"""


def rank_segments(segments, num_clips, min_len, max_len, model):
    prompt = RANK_PROMPT_TEMPLATE.format(
        num_clips=num_clips, min_len=min_len, max_len=max_len,
        transcript=format_transcript_for_llm(segments),
    )
    return parse_json_array(call_ollama(prompt, model))


def reduce_candidates(candidates, num_clips, model):
    prompt = REDUCE_PROMPT_TEMPLATE.format(
        num_clips=num_clips, candidates_json=json.dumps(candidates, indent=2)
    )
    return parse_json_array(call_ollama(prompt, model))


def dedupe_overlaps(highlights, overlap_threshold=0.5):
    highlights = sorted(highlights, key=lambda h: h["score"], reverse=True)
    kept = []
    for h in highlights:
        overlap = False
        for k in kept:
            latest_start = max(h["start"], k["start"])
            earliest_end = min(h["end"], k["end"])
            intersection = max(0, earliest_end - latest_start)
            union = max(h["end"], k["end"]) - min(h["start"], k["start"])
            if union > 0 and (intersection / union) > overlap_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(h)
    return kept


def chunk_transcript(transcript, chunk_sec, overlap_sec=30):
    if not transcript:
        return []
    duration = transcript[-1]["end"]
    step = max(1, chunk_sec - overlap_sec)
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_sec, duration)
        segs = [s for s in transcript if s["end"] > start and s["start"] < end]
        if segs:
            chunks.append((start, end, segs))
        start += step
    return chunks


def get_highlights(transcript, num_clips, min_len, max_len, model, chunk_minutes=15):
    """Single-pass ranking for short videos; chunked map-reduce for long ones."""
    duration = transcript[-1]["end"] if transcript else 0
    chunk_sec = chunk_minutes * 60

    if duration <= chunk_sec * 1.3:
        print("[2/6] Video is short enough for single-pass ranking.")
        highlights = rank_segments(transcript, num_clips, min_len, max_len, model)
    else:
        chunks = chunk_transcript(transcript, chunk_sec, overlap_sec=30)
        print(f"[2/6] Video is {duration/60:.0f} min -- ranking in {len(chunks)} chunks of ~{chunk_minutes} min...")

        per_chunk_n = max(3, math.ceil(num_clips * 1.5 / len(chunks)))
        all_candidates = []
        for idx, (cs, ce, segs) in enumerate(chunks, 1):
            print(f"    [{idx}/{len(chunks)}] ranking {cs/60:.1f}-{ce/60:.1f} min...")
            all_candidates.extend(rank_segments(segs, per_chunk_n, min_len, max_len, model))

        all_candidates = dedupe_overlaps(all_candidates)
        print(f"[2/6] {len(all_candidates)} candidates after per-chunk dedup. Running cross-video re-rank...")
        highlights = reduce_candidates(all_candidates, num_clips, model)

    highlights = dedupe_overlaps(highlights)
    highlights = sorted(highlights, key=lambda h: h["score"], reverse=True)[:num_clips]
    print(f"[2/6] Final: {len(highlights)} clips selected.")
    return highlights


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------

def cut_segment(video_path: Path, start: float, end: float, out_path: Path):
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(video_path), "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def probe_resolution(path: Path):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# ---------------------------------------------------------------------------
# Face-tracking vertical crop
# ---------------------------------------------------------------------------

_FACE_CASCADE = None


def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASCADE


def detect_face_centers(video_path: Path, sample_every_sec: float = 0.5):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_interval_frames = max(1, int(fps * sample_every_sec))
    cascade = _get_face_cascade()

    samples = []
    frame_idx = 0
    last_center = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval_frames == 0:
            small = cv2.resize(frame, None, fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest
                center_x_px = (x + w / 2) * 2
                last_center = center_x_px / width
            samples.append((frame_idx, last_center))
        frame_idx += 1

    total_frames = frame_idx
    cap.release()
    return samples, total_frames, fps, width


def build_smoothed_centers(samples, total_frames, alpha=0.12):
    if not samples:
        return np.full(max(total_frames, 1), 0.5)
    sample_frames = np.array([s[0] for s in samples])
    sample_vals = np.array([s[1] for s in samples])
    all_frames = np.arange(total_frames)
    interpolated = np.interp(all_frames, sample_frames, sample_vals)
    smoothed = np.empty_like(interpolated)
    smoothed[0] = interpolated[0]
    for i in range(1, len(interpolated)):
        smoothed[i] = alpha * interpolated[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def apply_face_tracked_vertical_crop(input_path: Path, output_path: Path, out_w: int = 1080, out_h: int = 1920):
    cap = cv2.VideoCapture(str(input_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    crop_w = min(orig_w, int(orig_h * 9 / 16))

    print("    Detecting faces across clip...")
    samples, total_frames, _, _ = detect_face_centers(input_path)
    centers = build_smoothed_centers(samples, total_frames)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-i", str(input_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-shortest",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(str(input_path))
    frame_idx = 0
    print("    Rendering face-tracked vertical crop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx = centers[min(frame_idx, len(centers) - 1)]
        cx_px = int(cx * orig_w)
        x0 = max(0, min(orig_w - crop_w, cx_px - crop_w // 2))
        cropped = frame[:, x0 : x0 + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h))
        proc.stdin.write(resized.tobytes())
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()


# ---------------------------------------------------------------------------
# Auto captions (word-timestamps -> chunked ASS subtitles -> burned in)
# ---------------------------------------------------------------------------

def collect_words_in_clip(transcript, clip_start, clip_end):
    words = []
    for seg in transcript:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue
        for w in seg["words"]:
            if clip_start <= w["start"] < clip_end:
                words.append(w)
    words.sort(key=lambda w: w["start"])
    return words


def chunk_words_for_captions(words, clip_start, clip_end, max_words=4, max_chunk_dur=1.6):
    """Group words into short on-screen caption chunks, timestamps relative
    to clip start."""
    chunks = []
    current = []
    current_start = None

    for w in words:
        rel_start = max(0.0, w["start"] - clip_start)
        rel_end = max(rel_start, min(w["end"], clip_end) - clip_start)
        if not current:
            current_start = rel_start
        current.append(w["word"].strip())
        current_end = rel_end
        if len(current) >= max_words or (current_end - current_start) >= max_chunk_dur:
            chunks.append({"start": current_start, "end": current_end, "text": " ".join(current)})
            current = []

    if current:
        chunks.append({"start": current_start, "end": current_end, "text": " ".join(current)})

    return chunks


def seconds_to_ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass_captions(chunks, out_path: Path, video_w: int, video_h: int):
    font_size = max(28, int(video_h * 0.045))
    margin_v = int(video_h * 0.12)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,Arial Black,{font_size},&H00FFFFFF,&H00000000,&H00000000,"
        f"1,0,1,3,0,2,40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for c in chunks:
        start = seconds_to_ass_time(c["start"])
        end = seconds_to_ass_time(c["end"])
        text = c["text"].upper().strip()
        if not text:
            continue
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def burn_captions(video_path: Path, ass_path: Path, output_path: Path):
    # ffmpeg's subtitles filter needs forward slashes and an escaped colon
    # for the drive letter on Windows paths.
    escaped = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    vf = f"subtitles='{escaped}'"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Per-video processing (shared by single-file mode and folder-watch mode)
# ---------------------------------------------------------------------------

def process_video(video_path: Path, out_dir: Path, args):
    """Run the full pipeline (transcribe -> rank -> cut -> crop -> caption)
    for one video, writing all outputs into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = transcribe(video_path, args.whisper_model)
    transcript_path = out_dir / "transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=2))
    print(f"[1/6] Transcript saved to {transcript_path}")

    highlights = get_highlights(transcript, args.clips, args.min_len, args.max_len, args.model, args.chunk_minutes)

    print("[3/6] Selected clips:")
    for i, h in enumerate(highlights, 1):
        print(f"  #{i} score={h['score']} {h['start']:.1f}s-{h['end']:.1f}s :: {h.get('hook', '')}")

    print("[4/6] Cutting full-frame segments with ffmpeg...")
    manifest = []
    for i, h in enumerate(highlights, 1):
        raw_path = out_dir / f"_raw_clip_{i:02d}.mp4"
        cut_segment(video_path, h["start"], h["end"], raw_path)
        working = raw_path

        if args.vertical:
            print(f"[5/6] Face-tracking crop for clip {i}...")
            cropped_path = out_dir / f"_cropped_clip_{i:02d}.mp4"
            apply_face_tracked_vertical_crop(working, cropped_path)
            working.unlink()
            working = cropped_path

        final_path = out_dir / f"clip_{i:02d}_score{h['score']}.mp4"

        if args.captions:
            print(f"[6/6] Building captions for clip {i}...")
            vid_w, vid_h = probe_resolution(working)
            words = collect_words_in_clip(transcript, h["start"], h["end"])
            chunks = chunk_words_for_captions(words, h["start"], h["end"])
            ass_path = out_dir / f"_captions_{i:02d}.ass"
            write_ass_captions(chunks, ass_path, vid_w, vid_h)
            burn_captions(working, ass_path, final_path)
            working.unlink()
            ass_path.unlink()
        else:
            working.rename(final_path)

        manifest.append({**h, "file": str(final_path)})
        print(f"  -> {final_path}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. {len(highlights)} clips in {out_dir}/")


# ---------------------------------------------------------------------------
# Folder watching
# ---------------------------------------------------------------------------

def is_file_stable(path: Path, wait_sec: float = 5.0) -> bool:
    """Returns True if the file's size hasn't changed over wait_sec seconds.
    Guards against picking up a video that's still being copied/recorded
    into the watch folder."""
    try:
        size1 = path.stat().st_size
    except FileNotFoundError:
        return False
    if size1 == 0:
        return False
    time.sleep(wait_sec)
    try:
        size2 = path.stat().st_size
    except FileNotFoundError:
        return False
    return size1 == size2


def load_watch_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_watch_state(state_path: Path, state: dict):
    state_path.write_text(json.dumps(state, indent=2))


def watch_folder(watch_dir: Path, out_root: Path, args):
    """Poll watch_dir for new/changed video files. Each one gets its own
    subfolder under out_root named after the video's filename stem, so
    clips from different source videos never mix together."""
    if not watch_dir.exists():
        print(f"Watch folder not found: {watch_dir}")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "_watch_state.json"
    processed = load_watch_state(state_path)

    print(f"Watching {watch_dir} for new videos (checking every {args.poll_interval}s). Ctrl+C to stop.")
    print(f"Clips will be written to {out_root}/<video-name>/")

    try:
        while True:
            candidates = sorted(
                f for f in watch_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            for f in candidates:
                stat = f.stat()
                fingerprint = f"{stat.st_size}:{stat.st_mtime}"
                if processed.get(f.name) == fingerprint:
                    continue  # already processed this exact file

                if not is_file_stable(f):
                    print(f"  {f.name} still being written, will check again next pass...")
                    continue

                print(f"\n=== New video: {f.name} ===")
                video_out_dir = out_root / f.stem
                try:
                    process_video(f, video_out_dir, args)
                    # re-stat in case the file changed while it was processing
                    stat = f.stat()
                    processed[f.name] = f"{stat.st_size}:{stat.st_mtime}"
                    save_watch_state(state_path, processed)
                except Exception:
                    print(f"!! Failed processing {f.name}, will retry next pass:")
                    traceback.print_exc()

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local highlight clipper: Whisper + Ollama + face-tracked crop + captions")
    parser.add_argument("video", type=Path, nargs="?", help="Path to a single input video file")
    parser.add_argument("--watch", type=Path, metavar="FOLDER", help="Watch this folder for new videos instead of processing one file. Each video gets its own subfolder under --out-dir.")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between folder scans in --watch mode (default 30)")
    parser.add_argument("--clips", type=int, default=8, help="Number of clips to generate")
    parser.add_argument("--min-len", type=int, default=20, help="Minimum clip length in seconds")
    parser.add_argument("--max-len", type=int, default=180, help="Maximum clip length in seconds")
    parser.add_argument("--vertical", action="store_true", help="Crop clips to 9:16 vertical, following the speaker's face")
    parser.add_argument("--captions", action="store_true", help="Burn in auto-generated captions")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for ranking (default: qwen2.5:32b-instruct)")
    parser.add_argument("--chunk-minutes", type=int, default=15, help="Chunk size in minutes for long-video map-reduce ranking")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--out-dir", type=Path, default=Path("clips"), help="Output directory (single-file mode) or output root (watch mode, gets one subfolder per video)")
    args = parser.parse_args()

    if args.watch and args.video:
        print("Pass either a video file OR --watch FOLDER, not both.")
        sys.exit(1)
    if not args.watch and not args.video:
        print("Pass a video file, or --watch FOLDER to watch a directory for new videos.")
        sys.exit(1)

    if args.watch:
        watch_folder(args.watch, args.out_dir, args)
    else:
        if not args.video.exists():
            print(f"Video not found: {args.video}")
            sys.exit(1)
        process_video(args.video, args.out_dir, args)


if __name__ == "__main__":
    main()