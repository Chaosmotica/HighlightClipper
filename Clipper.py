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
  4. If --vertical: detect faces with MediaPipe, pick whichever face is
     actually talking (mouth motion during transcript-active speech), and
     render a 9:16 crop that hard-cuts to that speaker -- a multi-cam-style
     edit, not a panning camera.
  5. If --captions: build short caption chunks from Whisper's word-level
     timestamps and burn them in as styled subtitles.

Requirements:
  pip install faster-whisper requests numpy mediapipe "opencv-contrib-python==5.0.0.93"
  MediaPipe face detection model (see models/ -- _get_face_detector() prints
  the download command if missing)

  Ollama running locally with a model pulled, e.g.:
      ollama pull qwen2.5:32b-instruct
  (32b gives noticeably better ranking judgment than 14b and fits
   comfortably on a 24GB 4090; drop to 14b with --model if it's too slow.)

Usage:
  python highlight_clipper.py input.mp4 --clips 8 --vertical --captions
"""

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests


def _register_cuda_dll_dirs():
    """faster-whisper's CUDA backend (ctranslate2) needs cuDNN/cuBLAS DLLs on the
    loader search path. The nvidia-cudnn-cu12 / nvidia-cublas-cu12 pip packages
    ship them but, unlike on Linux, don't register with Windows' DLL search path
    automatically -- so without this, GPU transcription fails with a missing
    cudnn_ops_infer64_8.dll error even though the package is installed."""
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cudnn", "nvidia.cublas"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            dll_dir = os.path.join(spec.submodule_search_locations[0], "bin")
            if os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)


_register_cuda_dll_dirs()

from faster_whisper import WhisperModel

# Force line-buffered stdout even when redirected to a file/log, so progress
# prints show up immediately instead of sitting in a buffer that gets lost
# if the process dies before a clean exit.
sys.stdout.reconfigure(line_buffering=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:32b-instruct"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".webm", ".m4v", ".flv"}


# ---------------------------------------------------------------------------
# Transcription (with word-level timestamps for captions)
# ---------------------------------------------------------------------------

def transcribe(video_path: Path, model_size: str = "large-v3"):
    """Transcribe video with faster-whisper on GPU. Returns segments with
    start/end/text plus a per-word timestamp list for caption generation.

    Also returns the WhisperModel itself -- the caller must keep a reference
    to it alive until all output has been written to disk. Freeing this
    model (garbage collection / ctranslate2 destructor) after a long GPU
    transcription reliably crashes the process with an access violation on
    this box. If `model` is allowed to go out of scope here, that crash
    happens immediately, before transcript.json is ever written. Deferring
    the drop until after everything is saved makes the (apparently
    unavoidable) crash harmless instead of catastrophic."""
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
    return transcript, model


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

_FACE_DETECTOR = None
_FACE_DETECTOR_MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_full_range.tflite"
_FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)

# Fixed pixel size for the mouth-motion analysis window -- deliberately NOT
# scaled to each face's bounding-box size. An earlier version sized the
# mouth ROI relative to the detected face box, which meant a closer/bigger
# face produced proportionally more raw pixel change for the exact same
# physical lip movement, silently biasing "who's moving more" back toward
# "who's biggest" -- the same problem this was supposed to fix. A fixed
# window anchored on the actual mouth keypoint (not a guessed rectangle)
# gives every candidate face a directly comparable motion score.
_MOUTH_PATCH_HALF_W = 26
_MOUTH_PATCH_HALF_H = 16


def _get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        if not _FACE_DETECTOR_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing face detection model: {_FACE_DETECTOR_MODEL_PATH}\n"
                f"Download it with:\n"
                f'  curl -L -o "{_FACE_DETECTOR_MODEL_PATH}" "{_FACE_DETECTOR_MODEL_URL}"'
            )
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision
        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(_FACE_DETECTOR_MODEL_PATH))
        options = mp_tasks_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.55)
        _FACE_DETECTOR = mp_tasks_vision.FaceDetector.create_from_options(options)
    return _FACE_DETECTOR


def _detect_faces(bgr_frame):
    """Detect faces with MediaPipe's BlazeFace (full-range variant, tuned for
    medium-distance shots like a two-person interview) instead of a Haar
    cascade. Haar cascade was unreliable on real footage here -- it threw
    3-5 "face" detections in shots with only 2 real people, and its boxes
    are too loose/inconsistent to derive an accurate mouth position from.
    BlazeFace gives far fewer false positives and includes a real mouth
    keypoint per face (index 3 in its 6-keypoint output: right eye, left
    eye, nose tip, mouth center, right ear, left ear).

    Returns a list of dicts with center_x (for position tracking), mouth_x/
    mouth_y (for motion scoring), and area (for size-based fallback), all in
    full-resolution pixel coordinates.
    """
    detector = _get_face_detector()
    h, w = bgr_frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)
    faces = []
    for d in result.detections:
        bbox = d.bounding_box
        mouth_kp = d.keypoints[3]
        faces.append({
            "center_x": bbox.origin_x + bbox.width / 2,
            "mouth_x": mouth_kp.x * w,
            "mouth_y": mouth_kp.y * h,
            "area": bbox.width * bbox.height,
        })
    return faces


def _mouth_motion_score(gray_frames, mouth_x, mouth_y, width, height):
    """Average frame-to-frame pixel difference within a fixed-size window
    centered on a face's mouth keypoint, across a short buffer of recent
    frames -- higher means more mouth movement, used as a cheap proxy for
    "this is the person talking"."""
    mx0 = max(0, int(mouth_x - _MOUTH_PATCH_HALF_W))
    mx1 = min(width, int(mouth_x + _MOUTH_PATCH_HALF_W))
    my0 = max(0, int(mouth_y - _MOUTH_PATCH_HALF_H))
    my1 = min(height, int(mouth_y + _MOUTH_PATCH_HALF_H))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    rois = [g[my0:my1, mx0:mx1] for g in gray_frames]
    if len(rois) < 2 or any(r.shape != rois[0].shape for r in rois):
        return 0.0
    diffs = [float(cv2.absdiff(rois[i], rois[i - 1]).mean()) for i in range(1, len(rois))]
    return sum(diffs) / len(diffs)


def _is_speech_active(t_sec, speech_intervals, pad=0.2):
    for start, end in speech_intervals:
        if start - pad <= t_sec <= end + pad:
            return True
    return False


def detect_face_centers(video_path: Path, sample_every_sec: float = 0.5,
                         switch_dwell_sec: float = 0.75, position_match_tol: float = 0.15,
                         speech_intervals=None, motion_buffer_frames: int = 6):
    """Sample face positions with a sticky, speech-aware active-speaker heuristic.

    Naively re-picking whichever face is largest in each individual sampled
    frame makes the crop whip back and forth in two-shots: tiny per-frame
    differences (a head turn, a detection wobble) are enough to flip which
    face is "largest", even though the same person keeps talking.

    Two layers fix this:
      1. Speech-aware selection: when speech_intervals (word-level timestamps
         from the transcript, clip-relative) is provided and multiple faces
         are detected during an active-speech moment, the candidate face is
         picked by mouth motion (frame-to-frame pixel change in a fixed-size
         window anchored on that face's actual mouth keypoint) instead of by
         size -- a real proxy for "who is actually talking", not just who's
         biggest or closest to camera. Falls back to largest-face when
         there's no speech signal, only one face, or the motion difference
         between faces is too small to trust.
      2. Dwell/hysteresis: a different face only becomes the tracked subject
         once it's won selection for switch_dwell_sec seconds straight (not
         just one sample) -- filters single-frame detection noise without
         requiring a long, cautious hold, since the output is a hard cut
         (build_stepped_centers) rather than a pan: once the tracker is
         confident who's talking, cutting to them immediately is the point,
         not something to be smoothed away.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_interval_frames = max(1, int(fps * sample_every_sec))
    min_dwell_samples = max(1, round(switch_dwell_sec / sample_every_sec))

    samples = []
    frame_idx = 0
    active_center = 0.5
    candidate_center = None
    candidate_streak = 0
    recent_gray = deque(maxlen=motion_buffer_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if speech_intervals is not None:
            recent_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if frame_idx % sample_interval_frames == 0:
            faces = _detect_faces(frame)

            detected_center = None
            if (len(faces) >= 2 and speech_intervals is not None and len(recent_gray) >= 2
                    and _is_speech_active(frame_idx / fps, speech_intervals)):
                scores = [_mouth_motion_score(recent_gray, f["mouth_x"], f["mouth_y"], width, height)
                          for f in faces]
                best = max(range(len(scores)), key=lambda i: scores[i])
                runner_up = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
                # Only trust the motion signal when one face is clearly
                # moving more than the others -- a marginal difference is
                # more likely camera/compression noise than a real cue.
                if scores[best] > 0 and scores[best] > 1.15 * runner_up:
                    detected_center = faces[best]["center_x"] / width

            if detected_center is None and len(faces) > 0:
                largest = max(faces, key=lambda f: f["area"])
                detected_center = largest["center_x"] / width

            if detected_center is not None:
                if abs(detected_center - active_center) <= position_match_tol:
                    # Same face we're already tracking. Deliberately do NOT
                    # move active_center to this frame's raw detection --
                    # face detection jitters a little from sample to sample
                    # even for a person sitting still, and with a hard-cut
                    # render (build_stepped_centers, no smoothing to hide
                    # it) that jitter would show up as small, jarring jumps
                    # while the same speaker keeps talking. The crop stays
                    # exactly put until a real speaker change is confirmed
                    # below.
                    candidate_center = None
                    candidate_streak = 0
                else:
                    # A different face won selection. Only switch to it
                    # once it's won selection for min_dwell_samples samples
                    # in a row.
                    if candidate_center is not None and abs(detected_center - candidate_center) <= position_match_tol:
                        candidate_center = detected_center
                        candidate_streak += 1
                    else:
                        candidate_center = detected_center
                        candidate_streak = 1

                    if candidate_streak >= min_dwell_samples:
                        active_center = candidate_center
                        candidate_center = None
                        candidate_streak = 0
            # If no face was detected this sample, keep the last active
            # center and leave any pending candidate switch as-is.
            samples.append((frame_idx, active_center))
        frame_idx += 1

    total_frames = frame_idx
    cap.release()
    return samples, total_frames, fps, width


def build_smoothed_centers(samples, total_frames, alpha=0.12):
    """Continuous pan: interpolate + exponentially smooth between samples,
    so the crop glides from one position to the next."""
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


def build_stepped_centers(samples, total_frames):
    """Hard cut: hold each sample's position steady until the next sample
    picks a different one, then jump straight there -- no panning glide.
    Matches a multi-cam-style edit that cuts directly to whoever the
    speech-aware tracker in detect_face_centers says is talking, rather than
    physically panning the camera across to them."""
    if not samples:
        return np.full(max(total_frames, 1), 0.5)
    sample_frames = np.array([s[0] for s in samples])
    sample_vals = np.array([s[1] for s in samples])
    all_frames = np.arange(total_frames)
    # For each frame, use the value of the most recent sample at or before
    # it (zero-order hold) instead of interpolating between samples.
    idx = np.searchsorted(sample_frames, all_frames, side="right") - 1
    idx = np.clip(idx, 0, len(sample_vals) - 1)
    return sample_vals[idx]


def apply_face_tracked_vertical_crop(input_path: Path, output_path: Path, out_w: int = 1080, out_h: int = 1920,
                                      speech_intervals=None):
    cap = cv2.VideoCapture(str(input_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    crop_w = min(orig_w, int(orig_h * 9 / 16))

    print("    Detecting faces across clip...")
    samples, total_frames, _, _ = detect_face_centers(input_path, speech_intervals=speech_intervals)
    centers = build_stepped_centers(samples, total_frames)

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
    for one video, writing all outputs into out_dir. Every output filename
    (clips, transcript, manifest) is prefixed with video_path's stem, so
    multiple videos can share the same out_dir without colliding -- this
    matters for --watch, where every processed video now lands directly in
    out_root instead of a per-video subfolder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # Keep _whisper_model referenced for the rest of this function -- see the
    # docstring in transcribe() for why it can't be allowed to go out of
    # scope until everything below has been written to disk.
    transcript, _whisper_model = transcribe(video_path, args.whisper_model)
    transcript_path = out_dir / f"{stem}_transcript.json"
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
            # Clip-relative word timings double as an active-speaker cue:
            # detect_face_centers only trusts mouth-motion over face size
            # during moments this transcript says someone is actually
            # talking. Requires the MediaPipe-based detector (see
            # _detect_faces) -- an earlier Haar-cascade-based version of
            # this same idea picked the wrong speaker most of the time due
            # to false-positive face boxes and a crude guessed mouth ROI.
            clip_words = collect_words_in_clip(transcript, h["start"], h["end"])
            speech_intervals = [(w["start"] - h["start"], w["end"] - h["start"]) for w in clip_words]
            apply_face_tracked_vertical_crop(working, cropped_path, speech_intervals=speech_intervals)
            working.unlink()
            working = cropped_path

        final_path = out_dir / f"{stem}_clip_{i:02d}_score{h['score']}.mp4"

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

    (out_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
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
    """Scan watch_dir for new/changed video files and process each one.
    Every video's clips land directly in out_root with the source
    filename's stem as a prefix, so nothing needs a per-video subfolder to
    stay unambiguous, and outputs from different source videos can share
    out_root without colliding.

    With args.once, this runs a single scan-and-process pass then returns --
    meant to be invoked by an external scheduler (e.g. a daily Task
    Scheduler job) rather than left running. Without it, this polls forever
    every args.poll_interval seconds until Ctrl+C, for watching a drop
    folder interactively."""
    if not watch_dir.exists():
        print(f"Watch folder not found: {watch_dir}")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "_watch_state.json"
    processed = load_watch_state(state_path)

    if args.once:
        print(f"Scanning {watch_dir} for new videos (single pass)...")
    else:
        print(f"Watching {watch_dir} for new videos (checking every {args.poll_interval}s). Ctrl+C to stop.")
    print(f"Clips will be written to {out_root}/ (named <video-name>_clip_NN_scoreNN.mp4)")

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
                # Processed in a child process, not in-process. transcribe()
                # deliberately keeps the CUDA Whisper model alive until
                # process_video() returns (see its docstring) so the known
                # crash-on-model-teardown happens after output is written --
                # but in-process, that crash would happen the instant
                # process_video() returns control here, killing the watcher
                # before this file could be marked processed (and, without
                # --once, taking the whole continuous watcher down with it).
                # A subprocess crashing on exit only kills the subprocess.
                manifest_path = out_root / f"{f.stem}_manifest.json"
                cmd = [
                    sys.executable, str(Path(__file__).resolve()), str(f),
                    "--out-dir", str(out_root),
                    "--clips", str(args.clips),
                    "--min-len", str(args.min_len),
                    "--max-len", str(args.max_len),
                    "--model", args.model,
                    "--chunk-minutes", str(args.chunk_minutes),
                    "--whisper-model", args.whisper_model,
                ]
                if args.vertical:
                    cmd.append("--vertical")
                if args.captions:
                    cmd.append("--captions")

                subprocess.run(cmd)
                # Ground truth is the manifest file, not the exit code -- the
                # known crash-on-teardown can give a nonzero/crash-looking
                # exit code on a run that otherwise completed successfully.
                if manifest_path.exists():
                    stat = f.stat()
                    processed[f.name] = f"{stat.st_size}:{stat.st_mtime}"
                    save_watch_state(state_path, processed)
                    print(f"  -> {f.name} processed successfully.")
                else:
                    print(f"!! {f.name} did not produce {manifest_path.name}, will retry next pass.")

            if args.once:
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local highlight clipper: Whisper + Ollama + face-tracked crop + captions")
    parser.add_argument("video", type=Path, nargs="?", help="Path to a single input video file")
    parser.add_argument("--watch", type=Path, metavar="FOLDER", help="Watch this folder for new videos instead of processing one file. Clips land directly in --out-dir, prefixed with each video's filename.")
    parser.add_argument("--once", action="store_true", help="With --watch, scan for new videos once and exit instead of polling forever. For invoking from an external scheduler (e.g. a daily Task Scheduler job) rather than leaving the process running.")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between folder scans in --watch mode (default 30, ignored with --once)")
    parser.add_argument("--clips", type=int, default=8, help="Number of clips to generate")
    parser.add_argument("--min-len", type=int, default=30, help="Minimum clip length in seconds")
    parser.add_argument("--max-len", type=int, default=180, help="Maximum clip length in seconds")
    parser.add_argument("--vertical", action="store_true", help="Crop clips to 9:16 vertical, following the speaker's face")
    parser.add_argument("--captions", action="store_true", help="Burn in auto-generated captions")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for ranking (default: qwen2.5:32b-instruct)")
    parser.add_argument("--chunk-minutes", type=int, default=15, help="Chunk size in minutes for long-video map-reduce ranking")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--out-dir", type=Path, default=Path("clips"), help="Output directory. In --watch mode every video's clips land here directly, prefixed with the source filename.")
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