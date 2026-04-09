"""
Build a slideshow video from story scenes, optional narration from pre-generated TTS (base64).
"""
from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import tempfile
from io import BytesIO
from typing import Tuple

import numpy as np
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from moviepy.video.fx.loop import loop as loop_fx
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

FPS = 24
SEC_PER_SCENE = 3
OUTPUT_SIZE = (512, 512)
IMAGE_PROMPT_PREFIX = (
    "Whimsical children's book illustration, colorful, friendly, no text, "
    "safe for children, scene: "
)

# Soft pastel placeholder when image generation fails
_PLACEHOLDER_BG = (138, 180, 248)


def split_story_into_scenes(story: str) -> list[str]:
    """Split story into 3–5 scenes (paragraphs and/or sentences)."""
    text = story.strip()
    if not text:
        raise ValueError("Story is empty")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    target_min, target_max = 3, 5

    if len(paragraphs) > target_max:
        merged: list[str] = []
        n = len(paragraphs)
        for i in range(target_max):
            start = (i * n) // target_max
            end = ((i + 1) * n) // target_max
            merged.append(" ".join(paragraphs[start:end]))
        return merged

    if len(paragraphs) >= target_min:
        return paragraphs[:target_max]

    sentences: list[str] = []
    for p in paragraphs:
        for s in re.split(r"(?<=[.!?])\s+", p):
            s = s.strip()
            if s:
                sentences.append(s)
    if not sentences:
        sentences = [text]

    num_scenes = min(target_max, max(target_min, min(len(sentences), target_max)))
    while len(sentences) < num_scenes:
        sentences.append(sentences[-1])

    scenes: list[str] = []
    idx = 0
    n_sent = len(sentences)
    for i in range(num_scenes):
        remaining_scenes = num_scenes - i
        remaining_sents = n_sent - idx
        take = max(1, (remaining_sents + remaining_scenes - 1) // remaining_scenes)
        scenes.append(" ".join(sentences[idx : idx + take]))
        idx += take
    return scenes


def scene_to_image_prompt(scene: str) -> str:
    body = re.sub(r"\s+", " ", scene.strip())[:400]
    return IMAGE_PROMPT_PREFIX + body


def _generate_scene_image_bytes(prompt: str, image_provider: str) -> bytes:
    backend = (image_provider or "gemini").strip().lower()
    if backend == "gemini":
        from utils.gemini_image import generate_gemini_illustration_image

        return generate_gemini_illustration_image(prompt)
    if backend in ("huggingface", "hf"):
        from utils.huggingface_client import generate_stable_diffusion_image

        return generate_stable_diffusion_image(prompt)
    raise ValueError(f"Unknown image_provider: {image_provider}")


def _placeholder_frame(scene_index: int, scene_text: str) -> np.ndarray:
    """RGB numpy array for a missing-image scene."""
    img = Image.new("RGB", OUTPUT_SIZE, color=_PLACEHOLDER_BG)
    draw = ImageDraw.Draw(img)
    label = f"Scene {scene_index + 1}"
    snippet = re.sub(r"\s+", " ", scene_text.strip())[:120]
    if len(scene_text.strip()) > 120:
        snippet += "…"
    title_font = ImageFont.load_default()
    draw.text((24, 200), label, fill=(30, 41, 59), font=title_font)
    y = 230
    for line in _wrap_text(snippet, width=42):
        draw.text((24, y), line, fill=(51, 65, 85), font=title_font)
        y += 16
        if y > OUTPUT_SIZE[1] - 40:
            break
    return np.array(img.convert("RGB"))


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if len(trial) <= width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w] if len(w) <= width else [w[:width]]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _media_type_to_suffix(media_type: str | None) -> str:
    if not media_type:
        return ".mp3"
    mt = media_type.split(";")[0].strip().lower()
    if "wav" in mt or mt == "audio/x-wav":
        return ".wav"
    if "mpeg" in mt or "mp3" in mt:
        return ".mp3"
    if "ogg" in mt:
        return ".ogg"
    if "flac" in mt:
        return ".flac"
    if mt == "application/octet-stream":
        return ".mp3"
    return ".mp3"


def _decode_tts_to_file(
    work_dir: str,
    tts_audio_base64: str,
    tts_media_type: str | None,
) -> Tuple[str, bool]:
    """
    Write decoded narration bytes to a temp file. Returns (path, ok).
    """
    raw = (tts_audio_base64 or "").strip()
    if not raw:
        return "", False
    try:
        # Allow URL-safe base64 from some clients
        pad = (-len(raw)) % 4
        if pad:
            raw += "=" * pad
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        logger.warning("Invalid TTS base64 payload: %s", exc)
        return "", False
    if not data:
        logger.warning("TTS base64 decoded to empty bytes")
        return "", False

    suffix = _media_type_to_suffix(tts_media_type)
    path = os.path.join(work_dir, f"narration{suffix}")
    try:
        with open(path, "wb") as f:
            f.write(data)
    except OSError as exc:
        logger.warning("Could not write TTS temp file: %s", exc)
        return "", False
    return path, True


def _build_slideshow_clips(story: str, image_provider: str) -> list[ImageClip]:
    scenes = split_story_into_scenes(story)
    clips: list[ImageClip] = []
    for i, scene in enumerate(scenes):
        prompt = scene_to_image_prompt(scene)
        try:
            raw = _generate_scene_image_bytes(prompt, image_provider)
            img = Image.open(BytesIO(raw)).convert("RGB").resize(
                OUTPUT_SIZE, Image.Resampling.LANCZOS
            )
            arr = np.array(img)
        except Exception as exc:
            logger.warning(
                "Scene %s image generation failed (%s); using placeholder",
                i + 1,
                exc,
            )
            arr = _placeholder_frame(i, scene)
        clip = ImageClip(arr).set_duration(SEC_PER_SCENE)
        clips.append(clip)
    return clips


def build_story_video_file(
    story: str,
    include_voice: bool,
    image_provider: str = "gemini",
    tts_audio_base64: str | None = None,
    tts_media_type: str | None = None,
) -> str:
    """
    Create an MP4 file path for the story video. Caller should delete the file after sending.

    When ``include_voice`` is True and valid ``tts_audio_base64`` is provided, narration is
    decoded to a temp file and muxed with the slideshow. If TTS is missing or invalid,
    the video is exported without an audio track.

    Raises:
        HuggingFaceError / GeminiImageError: only if all scenes fail is avoided — failures use placeholders
        RuntimeError: moviepy / IO failures
        ValueError: invalid story
    """
    work_dir = tempfile.mkdtemp(prefix="kiddoland_story_video_")
    output_path = os.path.join(work_dir, "story_video.mp4")
    clips: list[ImageClip] = []
    base_video = None
    final_video = None
    audio_clip = None

    try:
        clips = _build_slideshow_clips(story, image_provider)
        base_video = concatenate_videoclips(clips, method="compose")
        base_video = base_video.set_fps(FPS)

        ffmpeg_params = ["-movflags", "+faststart"]
        want_narration = bool(
            include_voice
            and tts_audio_base64
            and str(tts_audio_base64).strip()
        )
        audio_path = ""
        if want_narration:
            audio_path, ok = _decode_tts_to_file(
                work_dir,
                str(tts_audio_base64).strip(),
                (tts_media_type or "").strip() or None,
            )
            want_narration = ok and bool(audio_path) and os.path.getsize(audio_path) > 0

        narration_exported = False
        if want_narration:
            try:
                audio_clip = AudioFileClip(audio_path)
            except Exception as exc:
                logger.warning("Could not load narration audio with MoviePy: %s", exc)
                audio_clip = None

        if want_narration and audio_clip is not None:
            try:
                vid_dur = float(base_video.duration or 0)
                audio_dur = float(audio_clip.duration or 0)
                if audio_dur > 0 and vid_dur > 0:
                    if audio_dur > vid_dur:
                        vtimed = loop_fx(base_video, duration=audio_dur)
                    elif audio_dur < vid_dur:
                        vtimed = base_video.subclip(0, audio_dur)
                    else:
                        vtimed = base_video
                    final_video = vtimed.set_audio(audio_clip)
                    final_video.write_videofile(
                        output_path,
                        fps=FPS,
                        codec="libx264",
                        audio_codec="aac",
                        audio=True,
                        temp_audiofile=os.path.join(work_dir, "temp-audio.m4a"),
                        remove_temp=True,
                        ffmpeg_params=ffmpeg_params,
                        logger=None,
                    )
                    narration_exported = True
            except Exception as exc:
                logger.warning("Narration video export failed: %s", exc)

        if audio_clip is not None and not narration_exported:
            try:
                audio_clip.close()
            except Exception:
                pass
            audio_clip = None

        if not narration_exported:
            base_video.write_videofile(
                output_path,
                fps=FPS,
                codec="libx264",
                audio=False,
                ffmpeg_params=ffmpeg_params,
                logger=None,
            )

        for clip in (final_video, base_video, audio_clip):
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass
        final_video = None
        audio_clip = None
        base_video = None

        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        clips = []

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Video export produced an empty or missing file.")

        final_fd, final_path = tempfile.mkstemp(suffix=".mp4")
        os.close(final_fd)
        shutil.move(output_path, final_path)
        return final_path
    finally:
        for clip in (final_video, base_video, audio_clip):
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)
