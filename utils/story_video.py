"""
Build a simple slideshow video from story scenes and optional narration.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from io import BytesIO

import numpy as np
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from moviepy.video.fx.loop import loop as loop_fx
from PIL import Image

logger = logging.getLogger(__name__)

SEC_PER_SCENE = 3
OUTPUT_SIZE = (512, 512)
MAX_NARRATION_CHARS = 4500
IMAGE_PROMPT_PREFIX = (
    "Whimsical children's book illustration, colorful, friendly, no text, "
    "safe for children, scene: "
)

DEFAULT_COQUI_MODEL = "tts_models/en/vctk/vits"


def _synthesize_coqui_tts_to_wav(text: str, wav_path: str) -> None:
    """
    Generate narration audio using Coqui TTS and write a WAV file.

    Notes:
        - Requires the `TTS` package (Coqui TTS).
        - Uses CPU by default (safer for server deployments).
    """
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Coqui TTS is not installed on the server. Install it with: pip install TTS"
        ) from exc

    model_name = (os.getenv("COQUI_TTS_MODEL") or "").strip() or DEFAULT_COQUI_MODEL
    try:
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    except Exception as exc:
        logger.warning("Failed to initialize Coqui TTS model=%s: %s", model_name, exc)
        raise RuntimeError(
            "Could not initialize Coqui TTS model for narration. "
            "Check COQUI_TTS_MODEL or server dependencies."
        ) from exc

    try:
        # Coqui TTS writes audio directly to a file.
        tts.tts_to_file(text=text, file_path=wav_path)
    except Exception as exc:
        logger.warning("Coqui TTS synthesis failed: %s", exc)
        raise RuntimeError("Could not generate narration audio.") from exc

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("Narration audio generation returned an empty file.")


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


def build_story_video_file(
    story: str,
    include_voice: bool,
    image_provider: str = "gemini",
) -> str:
    """
    Create an MP4 file path for the story video. Caller should delete the file after sending.

    Raises:
        HuggingFaceError / GeminiImageError: image generation failures
        RuntimeError: moviepy / IO failures
        ValueError: invalid story
    """
    work_dir = tempfile.mkdtemp(prefix="kiddoland_story_video_")
    output_path = os.path.join(work_dir, "story_video.mp4")

    try:
        scenes = split_story_into_scenes(story)
        clips = []
        for scene in scenes:
            prompt = scene_to_image_prompt(scene)
            raw = _generate_scene_image_bytes(prompt, image_provider)
            img = Image.open(BytesIO(raw)).convert("RGB").resize(
                OUTPUT_SIZE, Image.Resampling.LANCZOS
            )
            arr = np.array(img)
            clip = ImageClip(arr).set_duration(SEC_PER_SCENE)
            clips.append(clip)

        video = concatenate_videoclips(clips, method="compose")
        video = video.set_fps(24)

        ffmpeg_params = ["-movflags", "+faststart"]

        if include_voice:
            narration_text = story.strip()[:MAX_NARRATION_CHARS]
            wav_path = os.path.join(work_dir, "narration.wav")
            try:
                _synthesize_coqui_tts_to_wav(narration_text, wav_path)
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("Coqui TTS failed: %s", exc)
                raise RuntimeError("Could not generate narration audio.") from exc

            audio = AudioFileClip(wav_path)
            try:
                vid_dur = float(video.duration)
                audio_dur = float(audio.duration)
                if audio_dur > vid_dur:
                    video = loop_fx(video, duration=audio_dur)
                elif audio_dur < vid_dur:
                    video = video.subclip(0, audio_dur)

                video = video.set_audio(audio)
                video.write_videofile(
                    output_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    audio=True,
                    temp_audiofile=os.path.join(work_dir, "temp-audio.m4a"),
                    remove_temp=True,
                    ffmpeg_params=ffmpeg_params,
                    logger=None,
                )
            finally:
                audio.close()
        else:
            video.write_videofile(
                output_path,
                fps=24,
                codec="libx264",
                audio=False,
                ffmpeg_params=ffmpeg_params,
                logger=None,
            )

        video.close()

        final_fd, final_path = tempfile.mkstemp(suffix=".mp4")
        os.close(final_fd)
        shutil.move(output_path, final_path)
        return final_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
