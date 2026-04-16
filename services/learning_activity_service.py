"""
Learning activity generation: prompt building, HF completion, JSON parse/validate, safety.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from schemas.activity import ActivityGenerateRequest, ActivityQuizData
from utils.huggingface_client import HuggingFaceError, sample_completion_activity
from utils.safety_filter import is_content_safe

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3


@dataclass
class ActivityGenerationResult:
    success: bool
    data: ActivityQuizData | None = None
    error: str | None = None


def _difficulty_guidance(difficulty: str) -> str:
    d = (difficulty or "medium").lower().strip()
    if d == "easy":
        return (
            "Difficulty: EASY — Use very simple words, obvious correct answers, "
            "and one clear best choice per question. Keep sentences short."
        )
    if d == "hard":
        return (
            "Difficulty: HARD — Still kid-appropriate, but use slightly trickier "
            "distractions in wrong answers and questions that need a bit more thinking."
        )
    return (
        "Difficulty: MEDIUM — Balanced challenge: plausible wrong answers, "
        "clear prompts, suitable for the age band."
    )


def _learning_goal_guidance(learning_goal: str) -> str:
    """Return extra prompt instructions tailored to the requested learning goal."""
    goal = learning_goal.lower().strip()

    if any(k in goal for k in ("count", "number", "math", "addition", "subtraction", "arithmetic")):
        return (
            "Learning goal is COUNTING / NUMBERS: "
            "Every question must be a short word-problem or counting scenario set inside the theme — "
            "NOT a trivia fact (e.g. do NOT ask 'how many moons does Saturn have'). "
            "Instead write scenarios like: 'There are 4 stars on the left and 3 on the right. "
            "How many stars are there in total?' "
            "Answer options MUST be plain numbers (e.g. '3', '7', '12'). "
            "Keep numbers small and age-appropriate. "
            "Vary the operations: some questions can be addition, some can be simple subtraction or comparison."
        )

    if any(k in goal for k in ("vocab", "word", "spelling", "language", "letters")):
        return (
            "Learning goal is VOCABULARY / WORDS: "
            "Ask about word meanings, correct naming of things in the theme, completing sentences, "
            "or identifying which word fits a description. "
            "Options should be short words or phrases, not full sentences."
        )

    if any(k in goal for k in ("science", "nature", "biology", "physics", "chemistry", "planet", "animal")):
        return (
            "Learning goal is SCIENCE FACTS: "
            "Ask simple, factual science questions related to the theme. "
            "Each question should have one clearly correct answer a child can learn from. "
            "Keep explanations friendly and educational."
        )

    if any(k in goal for k in ("geo", "country", "continent", "map", "capital", "place")):
        return (
            "Learning goal is GEOGRAPHY: "
            "Ask about places, countries, continents, or simple map facts connected to the theme. "
            "Options should be place names or short geographic facts."
        )

    if any(k in goal for k in ("read", "comprehension", "story", "sentence", "paragraph")):
        return (
            "Learning goal is READING COMPREHENSION: "
            "Each question should give a short 1-2 sentence mini-passage and then ask a question about it. "
            "Test understanding, not memorization."
        )

    if any(k in goal for k in ("color", "colour", "shape", "pattern", "size")):
        return (
            "Learning goal is SHAPES / COLORS: "
            "Ask questions about identifying shapes, colors, or visual patterns described in words. "
            "Keep descriptions vivid and imaginative within the theme."
        )

    if any(k in goal for k in ("emotion", "feeling", "social", "friend", "kind", "share", "team")):
        return (
            "Learning goal is SOCIAL / EMOTIONAL: "
            "Ask questions about feelings, how to be a good friend, problem-solving in social situations, "
            "or what the right kind action would be in a described scenario."
        )

    # Generic fallback — no special shaping needed
    return ""


def build_activity_prompt(request: ActivityGenerateRequest) -> str:
    difficulty_key = request.difficulty or "medium"
    difficulty_line = _difficulty_guidance(difficulty_key)
    goal_line = _learning_goal_guidance(request.learning_goal)
    goal_section = f"\n{goal_line}\n" if goal_line else ""
    return f"""You are a children's learning activity generator. Generate a fun quiz activity for a child aged {request.age_band} about {request.theme}, focusing on {request.learning_goal}.

Difficulty level: {difficulty_key}.
{difficulty_line}
{goal_section}
Return ONLY valid JSON with this exact structure (no markdown, no explanation, no code fences):
{{
  "title": "Activity title",
  "questions": [
    {{
      "prompt": "Question text",
      "options": ["Option A", "Option B", "Option C"],
      "correct_index": 0,
      "feedback_correct": "Positive feedback",
      "feedback_incorrect": "Correct answer explanation"
    }}
  ]
}}

Rules:
- Generate exactly 5 questions.
- Use simple vocabulary appropriate for the age band.
- Each question must have exactly 2 or 3 options (not 1, not 4+).
- correct_index is 0-based and must point to the correct option.
- Content must be safe, positive, and educational only.
- No harmful, scary, or inappropriate content.
"""


def _extract_json_object_text(raw: str) -> str | None:
    """Pull a JSON object from model output (handles ```json fences)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = text[start : end + 1]
    return text


def _parse_activity_json(raw: str) -> dict[str, Any] | None:
    try:
        blob = _extract_json_object_text(raw)
        if not blob:
            return None
        data = json.loads(blob)
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _activity_text_for_safety(data: dict[str, Any]) -> str:
    """Flatten activity dict for keyword safety check."""
    parts: list[str] = [str(data.get("title", ""))]
    for q in data.get("questions") or []:
        if not isinstance(q, dict):
            continue
        parts.append(str(q.get("prompt", "")))
        for opt in q.get("options") or []:
            parts.append(str(opt))
        parts.append(str(q.get("feedback_correct", "")))
        parts.append(str(q.get("feedback_incorrect", "")))
    return "\n".join(parts)


def _shuffle_activity_payload(activity: dict[str, Any]) -> None:
    """
    Randomize question order and option order per question.
    Updates correct_index so it still points to the same answer text.
    """
    questions = activity.get("questions")
    if not isinstance(questions, list):
        return

    random.shuffle(questions)

    for q in questions:
        if not isinstance(q, dict):
            continue
        options = q.get("options")
        if not isinstance(options, list) or len(options) < 2:
            continue
        correct = int(q.get("correct_index", 0))

        paired = list(enumerate(options))
        random.shuffle(paired)
        q["options"] = [opt for _, opt in paired]

        for new_index, (old_index, _) in enumerate(paired):
            if old_index == correct:
                q["correct_index"] = new_index
                break


async def generate_learning_activity(request: ActivityGenerateRequest) -> ActivityGenerationResult:
    """
    Call HF model (stochastic sampling), parse JSON, validate, safety-check,
    then shuffle questions/options for game UX.

    Retries up to MAX_GENERATION_ATTEMPTS on parse/validation/safety/HF failure.
    """
    try:
        return await _generate_learning_activity_inner(request)
    except Exception as exc:
        logger.exception("Learning activity generation failed unexpectedly: %s", exc)
        return ActivityGenerationResult(
            success=False,
            error="Failed to generate activity",
        )


async def _generate_learning_activity_inner(
    request: ActivityGenerateRequest,
) -> ActivityGenerationResult:
    prompt = build_activity_prompt(request)
    last_error = "Failed to generate activity"

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            raw = await asyncio.to_thread(sample_completion_activity, prompt)
        except HuggingFaceError as exc:
            logger.warning(
                "Attempt %s/%s failed due to Hugging Face error: %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                exc,
            )
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Retrying due to Hugging Face error")
            last_error = "Failed to generate activity"
            continue
        except Exception as exc:
            logger.warning(
                "Attempt %s/%s failed due to unexpected error during AI call: %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                exc,
            )
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Retrying due to AI call error")
            last_error = "Failed to generate activity"
            continue

        parsed = _parse_activity_json(raw)
        if parsed is None:
            logger.warning(
                "Attempt %s/%s failed due to invalid JSON",
                attempt,
                MAX_GENERATION_ATTEMPTS,
            )
            snippet = (raw[:400] + "…") if len(raw) > 400 else raw
            logger.debug("Model output snippet: %s", snippet)
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Retrying due to invalid JSON")
            last_error = "Failed to generate activity"
            continue

        try:
            validated = ActivityQuizData.model_validate(parsed)
        except ValidationError as exc:
            logger.warning(
                "Attempt %s/%s failed due to failed validation: %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                exc,
            )
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Retrying due to failed validation")
            last_error = "Failed to generate activity"
            continue

        safe_blob = _activity_text_for_safety(validated.model_dump())
        if not is_content_safe(safe_blob):
            logger.warning(
                "Attempt %s/%s failed due to unsafe content",
                attempt,
                MAX_GENERATION_ATTEMPTS,
            )
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Retrying due to unsafe content")
            last_error = "Failed to generate activity"
            continue

        try:
            payload = validated.model_dump()
            _shuffle_activity_payload(payload)
            shuffled = ActivityQuizData.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "Post-shuffle validation failed (attempt %s/%s): %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                exc,
            )
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Retrying due to post-shuffle validation failure")
            last_error = "Failed to generate activity"
            continue

        logger.info("Learning activity generated and shuffled successfully on attempt %s", attempt)
        return ActivityGenerationResult(success=True, data=shuffled)

    return ActivityGenerationResult(success=False, error=last_error)
