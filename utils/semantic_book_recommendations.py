"""
Semantic ranking of Open Library search results using sentence embeddings (all-MiniLM-L6-v2).

Pipeline: query expansion → fetch candidates → strong pre-filter → rich book text →
normalize embeddings → cosine (dot) similarity → domain boosts → exclude off-topic classics → top 5.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Tuple

import httpx
import numpy as np

from utils.book_embedding_service import encode_texts_normalized
from utils.open_library_client import (
    OPEN_LIBRARY_SEARCH_URL,
    REQUEST_TIMEOUT,
    RECOMMENDATION_LIMIT,
    _build_cover,
    _build_link,
    _first_author,
    _normalize_title,
    fetch_recommendations,
)

logger = logging.getLogger(__name__)

# Fetch a large pool; rank down to TOP_K after filtering + embedding.
CANDIDATE_LIMIT = 50
TOP_K = RECOMMENDATION_LIMIT

# Base subject markers (any must appear in combined subjects, case-insensitive).
_BASE_SUBJECT_MARKERS = ("children", "juvenile", "kids", "stories", "fiction")

# Obvious classics to drop unless the user query clearly matches.
_EXCLUDED_TITLE_SNIPPETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("a christmas carol", ("christmas", "carol", "dickens")),
    ("gulliver", ("gulliver", "swift", "lilliput")),
)


def build_expanded_query(topic: str, age: Optional[int]) -> str:
    """
    Mandatory enriched semantic query for embedding + search (replaces raw topic).
    """
    raw = topic.strip()
    if not raw:
        return ""
    t = raw.lower()
    age_seg = f"for age {age}" if age is not None else "for children"

    if "space" in t:
        return (
            f"children books about space, universe, astronauts, science fiction adventure {age_seg}"
        )
    if "animal" in t:
        return (
            f"children books about animals, pets, wildlife, simple stories for toddlers {age_seg}"
        )
    if "kind" in t or "friend" in t or "friendship" in t:
        return (
            f"children books about kindness, friendship, helping others, emotions {age_seg}"
        )
    return f"{raw} children kids story books {age_seg}"


def _subject_list(doc: dict, max_items: int = 20) -> List[str]:
    raw = doc.get("subject")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for s in raw[:max_items]:
        if isinstance(s, str) and s.strip():
            out.append(s.strip())
    return out


def _subject_blob_lower(doc: dict) -> str:
    return " ".join(_subject_list(doc)).lower()


def _description_plain(doc: dict) -> str:
    """Extract description / first_sentence safely for rich text."""
    chunks: List[str] = []
    fs = doc.get("first_sentence")
    if isinstance(fs, list):
        for x in fs[:2]:
            if isinstance(x, str) and x.strip():
                chunks.append(x.strip())
    elif isinstance(fs, str) and fs.strip():
        chunks.append(fs.strip())

    desc = doc.get("description")
    if isinstance(desc, str) and desc.strip():
        chunks.append(desc.strip())
    elif isinstance(desc, dict):
        value = desc.get("value")
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())

    text = " ".join(chunks).strip()
    if len(text) > 600:
        text = text[:600] + "…"
    return text


def _has_valid_title(doc: dict) -> bool:
    t = doc.get("title")
    if isinstance(t, str) and t.strip():
        return True
    if isinstance(t, list) and t and isinstance(t[0], str) and t[0].strip():
        return True
    return False


def _has_valid_author(doc: dict) -> bool:
    authors = doc.get("author_name")
    if not isinstance(authors, list) or not authors:
        return False
    name = authors[0]
    return isinstance(name, str) and bool(name.strip())


def _passes_base_subject_filter(blob: str) -> bool:
    return any(m in blob for m in _BASE_SUBJECT_MARKERS)


def _passes_topic_subject_filter(topic_lower: str, blob: str) -> bool:
    if "space" in topic_lower:
        return any(x in blob for x in ("space", "science fiction", "astronomy"))
    if "animal" in topic_lower:
        return any(x in blob for x in ("animal", "pets", "wildlife"))
    if "kind" in topic_lower or "friend" in topic_lower or "friendship" in topic_lower:
        return any(x in blob for x in ("friendship", "kindness", "emotion", "social"))
    return True


def pre_filter_candidates(topic: str, docs: List[dict]) -> List[dict]:
    """
    Strong pre-filter BEFORE embedding: children's subjects + topic hooks; valid title/author.
    """
    topic_lower = topic.strip().lower()
    out: List[dict] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if not _has_valid_title(doc) or not _has_valid_author(doc):
            continue
        blob = _subject_blob_lower(doc)
        if not _passes_base_subject_filter(blob):
            continue
        if not _passes_topic_subject_filter(topic_lower, blob):
            continue
        out.append(doc)
    return out


def rich_book_text(doc: dict) -> str:
    """
    Rich document string for embedding: title + subjects + description + reinforcement.
    """
    title = _normalize_title(doc.get("title"))
    subjects = _subject_list(doc)
    subj_str = " ".join(subjects)
    desc = _description_plain(doc)
    return f"{title}. {subj_str}. {desc}. children's book about {subj_str}"


def _domain_boost(query_lower: str, book_text_lower: str, score: float) -> float:
    s = float(score)
    if "space" in query_lower and "space" in book_text_lower:
        s += 0.1
    if "animal" in query_lower and "animal" in book_text_lower:
        s += 0.1
    if (
        "kind" in query_lower
        or "friend" in query_lower
        or "kindness" in query_lower
        or "friendship" in query_lower
    ):
        if "friend" in book_text_lower or "kind" in book_text_lower:
            s += 0.1
    return s


def _reason_from_score(score: float) -> str:
    if score > 0.8:
        return "Highly relevant"
    if score > 0.65:
        return "Relevant match"
    return "Loosely related"


def _topic_allows_excluded_classic(topic_lower: str, title_lower: str) -> bool:
    """Allow classic through if query explicitly matches (per user spec)."""
    if "christmas" in topic_lower or "carol" in topic_lower:
        return True
    if "gulliver" in topic_lower or "swift" in topic_lower:
        return True
    if "travel" in topic_lower and "gulliver" in title_lower:
        return True
    return False


def _is_excluded_classic(topic_lower: str, title: str) -> bool:
    tl = title.lower()
    for snippet, keywords in _EXCLUDED_TITLE_SNIPPETS:
        if snippet in tl:
            if _topic_allows_excluded_classic(topic_lower, tl):
                return False
            return True
    return False


def _rank_semantic(
    expanded_query: str,
    topic_original: str,
    docs: List[dict],
) -> List[dict[str, Any]]:
    if not docs:
        return []

    topic_lower = topic_original.strip().lower()
    query_lower = expanded_query.lower()
    book_texts = [rich_book_text(d) for d in docs]

    q_emb = encode_texts_normalized([expanded_query])[0]
    doc_emb = encode_texts_normalized(book_texts)
    scores = doc_emb @ q_emb.reshape(-1)

    boosted = np.array(
        [_domain_boost(query_lower, book_texts[i].lower(), float(scores[i])) for i in range(len(docs))],
        dtype=np.float64,
    )

    ranked = np.argsort(boosted)[::-1]

    print("FINAL QUERY:", expanded_query)
    print("Filtered books:", len(docs))
    print("Top results with score:")
    titles = [_normalize_title(d.get("title")) for d in docs]
    for i in ranked[:5]:
        ii = int(i)
        print(titles[ii], float(boosted[ii]))

    out: List[dict[str, Any]] = []
    for idx in ranked:
        ii = int(idx)
        doc = docs[ii]
        title = _normalize_title(doc.get("title"))
        if _is_excluded_classic(topic_lower, title):
            continue
        sc = float(boosted[ii])
        item = {
            "title": title,
            "author": _first_author(doc),
            "cover": _build_cover(doc),
            "link": _build_link(doc),
            "reason": _reason_from_score(sc),
            "score": round(sc, 3),
        }
        out.append(item)
        if len(out) >= TOP_K:
            break

    return out


async def fetch_candidates_broad(
    client: httpx.AsyncClient,
    expanded_query: str,
    limit: int = CANDIDATE_LIMIT,
) -> List[dict]:
    cleaned = expanded_query.strip()
    if not cleaned:
        return []
    # Use enriched query for retrieval; bias toward discoverable juvenile hits.
    q = f"{cleaned} juvenile fiction"
    response = await client.get(
        OPEN_LIBRARY_SEARCH_URL,
        params={
            "q": q,
            "limit": limit,
            "fields": (
                "key,title,subtitle,author_name,subject,first_sentence,description,cover_i"
            ),
        },
    )
    response.raise_for_status()
    data = response.json()
    docs = data.get("docs")
    if not isinstance(docs, list):
        return []
    return [d for d in docs[:limit] if isinstance(d, dict)]


async def _keyword_fallback(topic: str, age: Optional[int]) -> List[dict[str, Any]]:
    raw = await fetch_recommendations(topic, age)
    out: List[dict[str, Any]] = []
    for item in raw:
        row = dict(item)
        row["reason"] = "Keyword fallback result"
        row["score"] = None
        out.append(row)
    return out


async def fetch_semantic_recommendations(topic: str, age: Optional[int] = None) -> List[dict[str, Any]]:
    cleaned = topic.strip()
    if not cleaned:
        return []

    expanded = build_expanded_query(cleaned, age)

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            docs = await fetch_candidates_broad(client, expanded, CANDIDATE_LIMIT)

        if not docs:
            return await _keyword_fallback(cleaned, age)

        filtered = pre_filter_candidates(cleaned, docs)

        if len(filtered) < 5:
            print("FINAL QUERY:", expanded)
            print("Filtered books:", len(filtered))
            print("Top results with score:")
            print("(skipped — using keyword fallback)")
            return await _keyword_fallback(cleaned, age)

        ranked = await asyncio.to_thread(_rank_semantic, expanded, cleaned, filtered)
        if not ranked:
            return await _keyword_fallback(cleaned, age)
        return ranked

    except httpx.HTTPStatusError as exc:
        logger.warning("Open Library HTTP error in semantic flow: %s", exc)
        try:
            return await _keyword_fallback(cleaned, age)
        except Exception:
            return []
    except httpx.RequestError as exc:
        logger.warning("Open Library network error in semantic flow: %s", exc)
        try:
            return await _keyword_fallback(cleaned, age)
        except Exception:
            return []
    except Exception as exc:
        logger.warning("Semantic recommendation failed, keyword fallback: %s", exc, exc_info=True)
        try:
            return await _keyword_fallback(cleaned, age)
        except Exception:
            return []
