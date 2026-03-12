#!/usr/bin/env python3
"""Serve the app and provide a server-side word recognition proxy.

This keeps the OpenAI API key on the server side (not exposed in the browser).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LANGUAGE_STORE_DIR = ROOT / "data" / "languages"
LANGUAGE_INDEX_PATH = LANGUAGE_STORE_DIR / "index.json"
LANGUAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")

def load_local_env() -> None:
    """Load key=value pairs from local env files into process env.

    Existing environment variables are not overwritten.
    """
    for filename in (".env.local", ".env"):
        path = ROOT / filename
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()


SUBSTITUTION_COST: dict[tuple[str, str], float] = {}  # Empty: no lexicon-based substitution costs


def normalize_russian_word(text: str) -> str:
    """Stub: returns lowercase letters only. Used by optional word-recognize API."""
    return "".join(ch for ch in str(text).lower() if ch.isalpha())


# No word LM file — Zoogle uses letter-only CNN. Empty lexicon for optional /api/word-recognize.
WORD_LEXICON: set[str] = set()
WORD_LOG_PROB: dict[str, float] = {}
WORDS_BY_LENGTH: dict[int, list[str]] = {}
WORD_LM_LAST_SOURCE = "none"


def ensure_word_lexicon_current(force: bool = False) -> None:
    """No-op: no word LM to load."""
    pass


def _safe_language_id(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if not LANGUAGE_ID_RE.fullmatch(value):
        return None
    return value


def _normalize_language_profile(raw: Any, fallback_id: str | None = None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    lang_id = _safe_language_id(raw.get("id")) or _safe_language_id(fallback_id)
    if not lang_id:
        return None

    name = str(raw.get("name", "Unnamed Language")).strip()
    if not name:
        name = "Unnamed Language"
    name = name[:120]

    samples_by_letter: dict[str, list[Any]] = {}
    raw_samples = raw.get("samplesByLetter")
    if not isinstance(raw_samples, dict):
        raw_samples = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        bucket = raw_samples.get(letter, [])
        samples_by_letter[letter] = bucket if isinstance(bucket, list) else []

    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    clean_metrics = {
        "valAcc": metrics.get("valAcc"),
        "valLoss": metrics.get("valLoss"),
        "trainedAt": metrics.get("trainedAt"),
    }

    model = raw.get("model")
    if not (model is None or isinstance(model, dict)):
        model = None

    return {
        "id": lang_id,
        "name": name,
        "samplesByLetter": samples_by_letter,
        "metrics": clean_metrics,
        "model": model,
    }


def _ensure_language_store() -> None:
    LANGUAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def load_language_profiles_payload() -> dict[str, Any]:
    _ensure_language_store()

    index_payload: dict[str, Any] = {}
    if LANGUAGE_INDEX_PATH.exists():
        try:
            raw = json.loads(LANGUAGE_INDEX_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                index_payload = raw
        except Exception:  # noqa: BLE001
            index_payload = {}

    ordered_ids: list[str] = []
    seen: set[str] = set()
    raw_ids = index_payload.get("profileIds")
    if isinstance(raw_ids, list):
        for raw_id in raw_ids:
            lang_id = _safe_language_id(raw_id)
            if not lang_id or lang_id in seen:
                continue
            seen.add(lang_id)
            ordered_ids.append(lang_id)

    for path in sorted(LANGUAGE_STORE_DIR.glob("*.json")):
        if path.name == LANGUAGE_INDEX_PATH.name:
            continue
        lang_id = _safe_language_id(path.stem)
        if not lang_id or lang_id in seen:
            continue
        seen.add(lang_id)
        ordered_ids.append(lang_id)

    profiles: list[dict[str, Any]] = []
    for lang_id in ordered_ids:
        path = LANGUAGE_STORE_DIR / f"{lang_id}.json"
        if not path.exists():
            continue
        try:
            raw_profile = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        profile = _normalize_language_profile(raw_profile, fallback_id=lang_id)
        if profile is None:
            continue
        profiles.append(profile)

    valid_ids = [profile["id"] for profile in profiles]
    active_language_id = _safe_language_id(index_payload.get("activeLanguageId"))
    if active_language_id not in valid_ids:
        active_language_id = valid_ids[0] if valid_ids else None

    return {
        "version": 2,
        "activeLanguageId": active_language_id,
        "profiles": profiles,
        "savedAt": int(time.time() * 1000),
    }


def save_language_profiles_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_language_store()
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object payload")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("Expected `profiles` to be a list")

    profiles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_profile in raw_profiles:
        profile = _normalize_language_profile(raw_profile)
        if profile is None:
            continue
        lang_id = profile["id"]
        if lang_id in seen_ids:
            continue
        seen_ids.add(lang_id)
        profiles.append(profile)

    if not profiles:
        raise ValueError("No valid language profiles to save")

    for profile in profiles:
        lang_id = profile["id"]
        path = LANGUAGE_STORE_DIR / f"{lang_id}.json"
        path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")

    keep_names = {f"{profile['id']}.json" for profile in profiles}
    for path in LANGUAGE_STORE_DIR.glob("*.json"):
        if path.name == LANGUAGE_INDEX_PATH.name:
            continue
        if path.name in keep_names:
            continue
        try:
            path.unlink()
        except OSError:
            pass

    ordered_ids = [profile["id"] for profile in profiles]
    active_language_id = _safe_language_id(payload.get("activeLanguageId"))
    if active_language_id not in ordered_ids:
        active_language_id = ordered_ids[0]

    saved_at = int(time.time() * 1000)
    index_payload = {
        "version": 1,
        "activeLanguageId": active_language_id,
        "profileIds": ordered_ids,
        "savedAt": saved_at,
    }
    LANGUAGE_INDEX_PATH.write_text(json.dumps(index_payload, ensure_ascii=False), encoding="utf-8")

    return {
        "version": 2,
        "activeLanguageId": active_language_id,
        "profiles": profiles,
        "savedAt": saved_at,
    }


def _word_prior(word: str) -> float:
    return float(WORD_LOG_PROB.get(word, -20.0))


def _coerce_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return max(0.0, min(1.0, parsed))
    return None


def _word_prior_unit(word: str) -> float:
    # Map log-probability to roughly [0,1] for weighted candidate ranking.
    logp = _word_prior(word)
    z = (logp + 11.5) / 2.3
    return 1.0 / (1.0 + math.exp(-z))


def _parse_local_hint(local_hint: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "word": "",
        "confidence": None,
        "length": None,
        "top_words": [],
        "letter_hints": [],
    }
    if not isinstance(local_hint, dict):
        return out

    hint_word = normalize_russian_word(local_hint.get("word", ""))
    hint_conf = _coerce_confidence(local_hint.get("confidence"))
    out["word"] = hint_word
    out["confidence"] = hint_conf

    raw_len = local_hint.get("length")
    if isinstance(raw_len, (int, float)):
        n = int(raw_len)
        if 1 <= n <= 64:
            out["length"] = n

    top_words_map: dict[str, float] = {}
    raw_top_words = local_hint.get("topWords")
    if not isinstance(raw_top_words, list):
        raw_top_words = local_hint.get("top_words")
    if isinstance(raw_top_words, list):
        for i, item in enumerate(raw_top_words[:12]):
            word = ""
            conf = None
            if isinstance(item, dict):
                word = normalize_russian_word(item.get("word", ""))
                conf = _coerce_confidence(item.get("confidence"))
                if conf is None:
                    conf = _coerce_confidence(item.get("score"))
            else:
                word = normalize_russian_word(str(item))
            if not word:
                continue
            if conf is None:
                conf = max(0.12, 0.62 - 0.06 * i)
            prev = top_words_map.get(word)
            if prev is None or conf > prev:
                top_words_map[word] = conf

    if hint_word:
        top_words_map[hint_word] = max(
            top_words_map.get(hint_word, 0.0),
            hint_conf if hint_conf is not None else 0.55,
        )
    out["top_words"] = sorted(top_words_map.items(), key=lambda item: item[1], reverse=True)

    raw_letter_hints = local_hint.get("letterHints")
    if not isinstance(raw_letter_hints, list):
        raw_letter_hints = local_hint.get("letter_hints")
    letter_hints: list[dict[str, float]] = []
    if isinstance(raw_letter_hints, list):
        for row in raw_letter_hints[:32]:
            bucket: dict[str, float] = {}
            if isinstance(row, list):
                for item in row[:5]:
                    if not isinstance(item, dict):
                        continue
                    ch = normalize_russian_word(item.get("letter", ""))
                    if len(ch) != 1:
                        continue
                    prob = _coerce_confidence(item.get("p"))
                    if prob is None:
                        prob = _coerce_confidence(item.get("confidence"))
                    if prob is None:
                        continue
                    bucket[ch] = max(bucket.get(ch, 0.0), prob)
            elif isinstance(row, dict):
                for key, value in row.items():
                    ch = normalize_russian_word(str(key))
                    if len(ch) != 1:
                        continue
                    prob = _coerce_confidence(value)
                    if prob is None:
                        continue
                    bucket[ch] = max(bucket.get(ch, 0.0), prob)
            if bucket:
                letter_hints.append(bucket)
    out["letter_hints"] = letter_hints

    if out["length"] is None:
        if hint_word:
            out["length"] = len(hint_word)
        elif letter_hints:
            out["length"] = len(letter_hints)
    elif hint_word and abs(int(out["length"]) - len(hint_word)) >= 3:
        out["length"] = len(hint_word)

    return out


def _letter_hint_alignment(word: str, letter_hints: list[dict[str, float]]) -> float:
    if not word:
        return 0.0
    if not letter_hints:
        return 0.5

    n = min(len(word), len(letter_hints))
    if n <= 0:
        return 0.5

    score = 0.0
    for i in range(n):
        p = letter_hints[i].get(word[i], 0.0)
        if p <= 0:
            p = 0.01
        score += p
    score /= float(n)

    length_gap = abs(len(word) - len(letter_hints))
    length_mult = max(0.45, 1.0 - 0.16 * length_gap)
    aligned = (score * 2.6) * length_mult
    return max(0.0, min(1.0, aligned))


def build_hint_lexicon_shortlist(
    hint_word: str,
    hint_top_words: list[tuple[str, float]],
    target_length: int | None,
    limit: int = 36,
) -> list[str]:
    if not WORD_LEXICON:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def push(word: str) -> bool:
        if not word:
            return False
        if word in seen:
            return False
        seen.add(word)
        out.append(word)
        return len(out) >= limit

    for word, _ in hint_top_words[:12]:
        if push(word):
            return out

    if hint_word:
        for cand in nearest_lexicon_candidates(hint_word, limit=min(16, limit)):
            if push(cand):
                return out

    lengths: list[int] = []
    if isinstance(target_length, int) and target_length > 0:
        lengths = [target_length, target_length - 1, target_length + 1]
    elif hint_word:
        n = len(hint_word)
        lengths = [n, n - 1, n + 1]

    for n in lengths:
        if n <= 0:
            continue
        for cand in WORDS_BY_LENGTH.get(n, [])[:110]:
            if push(cand):
                return out
    return out


def substitution_penalty(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return SUBSTITUTION_COST.get((a, b), 1.0)


def connected_edit_distance(a: str, b: str, max_distance: float = 3.0) -> float:
    if a == b:
        return 0.0
    if abs(len(a) - len(b)) > max_distance + 1:
        return max_distance + 1.0
    if len(a) > len(b):
        a, b = b, a

    previous = [float(i) for i in range(len(a) + 1)]
    for i, cb in enumerate(b, 1):
        current = [float(i)]
        row_min = float(i)
        for j, ca in enumerate(a, 1):
            insert_cost = current[j - 1] + 1.0
            delete_cost = previous[j] + 1.0
            replace_cost = previous[j - 1] + substitution_penalty(ca, cb)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1.0
        previous = current
    return float(previous[-1])


def nearest_lexicon_candidates(query: str, limit: int = 5) -> list[str]:
    query = normalize_russian_word(query)
    if not query or not WORD_LEXICON:
        return []
    if query in WORD_LEXICON:
        return [query]

    n = len(query)
    max_dist = 2.2 if n <= 6 else 3.0
    length_band = [n]
    if n > 1:
        length_band.append(n - 1)
    length_band.append(n + 1)
    if n >= 5:
        length_band.extend([n - 2, n + 2])

    pool: list[str] = []
    seen: set[str] = set()
    for ln in length_band:
        arr = WORDS_BY_LENGTH.get(ln, [])
        # Keep candidate pool bounded to reduce request latency.
        for word in arr[:420]:
            if word in seen:
                continue
            pool.append(word)
            seen.add(word)

    scored: list[tuple[float, float, str]] = []
    for candidate in pool:
        dist = connected_edit_distance(query, candidate, max_distance=max_dist)
        if dist > max_dist:
            continue
        scored.append((dist, -_word_prior(candidate), candidate))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [w for _, _, w in scored[: max(1, limit)]]


def enforce_real_word_reasoning(
    result: dict[str, Any],
    local_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_word = normalize_russian_word(result.get("word", ""))
    primary_conf = _coerce_confidence(result.get("confidence"))
    alternatives_in = result.get("alternatives", [])
    candidates_in = result.get("candidates", [])

    parsed_hint = _parse_local_hint(local_hint)
    hint_word = parsed_hint["word"]
    hint_conf: float | None = parsed_hint["confidence"]
    hint_top_words: list[tuple[str, float]] = parsed_hint["top_words"]
    hint_top_word_map = {word: conf for word, conf in hint_top_words}
    hint_letter_hints: list[dict[str, float]] = parsed_hint["letter_hints"]
    hint_length: int | None = parsed_hint["length"]

    candidate_rows: list[dict[str, Any]] = []

    def add_candidate(raw_word: Any, raw_conf: Any, source: str, rank: int) -> None:
        word = normalize_russian_word(str(raw_word))
        if not word:
            return
        candidate_rows.append(
            {
                "raw_word": word,
                "raw_conf": _coerce_confidence(raw_conf),
                "source": source,
                "rank": rank,
            }
        )

    if primary_word:
        add_candidate(primary_word, primary_conf, "primary", 0)

    if isinstance(candidates_in, list):
        for i, item in enumerate(candidates_in[:8]):
            if isinstance(item, dict):
                add_candidate(item.get("word", ""), item.get("confidence"), "candidate", i + 1)
            else:
                add_candidate(item, None, "candidate", i + 1)

    if isinstance(alternatives_in, list):
        for i, item in enumerate(alternatives_in[:6]):
            if isinstance(item, dict):
                add_candidate(item.get("word", ""), item.get("confidence"), "alternative", i + 1)
            else:
                add_candidate(item, None, "alternative", i + 1)

    if hint_word:
        add_candidate(hint_word, hint_conf, "hint", 0)
    for i, (hint_candidate, hint_candidate_conf) in enumerate(hint_top_words[:10]):
        add_candidate(hint_candidate, hint_candidate_conf, "hint_top", i)

    if not candidate_rows:
        raise ValueError("Missing recognized word")

    scored_by_word: dict[str, dict[str, Any]] = {}
    adjusted_any = False

    for row in candidate_rows:
        raw_word = row["raw_word"]
        source = row["source"]
        rank = int(row["rank"])
        conf = row["raw_conf"]
        if conf is None:
            if source == "primary":
                conf = 0.62
            elif source == "hint":
                conf = 0.55 if hint_conf is None else max(0.45, hint_conf)
            elif source == "hint_top":
                conf = hint_top_word_map.get(raw_word, 0.52)
            else:
                conf = 0.48

        variants: list[tuple[str, bool, float]] = [(raw_word, False, 0.0)]
        if WORD_LEXICON and raw_word not in WORD_LEXICON:
            nearest = nearest_lexicon_candidates(raw_word, limit=1)
            if nearest:
                snapped_word = nearest[0]
                snapped_distance = connected_edit_distance(raw_word, snapped_word, max_distance=3.4)
                if snapped_word != raw_word:
                    variants.append((snapped_word, True, snapped_distance))
                    adjusted_any = True

        for canonical, snapped, snapped_distance in variants:
            prior_component = 0.5
            if WORD_LEXICON:
                if canonical in WORD_LEXICON:
                    prior_component = _word_prior_unit(canonical)
                else:
                    prior_component = 0.06

            hint_bonus = 0.0
            if hint_word and canonical == hint_word:
                hint_bonus = 0.28 * (hint_conf if hint_conf is not None else 0.55)
            hint_top_bonus = 0.0
            if canonical in hint_top_word_map:
                hint_top_bonus = 0.34 * hint_top_word_map[canonical]

            letter_align_bonus = 0.0
            if hint_letter_hints:
                alignment = _letter_hint_alignment(canonical, hint_letter_hints)
                letter_align_bonus = 0.26 * (alignment - 0.34)

            rank_penalty = min(0.18, 0.03 * rank)
            if source == "hint_top":
                rank_penalty *= 0.55
            length_penalty = 0.0
            if isinstance(hint_length, int) and hint_length > 0:
                length_penalty = min(0.34, 0.075 * abs(len(canonical) - hint_length))
            snap_penalty = (0.12 + min(0.24, 0.11 * snapped_distance)) if snapped else 0.0
            score = (
                0.56 * conf
                + 0.20 * prior_component
                + hint_bonus
                + hint_top_bonus
                + letter_align_bonus
                - rank_penalty
                - length_penalty
                - snap_penalty
            )

            row_out = {
                "word": canonical,
                "confidence": conf,
                "score": score,
                "source": source,
                "snapped": snapped,
            }
            prev = scored_by_word.get(canonical)
            if prev is None or row_out["score"] > prev["score"]:
                scored_by_word[canonical] = row_out

    ranked = sorted(
        scored_by_word.values(),
        key=lambda item: (
            -item["score"],
            -_word_prior(item["word"]),
            item["word"],
        ),
    )
    if not ranked:
        raise ValueError("No valid word candidates")

    best = ranked[0]
    word = best["word"]

    out_alts: list[str] = []
    seen = {word}
    for item in ranked[1:]:
        candidate = item["word"]
        if candidate in seen:
            continue
        out_alts.append(candidate)
        seen.add(candidate)
        if len(out_alts) >= 3:
            break

    for cand in nearest_lexicon_candidates(word, limit=6):
        if cand in seen:
            continue
        out_alts.append(cand)
        seen.add(cand)
        if len(out_alts) >= 3:
            break
    if len(out_alts) < 3:
        for cand in WORDS_BY_LENGTH.get(len(word), [])[:50]:
            if cand in seen:
                continue
            out_alts.append(cand)
            seen.add(cand)
            if len(out_alts) >= 3:
                break

    best_conf = _coerce_confidence(best["confidence"])
    if best_conf is None:
        best_conf = max(0.0, min(1.0, best["score"]))
    conf_out = max(0.0, min(0.99, 0.18 + 0.82 * best_conf))
    if best.get("snapped"):
        conf_out = min(conf_out, 0.83)

    top_candidates = []
    for item in ranked[:5]:
        top_candidates.append(
            {
                "word": item["word"],
                "confidence": round(float(item["confidence"]), 4),
            }
        )

    return {
        "word": word,
        "confidence": conf_out,
        "alternatives": out_alts[:3],
        "lexicon_adjusted": bool(adjusted_any),
        "candidates": top_candidates,
    }


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None
    text = raw_text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_assistant_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for part in raw_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "output_text"} and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(raw_content or "")


def parse_word_payload(raw_content: Any) -> dict[str, Any]:
    text = _parse_assistant_content(raw_content)
    parsed = _extract_json_object(text)

    if parsed is None:
        word = normalize_russian_word(text)
        if not word:
            raise ValueError("Could not parse a Russian word from model output")
        return {
            "word": word,
            "confidence": None,
            "alternatives": [],
            "candidates": [{"word": word, "confidence": None}],
        }

    candidates: list[dict[str, Any]] = []
    seen_words: set[str] = set()

    def push_candidate(raw_word: Any, raw_conf: Any) -> None:
        word = normalize_russian_word(str(raw_word))
        if not word or word in seen_words:
            return
        candidates.append({"word": word, "confidence": _coerce_confidence(raw_conf)})
        seen_words.add(word)

    push_candidate(parsed.get("word", ""), parsed.get("confidence"))

    cand_raw = parsed.get("candidates", [])
    if isinstance(cand_raw, list):
        for item in cand_raw[:10]:
            if isinstance(item, dict):
                push_candidate(item.get("word", ""), item.get("confidence"))
            else:
                push_candidate(item, None)

    alt_raw = parsed.get("alternatives", [])
    if isinstance(alt_raw, list):
        for item in alt_raw[:8]:
            if isinstance(item, dict):
                push_candidate(item.get("word", ""), item.get("confidence"))
            else:
                push_candidate(item, None)

    if not candidates:
        raise ValueError("Model output did not include a valid Russian word")

    primary = candidates[0]
    alternatives = [item["word"] for item in candidates[1:4]]
    return {
        "word": primary["word"],
        "confidence": primary["confidence"],
        "alternatives": alternatives,
        "candidates": candidates[:8],
    }


def call_openai_word_recognizer(
    image_data_url: str,
    local_hint: dict[str, Any] | None = None,
    extra_images: list[str] | None = None,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_WORD_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    parsed_hint = _parse_local_hint(local_hint)
    hint_word = parsed_hint["word"]
    hint_conf: float | None = parsed_hint["confidence"]
    hint_top_words: list[tuple[str, float]] = parsed_hint["top_words"]
    hint_letter_hints: list[dict[str, float]] = parsed_hint["letter_hints"]
    hint_length: int | None = parsed_hint["length"]
    lexicon_shortlist = build_hint_lexicon_shortlist(
        hint_word=hint_word,
        hint_top_words=hint_top_words,
        target_length=hint_length,
        limit=36,
    )

    message_content: list[dict[str, Any]] = []
    if hint_word:
        conf_text = f"{hint_conf:.2f}" if hint_conf is not None else "n/a"
        message_content.append(
            {
                "type": "text",
                "text": (
                    f"Local Bayesian-CNN hint (weak prior): {hint_word} "
                    f"(confidence {conf_text}). Use only if stroke evidence agrees."
                ),
            }
        )
    if hint_top_words:
        top_word_text = ", ".join(f"{word}:{conf:.2f}" for word, conf in hint_top_words[:8])
        message_content.append(
            {
                "type": "text",
                "text": (
                    "Local top word candidates (from CNN+Bayes, weak priors only): "
                    f"{top_word_text}."
                ),
            }
        )
    if hint_letter_hints:
        position_rows: list[str] = []
        for i, row in enumerate(hint_letter_hints[:18], 1):
            ranked = sorted(row.items(), key=lambda item: item[1], reverse=True)[:3]
            if not ranked:
                continue
            row_text = "/".join(f"{ch}:{prob:.2f}" for ch, prob in ranked)
            position_rows.append(f"{i}={row_text}")
        if position_rows:
            message_content.append(
                {
                    "type": "text",
                    "text": (
                        "Per-position letter priors from local model: "
                        + "; ".join(position_rows)
                        + "."
                    ),
                }
            )
    if lexicon_shortlist:
        message_content.append(
            {
                "type": "text",
                "text": (
                    "Likely real-word shortlist from local dataset lexicon (prior only, not strict): "
                    + ", ".join(lexicon_shortlist)
                    + "."
                ),
            }
        )
    message_content.append(
        {
            "type": "text",
            "text": (
                "Read this handwritten Russian cursive word. Letters are connected and "
                "boundaries are ambiguous. Infer from full stroke flow and ligatures. "
                "Output strict JSON only with keys: word, confidence, alternatives, candidates. "
                "Use local priors only when image evidence is consistent. "
                "candidates must be an array of up to 8 objects {word, confidence}. "
                "All words must be lowercase Russian letters only."
            ),
        }
    )
    message_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    if isinstance(extra_images, list):
        for raw in extra_images[:2]:
            if isinstance(raw, str) and raw.startswith("data:image/") and raw != image_data_url:
                message_content.append({"type": "image_url", "image_url": {"url": raw}})

    request_payload = {
        "model": model,
        "temperature": 0.05,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert at Russian cursive OCR. "
                    "Prefer real words, avoid impossible combinations, and treat letters "
                    "as connected writing with heavy ligature overlap. "
                    "Think silently and return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": message_content,
            },
        ],
    }

    raw_body = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=raw_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    try:
        raw_content = payload["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unexpected OpenAI response format") from exc

    result = enforce_real_word_reasoning(parse_word_payload(raw_content), local_hint=local_hint)
    result["model"] = model
    result["lexicon_size"] = len(WORD_LEXICON)
    return result


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            ensure_word_lexicon_current()
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
                    "lexicon_size": len(WORD_LEXICON),
                    "word_dataset_source": WORD_LM_LAST_SOURCE,
                },
            )
            return
        if self.path in {"/api/languages", "/api/languages/"}:
            payload = load_language_profiles_payload()
            self._send_json(HTTPStatus.OK, {"ok": True, **payload})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path in {"/api/languages", "/api/languages/"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0 or length > 120_000_000:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid request body size"})
                return

            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON body"})
                return

            try:
                saved = save_language_profiles_payload(payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"Save failed: {exc}"})
                return

            self._send_json(HTTPStatus.OK, {"ok": True, **saved})
            return

        if self.path != "/api/word-recognize":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > 8_000_000:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid request body size"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON body"})
            return

        image_data = payload.get("image")
        if not isinstance(image_data, str) or not image_data.startswith("data:image/"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Expected data URL in `image`"})
            return

        ensure_word_lexicon_current()

        extra_images: list[str] = []
        raw_images = payload.get("images")
        if isinstance(raw_images, list):
            for item in raw_images[:2]:
                if isinstance(item, str) and item.startswith("data:image/"):
                    extra_images.append(item)

        hint_payload = payload.get("hint")
        local_hint: dict[str, Any] | None = None
        if isinstance(hint_payload, dict):
            local_hint = dict(hint_payload)
            if "word" in local_hint:
                local_hint["word"] = normalize_russian_word(local_hint.get("word", ""))
            conf = _coerce_confidence(local_hint.get("confidence"))
            if conf is not None:
                local_hint["confidence"] = conf
            if "length" in local_hint and isinstance(local_hint["length"], (int, float)):
                local_hint["length"] = int(local_hint["length"])
            if not any(key in local_hint for key in ("word", "topWords", "top_words", "letterHints", "letter_hints")):
                local_hint = None

        try:
            result = call_openai_word_recognizer(
                image_data,
                local_hint=local_hint,
                extra_images=extra_images,
            )
        except RuntimeError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": str(exc)})
            return
        except urllib.error.HTTPError as exc:
            msg = f"OpenAI HTTP error {exc.code}"
            self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": msg})
            return
        except urllib.error.URLError:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": "Network error contacting OpenAI"})
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": f"Word recognition failed: {exc}"})
            return

        self._send_json(HTTPStatus.OK, {"ok": True, **result})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Russian Cursive Lab app server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", default=8000, type=int, help="Port to serve on")
    args = parser.parse_args()

    ensure_word_lexicon_current(force=True)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving {ROOT} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
