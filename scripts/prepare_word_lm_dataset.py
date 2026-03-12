#!/usr/bin/env python3
"""
Build a compact Russian word language model from a labeled word-image CSV dataset.

Input expected:
  - Train.csv with columns: Filename, Actual word

Output:
  - JSON payload with:
    * start/bigram/end log probabilities over Russian letters
    * word prior log-probabilities
    * vocabulary summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RUSSIAN_LETTERS = [
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ё",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]

LETTER_TO_IDX = {ch: i for i, ch in enumerate(RUSSIAN_LETTERS)}
LETTER_SET = set(RUSSIAN_LETTERS)


def normalize_word(raw: str) -> str:
    """
    Lowercase and keep only Russian letters included in model alphabet.
    Hyphens/spaces/punctuation are removed to stay compatible with letter decoder.
    """
    text = raw.strip().lower().replace("ё", "ё")
    cleaned = [ch for ch in text if ch in LETTER_SET]
    return "".join(cleaned)


def smooth_log_prob(count: int, total: int, k: float, num_classes: int) -> float:
    return math.log((count + k) / (total + k * num_classes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="/Users/arthurilyasov/Downloads/archive (1)/Train.csv",
        help="Path to Train.csv",
    )
    parser.add_argument(
        "--out",
        default="/Users/arthurilyasov/cs109Project/data/russian_words_lm.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.45,
        help="Add-k smoothing value for letter transitions.",
    )
    parser.add_argument(
        "--word-smoothing",
        type=float,
        default=0.12,
        help="Add-k smoothing value for word priors.",
    )
    args = parser.parse_args()

    train_csv = Path(args.train_csv).expanduser()
    out_path = Path(args.out).expanduser()
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    n_letters = len(RUSSIAN_LETTERS)
    start_counts = [0] * n_letters
    end_counts = [0] * n_letters
    bigram_counts = [[0] * n_letters for _ in range(n_letters)]
    unigram_counts = [0] * n_letters
    word_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()

    total_words = 0
    dropped_words = 0

    with train_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("Actual word", "")
            word = normalize_word(raw)
            if len(word) < 2:
                dropped_words += 1
                continue

            total_words += 1
            word_counts[word] += 1
            length_counts[len(word)] += 1

            first = LETTER_TO_IDX[word[0]]
            last = LETTER_TO_IDX[word[-1]]
            start_counts[first] += 1
            end_counts[last] += 1

            prev = LETTER_TO_IDX[word[0]]
            unigram_counts[prev] += 1
            for ch in word[1:]:
                curr = LETTER_TO_IDX[ch]
                unigram_counts[curr] += 1
                bigram_counts[prev][curr] += 1
                prev = curr

    if total_words == 0:
        raise RuntimeError("No valid words found after normalization.")

    k = args.smoothing
    start_log = [
        smooth_log_prob(start_counts[i], total_words, k, n_letters) for i in range(n_letters)
    ]
    end_log = [
        smooth_log_prob(end_counts[i], total_words, k, n_letters) for i in range(n_letters)
    ]

    bigram_log = []
    for i in range(n_letters):
        row_total = sum(bigram_counts[i])
        row = [
            smooth_log_prob(bigram_counts[i][j], row_total, k, n_letters)
            for j in range(n_letters)
        ]
        bigram_log.append(row)

    vocab = len(word_counts)
    wk = args.word_smoothing
    word_log_prob = {}
    for word, count in word_counts.items():
        word_log_prob[word] = math.log((count + wk) / (total_words + wk * (vocab + 1)))
    unk_word_log_prob = math.log(wk / (total_words + wk * (vocab + 1)))

    words_by_length: dict[str, list[str]] = defaultdict(list)
    for word in word_counts:
        words_by_length[str(len(word))].append(word)
    for key in words_by_length:
        words_by_length[key].sort(key=lambda w: (-word_counts[w], w))

    payload = {
        "meta": {
            "source_train_csv": str(train_csv),
            "total_words_used": total_words,
            "dropped_words": dropped_words,
            "vocab_size": vocab,
            "smoothing_letter": k,
            "smoothing_word": wk,
            "letter_alphabet_size": n_letters,
            "length_histogram": {str(k): v for k, v in sorted(length_counts.items())},
        },
        "letters": RUSSIAN_LETTERS,
        "startLogProb": start_log,
        "bigramLogProb": bigram_log,
        "endLogProb": end_log,
        "wordLogProb": word_log_prob,
        "unkWordLogProb": unk_word_log_prob,
        "wordsByLength": dict(words_by_length),
        "topWords": [w for w, _ in word_counts.most_common(250)],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Read: {train_csv}")
    print(f"Wrote: {out_path}")
    print(f"Words used: {total_words}")
    print(f"Vocabulary: {vocab}")


if __name__ == "__main__":
    main()
