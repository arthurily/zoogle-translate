# Zoogle Translate

Zoogle Translate is a CS109-inspired web app where an alien user teaches a model how their handwritten symbols map to English letters (`A-Z`).

## Core Flow

1. **Setup & Train**
- Name the alien language.
- Build a labeled dataset by drawing one symbol per English letter.
- Train a **letter CNN** on your own handwriting.

2. **Translate**
- Write one symbol per input box.
- A new box appears automatically when you start writing in the current last box.
- Click **Translate** (or press `Enter`) to decode into English text.
- Optional `Add Space` for sentence-style input.

3. **Test Letter**
- Draw one symbol and inspect:
  - Posterior `P(letter | symbol)`
  - Entropy (uncertainty)
  - Bayes-style evidence vs uniform prior
  - Occlusion saliency map
  - Convolution filter contribution breakdown

## CS109 Concepts Highlighted

- Conditional probability and posterior distributions
- Entropy as uncertainty quantification
- Cross-entropy loss minimization
- Train/validation split
- Confusion matrix analysis
- Bayesian framing of evidence

## Data Persistence

- Dataset, language name, and trained model are cached in browser `localStorage`.
- When running `server.py`, all language profiles are also persisted as files in `data/languages/` (one JSON per language).
- In **Setup & Train**, click **Save All Languages To Repo Files** before committing to ensure current profiles (for example, English + Alien) are written to disk.
- Reloading the page keeps your work, including across browser restarts.

## Run

```bash
cd /Users/arthurilyasov/cs109Project
python3 server.py --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## Notes

- This version is intentionally **letter-only CNN** (no full-word model, no external API dependency).
- Recommended minimum is **5 samples per letter** for stable performance.
