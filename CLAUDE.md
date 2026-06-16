# CLAUDE.md — project context for the Redrob ranker

## What this is
A ranker for the **Hack2skill "India Runs" Redrob Data & AI Challenge**
(Track 1: Intelligent Candidate Discovery). We rank the **top 100 of 100,000
candidates** in `candidates.jsonl` for one job description (Senior AI Engineer,
Founding Team) and emit `submission.csv`. Scored offline against a hidden
ground truth — no labels in the data, no live leaderboard, 3 submissions max.

## Hard rules (from submission_spec)
- Output CSV: `candidate_id,rank,score,reasoning`; exactly 100 rows; ranks 1–100
  unique; score non-increasing; ties broken by candidate_id ascending.
- Ranking step: **CPU only, no network, no GPU, <=5 min, <=16 GB**. Reproduced
  in a sandbox Docker at Stage 3 — must run in budget or DQ.
- ~80 honeypots in the pool; **>10% honeypots in top 100 = DQ**.
- 5 stages: format -> scoring -> code reproduction + honeypot -> manual review
  (reasoning quality, git-history authenticity) -> defend-your-work interview.
- Composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10. **Top-10
  ORDER is half the grade.**

## Approach
Transparent hybrid scoring (no training — there are no labels). The JD is
encoded as weighted requirements in `src/redrob_ranker/jd.py`. Per candidate:
title fit (ceiling), trust-weighted skill relevance (assessment scores /
endorsements / duration defeat keyword stuffers), semantic similarity to the
JD (precomputed bge-small embeddings if present, else offline TF-IDF),
experience curve (peak 6–8 yrs), location (relocation + work mode), notice;
times gates (domain mismatch, services-only, shallow-LLM, tenure/job-hopping,
small big-tech nudge) times a behavioral availability modifier; honeypots sink
via consistency checks. Scores are intentionally NOT clipped to 1.0 so the top
tier spreads (top-10 order matters).

## Run it (Windows note: python isn't on PATH; use the full exe)
```
$py = "C:\Users\sgawa\AppData\Local\Programs\Python\Python312\python.exe"
& $py rank.py --candidates ./candidates.jsonl --out ./submission.csv
& $py validate_submission.py submission.csv      # -> "Submission is valid."
& $py inspect_top.py 40 > top40.txt              # full profiles for review
```
The semantic step prints nothing while it runs (~1–3 min on a laptop); it is
not hung. `candidates.jsonl` must sit next to `rank.py` (it's gitignored).

## Current state: v2
- Validator passes; 0% honeypots in top 100; ~61 s runtime; 97 distinct scores
  in top 100 (top-10 ties fixed).
- Uses ~18 of the 23 redrob_signals. Deliberately skips expected_salary
  (no budget given), connection_count, signup_date (weak/gameable).

## Open tuning questions (decide with the user)
1. Experience band: a 15.6-yr candidate with 4/4 must-have coverage currently
   lands ~#11. JD says the 5–9 band is "not a requirement" and strong outliers
   are OK — so this may be correct. Enforce harder or leave flexible?
2. `BIG_TECH_PENALTY` in structured.py is 0.98 (a whisper). Off / whisper /
   real penalty?

## Still TODO for submission
- Swap TF-IDF for real embeddings: `precompute_embeddings.py` (needs
  sentence-transformers + model download; writes data/*.npy that rank.py loads
  offline).
- Push to GitHub with real commit history (Stage 4 checks for genuine iteration).
- Stand up the required sandbox demo (HF Spaces / Streamlit / Colab).
- Fill submission_metadata.yaml.
