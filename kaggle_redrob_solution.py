"""
Redrob Intelligent Candidate Discovery — single-file Kaggle solution.

Paste this into a Kaggle notebook cell (GPU T4 enabled, Internet ON so the
embedding model can download). It will:
  1. load candidates.jsonl,
  2. embed every profile + the JD on GPU (bge-base-en-v1.5), falling back to a
     CPU TF-IDF similarity if no GPU / no sentence-transformers,
  3. run the full hybrid scoring,
  4. write /kaggle/working/submission.csv,
  5. save /kaggle/working/candidate_embeddings.npy + jd_embedding.npy
     (commit these to your repo so rank.py reproduces offline on CPU),
  6. validate, and print the top 20 + honeypot rate + score spread.

NOTE ON THE RULES: GPU/Internet are fine *here* — this is where you generate
the submission and the precomputed embeddings. The actual Stage-3 reproduction
runs your repo's rank.py on CPU with no network, loading the .npy artifacts.
"""

# ===================== CONFIG =====================
CANDIDATES_PATH = "/kaggle/input/datasets/saur3x/redrobdatac/candidates.jsonl"
OUT_DIR = "/kaggle/working"
MODEL_NAME = "BAAI/bge-base-en-v1.5"   # 768-dim; swap to bge-small-en-v1.5 for speed
EMB_BATCH = 512
TOPK = 100
BIG_TECH_PENALTY = 0.98                # 1.0 disables the big-tech nudge

import os, json, math, csv
from datetime import date
import numpy as np

# ===================== JD SPEC (encoded requirements) =====================
import types

CORE_TITLE_TERMS = [
    "machine learning engineer", "ml engineer", "ai engineer", "applied scientist",
    "applied ml", "research engineer", "nlp engineer", "search engineer",
    "ranking engineer", "recommendation", "recsys", "mlops engineer",
    "deep learning engineer", "ml scientist", "machine learning scientist",
]
DATA_SCIENCE_TERMS = ["data scientist", "ml researcher", "research scientist"]
ADJACENT_TECH_TERMS = [
    "data engineer", "analytics engineer", "backend engineer", "software engineer",
    "full stack", "platform engineer", "staff engineer", "principal engineer",
]
OTHER_TECH_TERMS = [
    "frontend", "mobile developer", "android", "ios", "devops", "cloud engineer",
    "qa engineer", "java developer", ".net", "web developer", "sre",
    "database administrator",
]
TITLE_SCORE = {"core": 1.00, "data_science": 0.82, "adjacent": 0.50,
               "other_tech": 0.22, "non_tech": 0.04}
SKILL_GROUPS = {
    "retrieval_embeddings": (1.00, ["embedding", "sentence-transformer", "sbert",
        "bge", "e5", "retrieval", "rag", "semantic search", "dense retrieval"]),
    "vectordb_search": (1.00, ["faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "bm25", "vector search", "hybrid search",
        "lucene", "solr"]),
    "eval": (0.95, ["ndcg", "mrr", "map@", "a/b test", "ab test", "experimentation",
        "ranking metrics", "offline evaluation", "mlflow"]),
    "ranking_recsys": (0.95, ["learning to rank", "ltr", "ranking", "recommendation",
        "recommender", "recsys", "collaborative filtering"]),
    "ml_core": (0.65, ["machine learning", "deep learning", "pytorch", "tensorflow",
        "scikit", "xgboost", "lightgbm", "nlp", "natural language", "neural network",
        "classification", "regression"]),
    "llm": (0.55, ["llm", "large language model", "fine-tuning", "lora", "qlora",
        "peft", "transformers", "hugging face", "huggingface", "bert"]),
    "data_eng": (0.30, ["spark", "airflow", "kafka", "sql", "etl", "data pipeline",
        "snowflake", "dbt", "databricks"]),
}
DOMAIN_MISMATCH_TERMS = ["image classification", "object detection", "segmentation",
    "gans", "computer vision", "opencv", "speech recognition", "tts", "asr",
    "robotics", "slam", "lidar", "pose estimation", "ocr"]
CONSULTING_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree",
    "mphasis", "hexaware", "ibm services", "deloitte", "pwc"]
PREFERRED_CITIES = ["pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "ncr", "bengaluru", "bangalore"]
FRAMEWORK_ENTHUSIAST_TERMS = ["langchain", "llamaindex", "autogpt", "crewai"]
BIG_TECH = ["google", "meta", "facebook", "apple", "amazon", "microsoft", "netflix"]
JD_TEXT = (
    "Senior AI Engineer founding team. Own the intelligence layer: ranking, "
    "retrieval and matching systems deciding what recruiters see when they search "
    "candidates. Production experience with embeddings-based retrieval "
    "(sentence-transformers, BGE, E5), vector databases and hybrid search (FAISS, "
    "Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, OpenSearch), "
    "learning-to-rank, LLM fine-tuning, and rigorous evaluation of ranking systems "
    "with NDCG, MRR, MAP and A/B testing. Shipped end-to-end ranking, search or "
    "recommendation systems to real users at scale at product companies, not pure "
    "research and not pure services. Strong Python and code quality. Scrappy "
    "product-engineering attitude over pure research."
)
jd = types.SimpleNamespace(
    CORE_TITLE_TERMS=CORE_TITLE_TERMS, DATA_SCIENCE_TERMS=DATA_SCIENCE_TERMS,
    ADJACENT_TECH_TERMS=ADJACENT_TECH_TERMS, OTHER_TECH_TERMS=OTHER_TECH_TERMS,
    TITLE_SCORE=TITLE_SCORE, SKILL_GROUPS=SKILL_GROUPS,
    DOMAIN_MISMATCH_TERMS=DOMAIN_MISMATCH_TERMS, CONSULTING_FIRMS=CONSULTING_FIRMS,
    PREFERRED_CITIES=PREFERRED_CITIES,
    FRAMEWORK_ENTHUSIAST_TERMS=FRAMEWORK_ENTHUSIAST_TERMS, JD_TEXT=JD_TEXT,
)

# ===================== DATA LOADING =====================
def stream_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def profile_text(c):
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for role in c.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    parts.append(" ".join(s.get("name", "") for s in c.get("skills", [])))
    return " ".join(x for x in parts if x)

# ===================== HONEYPOT =====================
def _pdate(d):
    try:
        return date.fromisoformat(d)
    except Exception:
        return None

def honeypot_suspicion(c):
    flags = 0.0
    p = c.get("profile", {}); hist = c.get("career_history", [])
    skills = c.get("skills", []); yoe = p.get("years_of_experience", 0) or 0
    imp = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced")
              and (s.get("duration_months", 0) or 0) == 0)
    flags += 0.6 if imp >= 3 else (0.2 if imp >= 1 else 0)
    if any((s.get("duration_months", 0) or 0) > (yoe * 12 + 18) for s in skills):
        flags += 0.4
    total = sum(r.get("duration_months", 0) or 0 for r in hist)
    if total > (yoe * 12) * 1.6 + 24:
        flags += 0.4
    for r in hist:
        s, e = _pdate(r.get("start_date", "")), _pdate(r.get("end_date", ""))
        if s and e:
            if e < s:
                flags += 0.5
            span = (e - s).days / 30.4; dm = r.get("duration_months", 0) or 0
            if dm and abs(span - dm) > max(12, 0.5 * dm):
                flags += 0.3
    for ed in c.get("education", []):
        sy, ey = ed.get("start_year"), ed.get("end_year")
        if sy and ey and ey < sy:
            flags += 0.4
    return min(flags, 1.0)

# ===================== SKILLS (trust-weighted relevance) =====================
_PROF = {"beginner": 0.35, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
CORE_GROUPS = ("retrieval_embeddings", "vectordb_search", "eval", "ranking_recsys")

def _group_of(name):
    n = name.lower()
    for g, (w, terms) in jd.SKILL_GROUPS.items():
        if any(t in n for t in terms):
            return g, w
    return None, 0.0

def _trust(skill, assessments):
    claim = _PROF.get(skill.get("proficiency", ""), 0.4)
    dur = skill.get("duration_months", 0) or 0
    end = skill.get("endorsements", 0) or 0
    evidence = 0.55 * min(dur / 24.0, 1.0) + 0.25 * min(end / 25.0, 1.0) + 0.20 * claim
    name = skill.get("name", "")
    if name in assessments:
        a = assessments[name] / 100.0
        if claim >= 0.85 and a < 0.45:
            return min(evidence, 0.25) * a * 1.3
        evidence = 0.5 * evidence + 0.5 * a
    return max(0.0, min(evidence, 1.0))

def relevance(c):
    assessments = (c.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    group_best = {}; strong = []; assessed_rel = []
    for s in c.get("skills", []):
        g, gw = _group_of(s.get("name", ""))
        if not g:
            continue
        t = _trust(s, assessments); contrib = gw * t
        group_best[g] = max(group_best.get(g, 0.0), contrib)
        if s.get("name", "") in assessments:
            assessed_rel.append(assessments[s["name"]] / 100.0)
        if contrib >= 0.40:
            strong.append((contrib, s.get("name", "")))
    depth = sum(group_best.values())
    core_cov = sum(1 for g in CORE_GROUPS if group_best.get(g, 0) >= 0.4)
    breadth = core_cov / len(CORE_GROUPS)
    assess_mag = (sum(assessed_rel) / len(assessed_rel)) if assessed_rel else 0.5
    rel = 0.55 * (depth / 3.8) + 0.30 * breadth + 0.15 * assess_mag
    rel = max(0.0, min(rel, 1.15))
    strong.sort(reverse=True)
    return rel, [n for _, n in strong[:5]], {"core_cov": core_cov, "assess_mag": assess_mag}

def domain_mismatch(c):
    cv = nlp_ir = 0
    for s in c.get("skills", []):
        n = s.get("name", "").lower()
        if any(t in n for t in jd.DOMAIN_MISMATCH_TERMS):
            cv += 1
        if (any(t in n for grp in ("retrieval_embeddings", "vectordb_search", "ranking_recsys")
                for t in jd.SKILL_GROUPS[grp][1]) or "nlp" in n or "natural language" in n):
            nlp_ir += 1
    if cv >= 3 and nlp_ir == 0:
        return 1.0
    if cv >= 4 and nlp_ir <= 1:
        return 0.6
    return 0.0

# ===================== STRUCTURED =====================
def title_fit(c):
    p = c.get("profile", {})
    cur = p.get("current_title", "").lower()
    hist = [r.get("title", "").lower() for r in c.get("career_history", [])]
    def classify(t):
        if any(x in t for x in jd.CORE_TITLE_TERMS):
            return "core"
        if any(x in t for x in jd.DATA_SCIENCE_TERMS):
            return "data_science"
        if any(x in t for x in jd.ADJACENT_TECH_TERMS):
            return "adjacent"
        if any(x in t for x in jd.OTHER_TECH_TERMS):
            return "other_tech"
        return "non_tech"
    cc = classify(cur); cs = jd.TITLE_SCORE[cc]
    hb = max((jd.TITLE_SCORE[classify(t)] for t in hist), default=0.0)
    if cc in ("adjacent", "other_tech"):
        score = max(cs, 0.75 * hb)
    else:
        score = cs
    return score, cc

def experience_fit(c):
    y = c.get("profile", {}).get("years_of_experience", 0) or 0
    if y < 2: return 0.20
    if y < 4: return 0.50
    if y < 5: return 0.74
    if y < 6: return 0.90
    if y <= 8: return 1.00
    if y <= 9: return 0.92
    if y <= 12: return 0.78
    if y <= 15: return 0.55
    return 0.38

def location_fit(c):
    p = c.get("profile", {}); loc = p.get("location", "").lower()
    sig = c.get("redrob_signals", {}) or {}
    relo = sig.get("willing_to_relocate", False); mode = sig.get("preferred_work_mode", "")
    if p.get("country", "").lower() == "india":
        if any(city in loc for city in jd.PREFERRED_CITIES):
            return 1.0, "preferred-city"
        if relo:
            return 0.88, "india-other-will-relocate"
        return (0.58 if mode == "remote" else 0.66), "india-other-no-relocate"
    if relo:
        return 0.45, "abroad-will-relocate"
    return 0.10, "abroad-no-relocate"

def notice_fit(c):
    nd = (c.get("redrob_signals", {}) or {}).get("notice_period_days", 60) or 0
    if nd <= 30: return 1.0, nd
    if nd <= 60: return 0.85, nd
    if nd <= 90: return 0.60, nd
    return 0.40, nd

def consulting_penalty(c):
    hist = c.get("career_history", [])
    if not hist:
        return 1.0
    comps = [r.get("company", "").lower() for r in hist]
    nc = sum(1 for co in comps if any(f in co for f in jd.CONSULTING_FIRMS))
    if nc == len(comps) and len(comps) >= 2:
        return 0.45
    if nc == len(comps):
        return 0.65
    return 1.0

def nice_to_have_bonus(c):
    b = 0.0; sig = c.get("redrob_signals", {}) or {}
    text = " ".join(s.get("name", "").lower() for s in c.get("skills", []))
    if any(x in text for x in ("lora", "qlora", "peft")): b += 0.02
    if any(x in text for x in ("learning to rank", "ltr", "xgboost")): b += 0.02
    gh = sig.get("github_activity_score", -1)
    if gh and gh > 50: b += 0.02
    return min(b, 0.06)

def shallow_llm_flag(c):
    text = " ".join([c.get("profile", {}).get("summary", "").lower()]
        + [r.get("description", "").lower() for r in c.get("career_history", [])]
        + [s.get("name", "").lower() for s in c.get("skills", [])])
    return (any(x in text for x in jd.FRAMEWORK_ENTHUSIAST_TERMS)
            and not any(x in text for x in ("retrieval", "ranking", "embedding",
                "recommendation", "vector", "ndcg")))

def tenure_stability(c):
    hist = [h for h in c.get("career_history", []) if not h.get("is_current")]
    if len(hist) < 2:
        return 1.0, 0, 0.0
    durs = [(h.get("duration_months", 0) or 0) / 12.0 for h in hist]
    avg = sum(durs) / len(durs); short = sum(1 for d in durs if d < 1.5)
    if short >= 3 and avg < 1.8: return 0.80, short, avg
    if short >= 2 and avg < 2.2: return 0.90, short, avg
    if avg >= 3.0: return 1.04, short, avg
    return 1.0, short, avg

def big_tech_only(c):
    hist = c.get("career_history", [])
    if len(hist) < 2:
        return 1.0, False
    comps = [h.get("company", "").lower() for h in hist]
    if all(any(b in co for b in BIG_TECH) for co in comps):
        return BIG_TECH_PENALTY, True
    return 1.0, False

# ===================== BEHAVIORAL =====================
AS_OF = date(2026, 5, 20)
def behavioral_modifier(c):
    sig = c.get("redrob_signals", {}) or {}; m = 1.0; notes = []
    la = _pdate(sig.get("last_active_date", ""))
    if la:
        days = (AS_OF - la).days
        m *= 1.06 if days <= 14 else (1.0 if days <= 45 else (0.9 if days <= 120 else 0.7))
        if days > 120: notes.append(f"inactive {days}d")
    rr = sig.get("recruiter_response_rate", None)
    if rr is not None:
        if rr < 0.1: m *= 0.75; notes.append(f"low response {rr:.2f}")
        elif rr >= 0.6: m *= 1.05
    art = sig.get("avg_response_time_hours", None)
    if art is not None:
        if art <= 12: m *= 1.03
        elif art >= 120: m *= 0.95
    m *= 1.04 if sig.get("open_to_work_flag") else 0.94
    if (sig.get("applications_submitted_30d", 0) or 0) >= 1: m *= 1.02
    if (sig.get("saved_by_recruiters_30d", 0) or 0) >= 5: m *= 1.03
    if (sig.get("search_appearance_30d", 0) or 0) >= 150: m *= 1.01
    icr = sig.get("interview_completion_rate", None)
    if icr is not None and icr < 0.4: m *= 0.9; notes.append(f"low interview show {icr:.2f}")
    oar = sig.get("offer_acceptance_rate", None)
    if oar is not None and 0 <= oar < 0.2: m *= 0.95; notes.append("rarely accepts offers")
    pcs = sig.get("profile_completeness_score", 100) or 0
    if pcs < 50: m *= 0.93
    ver = sum(bool(sig.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected"))
    m *= 1.02 if ver >= 2 else (0.97 if ver == 0 else 1.0)
    return max(0.5, min(m, 1.18)), "; ".join(notes)

# ===================== REASONING =====================
def build_reasoning(c, parts):
    p = c.get("profile", {})
    title = parts.get("title") or "Candidate"; company = parts.get("company") or "?"
    yoe = p.get("years_of_experience", 0) or 0
    seg = [f"{title} at {company}, {yoe:.1f} yrs"]
    cov = parts.get("core_cov", 0); strong = parts.get("strong_skills", [])
    seg.append(f"covers {cov}/4 must-have areas via {', '.join(strong[:3])}" if strong
               else "no verified retrieval/ranking depth")
    am = parts.get("assess_mag", 0)
    if am >= 0.8: seg.append("high verified assessment scores")
    elif am and am < 0.45 and strong: seg.append("but assessment scores are weak")
    avg = parts.get("avg_tenure", 0)
    if parts.get("n_short", 0) >= 2 and avg and avg < 2.2:
        seg.append(f"frequent short stints (~{avg:.1f}y avg)")
    elif avg and avg >= 3.0:
        seg.append(f"stable tenure (~{avg:.1f}y avg)")
    if parts.get("title_class") == "adjacent" and parts.get("rescued"):
        seg.append("adjacent current title but career shows real ML/IR work")
    ln = parts.get("loc_note")
    lp = {"preferred-city": f"based in {p.get('location')}",
          "india-other-will-relocate": "open to relocating within India",
          "india-other-no-relocate": f"in {p.get('location')}, not relocating",
          "abroad-will-relocate": "abroad but open to relocating",
          "abroad-no-relocate": "abroad and not relocating (no visa sponsorship)"}.get(ln)
    if lp: seg.append(lp)
    concerns = []
    if parts.get("behav_note"): concerns.append(parts["behav_note"])
    nd = parts.get("notice_days")
    if nd and nd > 90: concerns.append(f"notice {nd}d")
    if parts.get("domain_mismatch", 0) >= 0.6: concerns.append("CV/speech-heavy, light NLP/IR")
    if parts.get("consulting_mult", 1.0) < 0.7: concerns.append("services-only career")
    if parts.get("big_tech_only"): concerns.append("entire career at big tech (JD prefers scrappy/product)")
    if parts.get("honeypot", 0) >= 0.6: concerns.append("profile consistency flags")
    if parts.get("shallow_llm"): concerns.append("LLM exposure looks framework-only")
    text = "; ".join(seg)
    if concerns: text += ". Concerns: " + ", ".join(concerns)
    return text + "."

# ===================== SCORE =====================
W_TITLE, W_SKILLS, W_SEMANTIC, W_EXPERIENCE, W_LOCATION, W_NOTICE = \
    0.24, 0.32, 0.15, 0.12, 0.11, 0.06

def score_candidate(c, sim):
    t_fit, t_class = title_fit(c)
    rel, strong, rd = relevance(c)
    exp = experience_fit(c); loc, loc_note = location_fit(c); notice, nd = notice_fit(c)
    base = (W_TITLE*t_fit + W_SKILLS*rel + W_SEMANTIC*sim + W_EXPERIENCE*exp
            + W_LOCATION*loc + W_NOTICE*notice) + nice_to_have_bonus(c)
    dm = domain_mismatch(c)
    if dm: base *= (1.0 - 0.45 * dm)
    cons = consulting_penalty(c); base *= cons
    shallow = shallow_llm_flag(c)
    if shallow: base *= 0.8
    ten_mult, n_short, avg_ten = tenure_stability(c); base *= ten_mult
    bt_mult, bt_only = big_tech_only(c); base *= bt_mult
    behav_mult, behav_note = behavioral_modifier(c); base *= behav_mult
    hp = honeypot_suspicion(c)
    if hp >= 0.6: base *= 0.03
    elif hp >= 0.3: base *= 0.5
    final = max(0.0, base)
    parts = {"strong_skills": strong, "title_class": t_class,
        "rescued": t_class in ("adjacent", "other_tech") and t_fit > 0.5,
        "loc_note": loc_note, "behav_note": behav_note, "notice_days": nd,
        "domain_mismatch": dm, "consulting_mult": cons, "honeypot": hp,
        "shallow_llm": shallow, "core_cov": rd["core_cov"], "assess_mag": rd["assess_mag"],
        "n_short": n_short, "avg_tenure": avg_ten, "big_tech_only": bt_only,
        "company": c.get("profile", {}).get("current_company", ""),
        "title": c.get("profile", {}).get("current_title", "")}
    return final, build_reasoning(c, parts), parts

# ===================== SEMANTIC (GPU embeddings or TF-IDF fallback) =====================
def compute_similarities(texts):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[semantic] embedding with {MODEL_NAME} on {dev}")
        model = SentenceTransformer(MODEL_NAME, device=dev)
        emb = model.encode(texts, batch_size=EMB_BATCH, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        # bge wants an instruction on the QUERY (JD) side for retrieval.
        q = "Represent this sentence for searching relevant passages: " + jd.JD_TEXT
        jd_emb = model.encode([q], convert_to_numpy=True,
                              normalize_embeddings=True)[0].astype("float32")
        np.save(os.path.join(OUT_DIR, "candidate_embeddings.npy"), emb)
        np.save(os.path.join(OUT_DIR, "jd_embedding.npy"), jd_emb)
        print("[semantic] saved candidate_embeddings.npy + jd_embedding.npy")
        sims = emb @ jd_emb
        return (sims + 1.0) / 2.0
    except Exception as e:
        print(f"[semantic] embedding unavailable ({e}); using TF-IDF fallback")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2),
                              sublinear_tf=True, stop_words="english", min_df=3)
        tf = vec.fit_transform(texts + [jd.JD_TEXT])
        sims = (tf[:-1] @ tf[-1].T).toarray().ravel()
        return sims / (sims.max() + 1e-9)

# ===================== MAIN =====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[load] reading candidates...")
    cands = list(stream_candidates(CANDIDATES_PATH))
    texts = [profile_text(c) for c in cands]
    print(f"[load] {len(cands)} candidates")

    sims = compute_similarities(texts)

    rows = []
    for c, sim in zip(cands, sims):
        s, reason, _ = score_candidate(c, float(sim))
        rows.append((round(s, 4), c["candidate_id"], reason))
    rows.sort(key=lambda r: (-r[0], r[1]))
    top = rows[:TOPK]

    out_csv = os.path.join(OUT_DIR, "submission.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (s, cid, reason) in enumerate(top, start=1):
            w.writerow([cid, i, f"{s:.4f}", reason])
    print(f"[done] wrote {out_csv}")

    # --- self-checks ---
    scores = [s for s, _, _ in top]
    assert len(top) == TOPK, "must be 100 rows"
    assert scores == sorted(scores, reverse=True), "scores must be non-increasing"
    cid_index = {c["candidate_id"] for c in cands}
    assert all(cid in cid_index for _, cid, _ in top), "unknown candidate_id"
    hp_rate = sum(1 for _, cid, _ in top
                  if honeypot_suspicion(next(c for c in cands if c["candidate_id"] == cid)) >= 0.6)
    print(f"[check] honeypots in top100: {hp_rate}%  (DQ if >10)")
    print(f"[check] distinct scores in top100: {len(set(scores))}")
    print(f"[check] score range: {scores[0]} .. {scores[-1]}")

    idx = {c["candidate_id"]: c for c in cands}
    print("\n===== TOP 20 =====")
    for s, cid, reason in top[:20]:
        p = idx[cid]["profile"]
        print(f"{s:<7} {p['current_title'][:26]:<26}@{p['current_company'][:12]:<12} "
              f"{p['years_of_experience']:>4}y {p['location'][:16]}")

if __name__ == "__main__":
    main()
