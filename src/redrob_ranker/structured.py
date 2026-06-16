"""
structured.py — Structured (non-text) fit components derived from the JD.

Each function returns a value in [0, 1] (or a penalty multiplier), and every
threshold maps to a specific JD line so it can be defended at Stage 5.
"""

from typing import Dict, Any, Tuple, List
from . import jd


def title_fit(c: Dict[str, Any]) -> Tuple[float, str]:
    """Best title match across current + career history. Current weighted most."""
    p = c.get("profile", {})
    cur = p.get("current_title", "").lower()
    hist = [r.get("title", "").lower() for r in c.get("career_history", [])]

    def classify(t: str) -> str:
        if any(x in t for x in jd.CORE_TITLE_TERMS):
            return "core"
        if any(x in t for x in jd.DATA_SCIENCE_TERMS):
            return "data_science"
        if any(x in t for x in jd.ADJACENT_TECH_TERMS):
            return "adjacent"
        if any(x in t for x in jd.OTHER_TECH_TERMS):
            return "other_tech"
        return "non_tech"

    cur_class = classify(cur)
    cur_score = jd.TITLE_SCORE[cur_class]

    # A strong past ML title can rescue an adjacent current title (someone who
    # moved from "ML Engineer" to "Backend Engineer" is still relevant), but
    # never rescues a non-tech current title (JD: Marketing Manager is not a fit).
    hist_best = max((jd.TITLE_SCORE[classify(t)] for t in hist), default=0.0)
    if cur_class in ("adjacent", "other_tech"):
        score = max(cur_score, 0.75 * hist_best)
    elif cur_class == "non_tech":
        score = cur_score  # capped low regardless of skill list
    else:
        score = cur_score
    return score, cur_class


def experience_fit(c: Dict[str, Any]) -> float:
    """JD: range 5-9, IDEAL 6-8. Smooth peak at 6-8, slight taper to the edges."""
    y = c.get("profile", {}).get("years_of_experience", 0) or 0
    if y < 2:
        return 0.20
    if y < 4:
        return 0.50
    if y < 5:
        return 0.74
    if y < 6:
        return 0.90          # 5-6: very good, not quite ideal
    if y <= 8:
        return 1.00          # 6-8: the stated ideal band
    if y <= 9:
        return 0.92
    if y <= 12:
        return 0.78
    if y <= 15:
        return 0.55
    return 0.38


def location_fit(c: Dict[str, Any]) -> Tuple[float, str]:
    """JD: Pune/Noida preferred; Hyderabad/Mumbai/Delhi NCR welcome; flexible on
    in-office days but expects Tue/Thu office use + quarterly offsites. So a
    candidate in a non-listed city who also won't relocate is workable but not
    ideal; abroad + won't relocate is a serious concern (no visa sponsorship)."""
    p = c.get("profile", {})
    loc = p.get("location", "").lower()
    sig = c.get("redrob_signals", {}) or {}
    relo = sig.get("willing_to_relocate", False)
    mode = sig.get("preferred_work_mode", "")
    country = p.get("country", "").lower()

    if country == "india":
        if any(city in loc for city in jd.PREFERRED_CITIES):
            return 1.0, "preferred-city"
        # Non-listed Indian city: relocation willingness matters.
        if relo:
            return 0.88, "india-other-will-relocate"
        # Won't relocate: remote-only preference makes it worse for a Tue/Thu role.
        return (0.58 if mode == "remote" else 0.66), "india-other-no-relocate"

    # Outside India: case-by-case, no visa sponsorship.
    if relo:
        return 0.45, "abroad-will-relocate"
    return 0.10, "abroad-no-relocate"


def tenure_stability(c: Dict[str, Any]) -> Tuple[float, int, float]:
    """JD: title-chasers who switch companies every ~1.5 yrs are not a fit;
    they want 3+ year commitment. Returns (multiplier, n_short_stints, avg_yrs).
    Only completed roles count; the current role's short tenure isn't penalized."""
    hist = c.get("career_history", [])
    completed = [h for h in hist if not h.get("is_current")]
    if len(completed) < 2:
        return 1.0, 0, 0.0
    durs = [(h.get("duration_months", 0) or 0) / 12.0 for h in completed]
    avg = sum(durs) / len(durs)
    short = sum(1 for d in durs if d < 1.5)
    if short >= 3 and avg < 1.8:
        return 0.80, short, avg          # serial hopper
    if short >= 2 and avg < 2.2:
        return 0.90, short, avg
    if avg >= 3.0:
        return 1.04, short, avg          # demonstrated commitment
    return 1.0, short, avg


# Mega-cap big tech. JD caveats (but does not ban) Google/Meta-style careers
# for people who want a well-scoped role. Treated as a SMALL, tunable nudge.
BIG_TECH = ["google", "meta", "facebook", "apple", "amazon", "microsoft",
            "netflix"]
BIG_TECH_PENALTY = 0.98   # set to 1.0 to disable; lower to penalize harder


def big_tech_only(c: Dict[str, Any]) -> Tuple[float, bool]:
    hist = c.get("career_history", [])
    if len(hist) < 2:
        return 1.0, False
    companies = [h.get("company", "").lower() for h in hist]
    if all(any(b in co for b in BIG_TECH) for co in companies):
        return BIG_TECH_PENALTY, True
    return 1.0, False


def notice_fit(c: Dict[str, Any]) -> Tuple[float, int]:
    nd = (c.get("redrob_signals", {}) or {}).get("notice_period_days", 60) or 0
    if nd <= 30:
        return 1.0, nd
    if nd <= 60:
        return 0.85, nd
    if nd <= 90:
        return 0.60, nd
    return 0.40, nd


def consulting_penalty(c: Dict[str, Any]) -> float:
    """Multiplier. JD: only-consulting career is not a fit unless prior product."""
    hist = c.get("career_history", [])
    if not hist:
        return 1.0
    companies = [r.get("company", "").lower() for r in hist]
    n_consult = sum(1 for co in companies
                    if any(f in co for f in jd.CONSULTING_FIRMS))
    if n_consult == len(companies) and len(companies) >= 2:
        return 0.45          # entire career in services
    if n_consult == len(companies):
        return 0.65
    return 1.0


def nice_to_have_bonus(c: Dict[str, Any]) -> float:
    """Small additive bonus (<=0.06) for JD nice-to-haves: LoRA/PEFT, LTR,
    HR-tech, OSS / github activity."""
    bonus = 0.0
    sig = c.get("redrob_signals", {}) or {}
    text = " ".join(s.get("name", "").lower() for s in c.get("skills", []))
    if any(x in text for x in ("lora", "qlora", "peft")):
        bonus += 0.02
    if any(x in text for x in ("learning to rank", "ltr", "xgboost")):
        bonus += 0.02
    gh = sig.get("github_activity_score", -1)
    if gh and gh > 50:
        bonus += 0.02
    return min(bonus, 0.06)


def shallow_llm_flag(c: Dict[str, Any]) -> bool:
    """JD: '<12 months of LangChain calling OpenAI' framework-enthusiast flag,
    when there is no deeper ML evidence."""
    text = " ".join(
        [c.get("profile", {}).get("summary", "").lower()]
        + [r.get("description", "").lower() for r in c.get("career_history", [])]
        + [s.get("name", "").lower() for s in c.get("skills", [])]
    )
    framework = any(x in text for x in jd.FRAMEWORK_ENTHUSIAST_TERMS)
    deep = any(x in text for x in ("retrieval", "ranking", "embedding",
                                   "recommendation", "vector", "ndcg"))
    return framework and not deep
