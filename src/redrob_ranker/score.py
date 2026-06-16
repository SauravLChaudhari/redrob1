"""
score.py — Combine all components into one fit score per candidate.

Changes vs v1:
  - Skill relevance is magnitude/breadth-aware (skills.py) and the final score
    is NOT clipped to 1.0, so the strongest candidates spread apart instead of
    tying at a saturated 1.0 — critical because NDCG@10 (top-10 ORDER) is 50%
    of the grade.
  - Adds tenure-stability (anti job-hopping) and a small big-tech-only nudge.
  - Broader behavioral coverage (behavioral.py).

No labels exist in this challenge, so weights are hand-tuned against manual
inspection of the top ranks, not trained.
"""

from typing import Dict, Any, Tuple
from . import structured, skills, behavioral, honeypot, reasoning

# Structured-fit blend weights.
W_TITLE = 0.24
W_SKILLS = 0.32
W_SEMANTIC = 0.15
W_EXPERIENCE = 0.12
W_LOCATION = 0.11
W_NOTICE = 0.06


def score_candidate(c: Dict[str, Any], semantic_sim: float
                    ) -> Tuple[float, str, Dict[str, Any]]:
    t_fit, t_class = structured.title_fit(c)
    rel, strong, rel_detail = skills.relevance(c)
    exp = structured.experience_fit(c)
    loc, loc_note = structured.location_fit(c)
    notice, nd = structured.notice_fit(c)

    base = (
        W_TITLE * t_fit
        + W_SKILLS * rel
        + W_SEMANTIC * semantic_sim
        + W_EXPERIENCE * exp
        + W_LOCATION * loc
        + W_NOTICE * notice
    )
    base += structured.nice_to_have_bonus(c)

    # Gates / penalties (multiplicative).
    dm = skills.domain_mismatch(c)
    if dm:
        base *= (1.0 - 0.45 * dm)
    cons = structured.consulting_penalty(c)
    base *= cons
    if structured.shallow_llm_flag(c):
        base *= 0.8
        shallow = True
    else:
        shallow = False

    ten_mult, n_short, avg_ten = structured.tenure_stability(c)
    base *= ten_mult
    bt_mult, bt_only = structured.big_tech_only(c)
    base *= bt_mult

    # Behavioral availability (multiplier).
    behav_mult, behav_note = behavioral.modifier(c)
    base *= behav_mult

    # Honeypot / impossible-profile sink.
    hp = honeypot.suspicion(c)
    if hp >= 0.6:
        base *= 0.03
    elif hp >= 0.3:
        base *= 0.5

    final = max(0.0, base)   # intentionally NOT clipped to 1.0

    parts = {
        "strong_skills": strong, "title_class": t_class,
        "rescued": t_class in ("adjacent", "other_tech") and t_fit > 0.5,
        "loc_note": loc_note, "behav_note": behav_note, "notice_days": nd,
        "domain_mismatch": dm, "consulting_mult": cons, "honeypot": hp,
        "shallow_llm": shallow, "core_cov": rel_detail["core_cov"],
        "assess_mag": rel_detail["assess_mag"], "n_short": n_short,
        "avg_tenure": avg_ten, "big_tech_only": bt_only,
        "company": c.get("profile", {}).get("current_company", ""),
        "title": c.get("profile", {}).get("current_title", ""),
    }
    reason = reasoning.build(c, parts)
    return final, reason, parts
