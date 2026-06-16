"""
skills.py — Trust-weighted skill relevance.

The single most important anti-trap logic. A keyword stuffer lists many AI
skills at "expert" with no real depth. We never trust the claimed proficiency
on its own; we discount it by verifiable evidence:

  - duration_months the skill was actually used
  - endorsements from others
  - skill_assessment_scores (objective tests taken on the Redrob platform)

A skill claimed "expert" but with a 38/100 assessment and 0 months used
contributes almost nothing. A skill backed by a high assessment, real
duration, and endorsements contributes fully. Relevance is then the
trust-weighted sum across the JD's skill groups, saturated so that *depth in
the few skills that matter* beats *breadth across many that don't*.
"""

import math
from typing import Dict, Any, Tuple, List
from . import jd

_PROF = {"beginner": 0.35, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}


def _group_of(skill_name: str):
    n = skill_name.lower()
    for group, (weight, terms) in jd.SKILL_GROUPS.items():
        if any(t in n for t in terms):
            return group, weight
    return None, 0.0


def _trust(skill: Dict[str, Any], assessments: Dict[str, float]) -> float:
    """Confidence in [0, 1] that the claimed skill is real and deep."""
    claim = _PROF.get(skill.get("proficiency", ""), 0.4)
    dur = skill.get("duration_months", 0) or 0
    end = skill.get("endorsements", 0) or 0

    dur_f = min(dur / 24.0, 1.0)          # 2+ yrs of use -> full
    end_f = min(end / 25.0, 1.0)          # 25+ endorsements -> full
    evidence = 0.55 * dur_f + 0.25 * end_f + 0.20 * claim

    # Objective assessment, when present, dominates and can veto the claim.
    name = skill.get("name", "")
    if name in assessments:
        a = assessments[name] / 100.0
        # If they claim advanced/expert but bomb the test, crush the trust.
        if claim >= 0.85 and a < 0.45:
            return min(evidence, 0.25) * a * 1.3
        evidence = 0.5 * evidence + 0.5 * a
    return max(0.0, min(evidence, 1.0))


# JD "absolutely need" groups — coverage across these is what separates the
# real top tier from the merely-good.
CORE_GROUPS = ("retrieval_embeddings", "vectordb_search", "eval", "ranking_recsys")


def relevance(c: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, float]]:
    """Return (relevance, strong-skill names, detail).

    Relevance is intentionally NOT saturated to ~1.0 for everyone with a couple
    of matches — it scales with (a) trust-weighted DEPTH per JD group, (b)
    BREADTH across the must-have groups, and (c) the magnitude of objective
    assessment scores on relevant skills. This spreads the strongest candidates
    apart so the top-10 ordering (50% of the grade) is meaningful.
    """
    assessments = (c.get("redrob_signals", {}) or {}).get(
        "skill_assessment_scores", {}) or {}

    group_best: Dict[str, float] = {}
    strong: List[Tuple[float, str]] = []
    assessed_rel: List[float] = []

    for s in c.get("skills", []):
        group, gweight = _group_of(s.get("name", ""))
        if not group:
            continue
        t = _trust(s, assessments)
        contrib = gweight * t
        group_best[group] = max(group_best.get(group, 0.0), contrib)
        if s.get("name", "") in assessments:
            assessed_rel.append(assessments[s["name"]] / 100.0)
        if contrib >= 0.40:
            strong.append((contrib, s.get("name", "")))

    # (a) Depth: trust-weighted sum across all matched groups.
    depth = sum(group_best.values())                       # ~0 .. ~4.4
    # (b) Breadth across the four must-have groups.
    core_cov = sum(1 for g in CORE_GROUPS if group_best.get(g, 0) >= 0.4)
    breadth = core_cov / len(CORE_GROUPS)                   # 0 .. 1
    # (c) Objective assessment magnitude on relevant skills.
    assess_mag = (sum(assessed_rel) / len(assessed_rel)) if assessed_rel else 0.5

    rel = 0.55 * (depth / 3.8) + 0.30 * breadth + 0.15 * assess_mag
    rel = max(0.0, min(rel, 1.15))                          # allow >1 to break ties

    strong.sort(reverse=True)
    detail = {"depth": depth, "core_cov": core_cov, "assess_mag": assess_mag}
    return rel, [name for _, name in strong[:5]], detail


def domain_mismatch(c: Dict[str, Any]) -> float:
    """1.0 if CV/speech/robotics-heavy AND weak on NLP/IR; else 0. (JD 'do not want')."""
    cv = 0
    nlp_ir = 0
    for s in c.get("skills", []):
        n = s.get("name", "").lower()
        if any(t in n for t in jd.DOMAIN_MISMATCH_TERMS):
            cv += 1
        if any(t in n for grp in ("retrieval_embeddings", "vectordb_search",
                                  "ranking_recsys")
               for t in jd.SKILL_GROUPS[grp][1]) or "nlp" in n or "natural language" in n:
            nlp_ir += 1
    if cv >= 3 and nlp_ir == 0:
        return 1.0
    if cv >= 4 and nlp_ir <= 1:
        return 0.6
    return 0.0
