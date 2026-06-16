"""
honeypot.py — Detect the ~80 planted honeypot profiles.

Spec section 7: honeypots are "subtly impossible profiles (e.g., 8 years of
experience at a company founded 3 years ago; 'expert' proficiency in 10 skills
with 0 years used)". They are forced to relevance tier 0, and a top-100
honeypot rate > 10% is an automatic Stage-3 disqualification.

The organizers say a good system should *naturally* avoid them by actually
reading profiles. We do that by checking internal consistency. We deliberately
do NOT hard-special-case them in the final ranking — we return a suspicion
score that simply makes the profile sink, which is both safer (avoids false
positives on genuine candidates) and matches the spec's intent.
"""

from datetime import date
from typing import Dict, Any


def _parse(d):
    try:
        return date.fromisoformat(d)
    except Exception:
        return None


def suspicion(c: Dict[str, Any]) -> float:
    """Return a suspicion score in [0, 1]; higher = more likely honeypot."""
    flags = 0.0
    p = c.get("profile", {})
    hist = c.get("career_history", [])
    skills = c.get("skills", [])

    yoe = p.get("years_of_experience", 0) or 0

    # 1. "expert"/"advanced" proficiency with 0 months of actual use.
    impossible_skill = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months", 0) or 0) == 0
    )
    if impossible_skill >= 3:
        flags += 0.6
    elif impossible_skill >= 1:
        flags += 0.2

    # 2. A single skill used longer than the whole career.
    if any((s.get("duration_months", 0) or 0) > (yoe * 12 + 18) for s in skills):
        flags += 0.4

    # 3. Career-history months wildly exceed stated experience.
    total_months = sum(r.get("duration_months", 0) or 0 for r in hist)
    if total_months > (yoe * 12) * 1.6 + 24:
        flags += 0.4

    # 4. Role dates internally inconsistent (end before start, or
    #    duration_months disagrees badly with the date span).
    for r in hist:
        s, e = _parse(r.get("start_date", "")), _parse(r.get("end_date", ""))
        if s and e:
            if e < s:
                flags += 0.5
            span = (e - s).days / 30.4
            dm = r.get("duration_months", 0) or 0
            if dm and abs(span - dm) > max(12, 0.5 * dm):
                flags += 0.3

    # 5. Education timeline impossible.
    for ed in c.get("education", []):
        sy, ey = ed.get("start_year"), ed.get("end_year")
        if sy and ey and ey < sy:
            flags += 0.4

    # 6. "expert" everywhere but assessments tell another story is handled in
    #    skills.py (trust); here we only catch structural impossibility.

    return min(flags, 1.0)


def is_honeypot(c: Dict[str, Any], threshold: float = 0.6) -> bool:
    return suspicion(c) >= threshold
