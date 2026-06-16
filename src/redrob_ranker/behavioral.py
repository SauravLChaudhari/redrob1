"""
behavioral.py — Availability / engagement modifier.

JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has a
5% recruiter response rate is, for hiring purposes, not actually available.
Down-weight them appropriately."

Returns a multiplier in roughly [0.6, 1.08] applied to the fit score, plus a
short human-readable note for the reasoning column.
"""

from datetime import date
from typing import Dict, Any, Tuple

# Dataset "as of" date — newest last_active in the pool is ~2026-05-20.
AS_OF = date(2026, 5, 20)


def _parse(d):
    try:
        return date.fromisoformat(d)
    except Exception:
        return None


def modifier(c: Dict[str, Any]) -> Tuple[float, str]:
    sig = c.get("redrob_signals", {}) or {}
    m = 1.0
    notes = []

    # Recency of activity.
    la = _parse(sig.get("last_active_date", ""))
    if la:
        days = (AS_OF - la).days
        if days <= 14:
            m *= 1.06
        elif days <= 45:
            m *= 1.0
        elif days <= 120:
            m *= 0.9
        else:
            m *= 0.7
            notes.append(f"inactive {days}d")

    # Recruiter responsiveness (rate + speed).
    rr = sig.get("recruiter_response_rate", None)
    if rr is not None:
        if rr < 0.1:
            m *= 0.75
            notes.append(f"low response {rr:.2f}")
        elif rr >= 0.6:
            m *= 1.05
    art = sig.get("avg_response_time_hours", None)
    if art is not None:
        if art <= 12:
            m *= 1.03
        elif art >= 120:
            m *= 0.95

    # Explicit availability + active job-seeking.
    if sig.get("open_to_work_flag"):
        m *= 1.04
    else:
        m *= 0.94
    apps = sig.get("applications_submitted_30d", 0) or 0
    if apps >= 1:
        m *= 1.02

    # Recruiter demand (search visibility + bookmarks).
    if (sig.get("saved_by_recruiters_30d", 0) or 0) >= 5:
        m *= 1.03
    if (sig.get("search_appearance_30d", 0) or 0) >= 150:
        m *= 1.01

    # Reliability: shows up to interviews; accepts offers.
    icr = sig.get("interview_completion_rate", None)
    if icr is not None and icr < 0.4:
        m *= 0.9
        notes.append(f"low interview show {icr:.2f}")
    oar = sig.get("offer_acceptance_rate", None)
    if oar is not None and oar >= 0 and oar < 0.2:
        m *= 0.95
        notes.append("rarely accepts offers")

    # Profile trust / completeness.
    pcs = sig.get("profile_completeness_score", 100) or 0
    if pcs < 50:
        m *= 0.93
    verified = sum(bool(sig.get(k)) for k in
                   ("verified_email", "verified_phone", "linkedin_connected"))
    if verified >= 2:
        m *= 1.02
    elif verified == 0:
        m *= 0.97

    return max(0.5, min(m, 1.18)), "; ".join(notes)
