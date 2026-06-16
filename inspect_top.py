#!/usr/bin/env python3
"""inspect_top.py — Dump FULL profiles of your top-N for manual review.

Usage:
    python inspect_top.py 40              # full detail for top 40
    python inspect_top.py 40 > top40.txt  # send to a file to scroll/search

Shows complete career history (to judge job-hopping / prior experience /
services-vs-product), all skills with assessment scores, and all 23 Redrob
signals, alongside this ranker's score and reasoning.
"""
import json, csv, sys
from datetime import date

n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
idx = {}
for line in open("candidates.jsonl", encoding="utf-8"):
    if line.strip():
        c = json.loads(line); idx[c["candidate_id"]] = c

rows = list(csv.DictReader(open("submission.csv", encoding="utf-8")))[:n]

for r in rows:
    c = idx[r["candidate_id"]]; p = c["profile"]; s = c["redrob_signals"]
    print("=" * 90)
    print(f"#{r['rank']}  score={r['score']}  {r['candidate_id']}  ::  {p['anonymized_name']}")
    print(f"  {p['current_title']} @ {p['current_company']} ({p['current_company_size']}, {p['current_industry']})")
    print(f"  {p['years_of_experience']}y total | {p['location']}, {p['country']}")
    print(f"  headline: {p['headline']}")
    print(f"  summary:  {p['summary']}")

    print("  -- career history (most recent first) --")
    for h in c.get("career_history", []):
        yrs = (h.get("duration_months", 0) or 0) / 12.0
        cur = " [current]" if h.get("is_current") else ""
        print(f"     {h['title']} @ {h['company']} ({h.get('company_size','?')}, {h.get('industry','?')}) "
              f"| {h.get('start_date','?')}->{h.get('end_date') or 'now'} = {yrs:.1f}y{cur}")

    edu = c.get("education", [])
    if edu:
        print("  -- education --")
        for e in edu:
            print(f"     {e.get('degree','?')} {e.get('field_of_study','')} @ {e.get('institution','?')} "
                  f"({e.get('start_year','?')}-{e.get('end_year','?')}, tier={e.get('tier','?')})")

    print("  -- skills (name | proficiency | endorsements | months | assessment) --")
    assess = s.get("skill_assessment_scores", {}) or {}
    for sk in c.get("skills", []):
        a = assess.get(sk["name"])
        a = f"{a:.0f}" if a is not None else "-"
        print(f"     {sk['name']:<26} {sk.get('proficiency','?'):<12} "
              f"end={sk.get('endorsements',0):<4} mo={sk.get('duration_months',0):<4} assess={a}")

    print("  -- all 23 redrob signals --")
    order = ["profile_completeness_score","signup_date","last_active_date","open_to_work_flag",
             "profile_views_received_30d","applications_submitted_30d","recruiter_response_rate",
             "avg_response_time_hours","connection_count","endorsements_received","notice_period_days",
             "preferred_work_mode","willing_to_relocate","github_activity_score","search_appearance_30d",
             "saved_by_recruiters_30d","interview_completion_rate","offer_acceptance_rate",
             "verified_email","verified_phone","linkedin_connected"]
    for k in order:
        print(f"     {k:<30} {s.get(k)}")
    sal = s.get("expected_salary_range_inr_lpa", {})
    print(f"     expected_salary_inr_lpa        {sal.get('min')}-{sal.get('max')}")
    print(f"  -- ranker reasoning --\n     {r['reasoning']}")
