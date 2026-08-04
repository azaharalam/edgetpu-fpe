#!/usr/bin/env python3
"""
make_report.py -- turn campaign JSON files into poster-ready tables.

Reads one or more campaign_*.json files produced by run_campaign.py and emits
every table and figure series the poster needs, in markdown and LaTeX, so the
numbers come from the runs rather than being transcribed by hand.

Outputs
    Table 1  outcome breakdown per fault class, per model
    Table 2  detection rate by mechanism, with Wilson intervals
    Table 3  IEEE 754 coverage, with honest status per exception class
    Table 4  cross-model exception reachability
    fig1.csv detection-rate bars (the headline figure)
    fig2.csv reachability across models

Usage
    python make_report.py *_campaign.json --out report
"""

import argparse
import csv
import json
import math
import os

MEANINGFUL = ("invalid", "divbyzero", "overflow", "underflow")
STATUSES = ["REJECTED", "CRASHED", "NOT_ACTIVATED", "ACTIVATED_OUTPUT",
            "ACTIVATED_EXCEPTION", "ACTIVATED_BOTH"]
SHORT = {"REJECTED": "rej", "CRASHED": "crash", "NOT_ACTIVATED": "inert",
         "ACTIVATED_OUTPUT": "silent", "ACTIVATED_EXCEPTION": "exc-masked",
         "ACTIVATED_BOTH": "exc+out"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load_all(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        d["_file"] = os.path.basename(p)
        out.append(d)
    return out


def opportunities(rows):
    """Trials that actually raised an IEEE exception."""
    return [r for r in rows
            if r["status"] in ("ACTIVATED_EXCEPTION", "ACTIVATED_BOTH")]


def table_outcomes(campaigns):
    lines = ["## Table 1 — Fault outcome breakdown", ""]
    head = ("| model | fault | n | " +
            " | ".join(SHORT[s] for s in STATUSES) + " |")
    lines += [head, "|" + "---|" * (len(STATUSES) + 3)]
    for c in campaigns:
        rows = c["rows"]
        for f in sorted({r["fault"] for r in rows}):
            sub = [r for r in rows if r["fault"] == f]
            counts = [sum(r["status"] == s for r in sub) for s in STATUSES]
            lines.append(f"| {c['model'][:28]} | {f} | {len(sub)} | " +
                         " | ".join(str(x) for x in counts) + " |")
    return "\n".join(lines)


def table_detection(campaigns):
    lines = ["## Table 2 — Detection rate by mechanism", "",
             "Denominator: trials that raised an IEEE exception.", "",
             "| model | n | flag polling | tensor scan | output only |",
             "|---|---|---|---|---|"]
    series = []
    for c in campaigns:
        opp = opportunities(c["rows"])
        if not opp:
            lines.append(f"| {c['model'][:28]} | 0 | — | — | — |")
            continue
        cells = []
        for key in ("det_flags", "det_scan", "det_output_only"):
            k = sum(r[key] for r in opp)
            p, lo, hi = wilson(k, len(opp))
            cells.append(f"{p*100:.1f}% [{lo*100:.0f}–{hi*100:.0f}]")
            series.append({"model": c["model"], "mechanism": key,
                           "k": k, "n": len(opp), "rate": p,
                           "ci_lo": lo, "ci_hi": hi})
        lines.append(f"| {c['model'][:28]} | {len(opp)} | " +
                     " | ".join(cells) + " |")
    return "\n".join(lines), series


def table_coverage(campaigns):
    """IEEE 754 coverage with honest status per class."""
    seen = {c: 0 for c in MEANINGFUL}
    det = {c: 0 for c in MEANINGFUL}
    for c in campaigns:
        for r in opportunities(c["rows"]):
            for cls in r["exceptions"].split(";"):
                if cls in seen:
                    seen[cls] += 1
                    det[cls] += bool(r["det_flags"])

    lines = ["## Table 3 — IEEE 754 exception coverage", "",
             "| exception | opportunities | detected | status |",
             "|---|---|---|---|"]
    for cls in MEANINGFUL:
        n, k = seen[cls], det[cls]
        if n == 0:
            status = "reachable in principle; not observed in this campaign"
        elif k == n:
            status = "demonstrated, full detection"
        else:
            status = f"demonstrated, {k}/{n} detected"
        lines.append(f"| {cls} | {n} | {k} | {status} |")
    lines.append("| inexact | — | — | detectable but excluded by design "
                 "(fires on ~all float ops) |")
    return "\n".join(lines)


def table_reachability(campaigns):
    lines = ["## Table 4 — Exception reachability across models", "",
             "| model | trials | exception opportunities | silent corruption |",
             "|---|---|---|---|"]
    series = []
    for c in campaigns:
        rows = c["rows"]
        opp = len(opportunities(rows))
        silent = sum(r["status"] == "ACTIVATED_OUTPUT" for r in rows)
        lines.append(f"| {c['model'][:34]} | {len(rows)} | {opp} | {silent} |")
        series.append({"model": c["model"], "trials": len(rows),
                       "opportunities": opp, "silent": silent})
    return "\n".join(lines), series


def headline(campaigns):
    all_opp, all_rows = [], []
    for c in campaigns:
        all_opp += opportunities(c["rows"])
        all_rows += c["rows"]
    if not all_opp:
        return "No exception opportunities across campaigns."
    n = len(all_opp)
    flags = sum(r["det_flags"] for r in all_opp)
    scan = sum(r["det_scan"] for r in all_opp)
    out = sum(r["det_output_only"] for r in all_opp)
    masked = sum(r["status"] == "ACTIVATED_EXCEPTION" for r in all_opp)
    silent = sum(r["status"] == "ACTIVATED_OUTPUT" for r in all_rows)
    corrupting = silent + n

    return (
        f"## Headline\n\n"
        f"Across {len(all_rows)} injected faults in {len(campaigns)} models, "
        f"{corrupting} corrupted the output. Of those, {silent} "
        f"({silent/corrupting*100:.0f}%) raised no IEEE exception at all — "
        f"silent fixed-point corruption. Of the {n} that did raise an "
        f"exception, {masked} ({masked/n*100:.0f}%) were masked before the "
        f"output.\n\n"
        f"Detection over the {n} exception opportunities: "
        f"flag polling {flags}/{n} ({flags/n*100:.1f}%), "
        f"tensor scanning {scan}/{n} ({scan/n*100:.1f}%), "
        f"output-only inspection {out}/{n} ({out/n*100:.1f}%)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="+")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()

    cs = load_all(args.campaigns)
    os.makedirs(args.out, exist_ok=True)

    det_md, det_series = table_detection(cs)
    reach_md, reach_series = table_reachability(cs)

    doc = "\n\n".join([
        f"# FPE detection campaign report",
        headline(cs),
        table_outcomes(cs),
        det_md,
        table_coverage(cs),
        reach_md,
    ])

    md_path = os.path.join(args.out, "report.md")
    with open(md_path, "w") as fh:
        fh.write(doc + "\n")

    with open(os.path.join(args.out, "fig1_detection.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(det_series[0].keys())
                           if det_series else ["model"])
        w.writeheader()
        w.writerows(det_series)

    with open(os.path.join(args.out, "fig2_reachability.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(reach_series[0].keys()))
        w.writeheader()
        w.writerows(reach_series)

    print(doc)
    print(f"\n\nwrote {md_path}, fig1_detection.csv, fig2_reachability.csv")


if __name__ == "__main__":
    main()
