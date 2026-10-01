#!/usr/bin/env python3
"""
make_poster_figure.py -- one figure per poster panel, p1..p6.

    python scripts/make_poster_figure.py --outdir figs
    python scripts/make_poster_figure.py --outdir figs --only overhead

Figures produced
    fig_overhead_detection   (a) scan cost vs elements observed
                             (b) detection rate by mechanism
    fig_reachability         (a) exception reachability per model
                             (b) fault outcome composition
    fig_delegation           (a) activation retention under delegation
                             (b) metadata validation on the CPU path

All three are generated at 7.16 in wide, which is \\textwidth in the
IEEEtran two-column layout. Include them with

    \\includegraphics[width=\\textwidth]{...}

inside a figure* environment so LaTeX scales by 1.0. Passing any other
width rescales the artwork and shrinks every label with it, which is the
usual cause of unreadable axis text.

Every number below is transcribed from the committed campaign reports and
the overhead measurement. Re-running a campaign means editing the tables
here, not the plotting code.
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================================================
# palette
# ==========================================================================
# Three primaries, taken from the detection-rate figure and reused
# everywhere. Extra shades are only introduced where a panel needs more
# than three series.
C1 = "#1d3f6e"   # dark navy
C2 = "#4a89c8"   # mid blue
C3 = "#c4bfae"   # warm grey
C_EXTRA_A = "#9c6b2f"   # ochre, for the CPU path
C_EXTRA_B = "#1f7a68"   # teal, for the TPU path
C_EXTRA_C = "#c2703f"   # rust, for the silent-corruption segment
C_TRACK = "#f0eee8"     # inert background track
C_RED = "#a32d2d"       # fitted curve

# ==========================================================================
# measured data
# ==========================================================================

# --- overhead: model, arm, tensors, elements, invoke ms, scan ms ----------
OVERHEAD = [
    ("MobileNetV2",  "cpu",  66,  6_898_779,  35.47,  13.59),
    ("MobileNetV2",  "tpu",   1,      1_001,   5.02,   0.36),
    ("DeepLabV3",    "cpu",  72, 50_279_352, 269.21, 104.77),
    ("DeepLabV3",    "tpu",  10,  7_508_416,  67.39,  15.90),
    ("SSD MBv2",     "cpu", 114, 13_623_198,  76.04,  31.23),
    ("SSD MBv2",     "tpu",   8,    364_351,  12.93,   3.50),
    ("EfficientDet", "cpu", 270, 27_538_499, 179.12,  69.26),
    ("EfficientDet", "tpu",  16,  8_687_959,  54.84,  31.63),
]

MARKERS = {"MobileNetV2": "o", "DeepLabV3": "^",
           "SSD MBv2": "s", "EfficientDet": "D"}

# label offsets in points; the four mid-range points sit within half a
# decade of one another and collide under automatic placement
LABEL_OFF = {
    ("MobileNetV2",  "cpu"): ( 14, -34),
    ("MobileNetV2",  "tpu"): ( 18,  -8),
    ("DeepLabV3",    "cpu"): ( 16, -12),
    ("DeepLabV3",    "tpu"): (-82, -13),
    ("SSD MBv2",     "cpu"): ( 18, -10),
    ("SSD MBv2",     "tpu"): ( 18,  -8),
    ("EfficientDet", "cpu"): (-74,  20),
    ("EfficientDet", "tpu"): (-78,   4),
}

# --- detection: label, n, detections by (flags, scan, output only) --------
DETECTION = [
    ("SSD\nCPU",    87, (87, 86, 27)),
    ("SSD\nTPU",    99, (99, 96, 24)),
    ("EffDet\nCPU", 60, (60, 60,  0)),
    ("EffDet\nTPU", 92, (92, 92, 32)),
]
MECHANISMS = ["Flag polling", "Tensor scan", "Output only"]

# --- reachability: model, trials, exception opportunities -----------------
REACH = [
    ("SSD MBv2",    240, 59),
    ("EffDet",      240, 60),
    ("MobileNetV2", 150,  1),
    ("MoveNet",     150,  0),
    ("DeepLabV3",   150,  0),
]

# --- outcomes: model, rejected, crashed, inert, silent, exception --------
OUTCOMES = [
    ("SSD MBv2",    73,  2,  44, 62, 59),
    ("EffDet",       2,  8,  84, 86, 60),
    ("MobileNetV2", 79,  3,  39, 28,  1),
    ("MoveNet",      8,  9,  63, 70,  0),
    ("DeepLabV3",   80,  1,  11, 58,  0),
]
OUT_LABELS = ["Rejected", "Crashed", "Inert", "Silent", "Exception"]
OUT_COLS = [C3, C_EXTRA_A, "#dfd9c8", C_EXTRA_C, C1]

# --- retention: model, (float cpu, tpu), (quantized cpu, tpu) ------------
# Delegation fuses the mapped partition, which is quantized throughout.
# Float activations sit at the partition boundary and on the host, so none
# of them is absorbed -- which is why the two floating-point mechanisms are
# unchanged on the accelerator while saturation coverage is not.
VISIBILITY = [
    ("SSD",       ( 6,  6), (108,  2)),
    ("EffDet",    ( 6,  6), (264, 10)),
    ("MoveNet",   (11, 11), (148, 55)),
    ("DeepLab",   ( 0,  0), ( 72, 10)),
    ("MobileNet", ( 0,  0), ( 66,  1)),
]

# --- validation: fault class -> injections rejected out of 30 ------------
# The rate on the compiled model is zero for every class and both models.
REJECT_CPU = [
    ("overflow",  16, 30),
    ("underflow", 19, 30),
    ("scale NaN", 15, 30),
    ("zp shift",   8, 14),
]

TRIALS_PER_CLASS = 30

# Below ~1e6 elements the per-tensor call overhead dominates and the scan
# is not throughput-bound; a quadratic fit in log-log space captures both
# regimes (log residual 0.093) where a straight line does not (0.166).
FIT_DEG = 2


# ==========================================================================
# helpers
# ==========================================================================

# Each panel becomes its own figure, sized for one poster slot. Generate at
# the size you will place: PowerPoint scaling a small figure up softens
# every label, and scaling a large one down makes them illegible.
W, H = 6.3, 5.0        # inches, the poster's three-per-row slot; --width / --height
DPI = 600
FONT = 24.0            # pt; poster caption size (body is 28); --font
SCALE = 2.5            # (unused) fonts relative to the paper version


def _set_geometry(w, h, dpi):
    global W, H, DPI
    W, H, DPI = w, h, dpi


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Carlito", "DejaVu Sans"],
        "font.size": FONT,
        "axes.labelsize": FONT,
        "axes.titlesize": FONT,
        "axes.titleweight": "bold",
        "xtick.labelsize": FONT,
        "ytick.labelsize": FONT,
        "legend.fontsize": FONT,
        "lines.linewidth": 1.6,
        "patch.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",   # text as outlines: identical on any machine
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def wilson(k, n, z=1.96):
    """Wilson score interval. Correct at small n and near 0 or 1, where the
    normal approximation produces bounds outside the unit interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": DPI}), ("svg", {})):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, **kw)
    print(f"wrote {outdir}/{name}.pdf, .png and .svg")
    plt.close(fig)


# ==========================================================================
# panels -- the same drawings as the paper figures, one per file
# ==========================================================================

SHORT = {"SSD MBv2": "SSD MBv2", "SSD": "SSD MBv2", "EffDet": "EfficientDet",
         "MobileNetV2": "MobileNetV2", "MobileNet": "MobileNetV2",
         "MoveNet": "MoveNet", "DeepLabV3": "DeepLabV3", "DeepLab": "DeepLabV3"}
LAB = "#333"


def num():
    """In-plot value labels sit one step below the label size."""
    return FONT - 4


def legend_below(fig, handles, ncol, y):
    fig.legend(handles=handles, frameon=False, ncol=ncol, loc="upper center",
               bbox_to_anchor=(0.5, y), handlelength=1.0, handletextpad=0.4,
               columnspacing=0.9, labelspacing=0.25)


def panel_overhead(ax):
    """Scan cost; execution path by colour, one marker."""
    off = {("MobileNetV2", "cpu"): (6, -20), ("MobileNetV2", "tpu"): (13, -6),
           ("DeepLabV3", "cpu"): (13, -4), ("DeepLabV3", "tpu"): (-60, -2),
           ("SSD MBv2", "cpu"): (8, -14), ("SSD MBv2", "tpu"): (-54, 2),
           ("EfficientDet", "cpu"): (-52, 12), ("EfficientDet", "tpu"): (-54, 6)}
    for name, arm, _, el, inv, scn in OVERHEAD:
        col = C_EXTRA_A if arm == "cpu" else C_EXTRA_B
        ax.scatter([el], [scn], s=90, marker=MARKERS[name], facecolors="none",
                   edgecolors=col, linewidths=2.0, zorder=3)
        ax.annotate(f"{100 * scn / inv:.0f}%", (el, scn), fontsize=num(),
                    color=LAB, textcoords="offset points", xytext=off[(name, arm)])
    x = np.array([r[3] for r in OVERHEAD], float)
    y = np.array([r[5] for r in OVERHEAD], float)
    coef = np.polyfit(np.log10(x), np.log10(y), FIT_DEG)
    grid = np.logspace(np.log10(x.min() * 0.55), np.log10(x.max() * 2.2), 300)
    ax.plot(grid, 10 ** np.polyval(coef, np.log10(grid)), c=C_RED, lw=2.2, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x.min() * 0.16, x.max() * 30)
    ax.set_ylim(y.min() * 0.18, y.max() * 9.0)
    ax.set_xticks([1e4, 1e6, 1e8]); ax.set_yticks([1e-1, 1e1, 1e3])
    ax.set_xlabel("Elements scanned per inference", labelpad=2)
    ax.set_ylabel("Scan time (ms)", labelpad=2)
    ax.set_title("Instrumentation cost", loc="left", pad=6)
    ax.grid(True, which="major", ls=":", lw=0.6, c="#d0d0d0", zorder=0)
    despine(ax)
    hs = [Line2D([], [], ls="", marker=MARKERS[m], mfc="none", mec="#555",
                 mew=2.0, ms=9, label=m) for m in MARKERS]
    hs += [Line2D([], [], ls="", marker="o", mfc="none", mec=C_EXTRA_A, mew=2.0,
                  ms=9, label="CPU path"),
           Line2D([], [], ls="", marker="o", mfc="none", mec=C_EXTRA_B, mew=2.0,
                  ms=9, label="TPU path")]
    return hs


def panel_detection(ax):
    """Vertical bars: at poster size a 20 pt label needs the horizontal room
    above a bar; three stacked horizontal bars per group are too thin."""
    idx = np.arange(len(DETECTION)); width = 0.27
    for j, colour in enumerate((C1, C2, C3)):
        rates = [wilson(ks[j], n)[0] * 100 for _, n, ks in DETECTION]
        pos = idx + (j - 1) * width
        ax.bar(pos, rates, width * 0.9, color=colour, zorder=3)
        dy = (3.0, 19.0, 3.0)[j]     # middle label one row higher: no collision
        for xp, r in zip(pos, rates):
            ax.text(xp, r + dy, f"{r:.0f}", ha="center", fontsize=num(), color=LAB)
    ax.set_xticks(idx)
    ax.set_xticklabels([lab for lab, _, _ in DETECTION])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 140); ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Detected (%)", labelpad=2)
    ax.set_title("Detection rate by mechanism", loc="left", pad=6)
    ax.grid(True, axis="y", ls=":", lw=0.6, c="#d0d0d0", zorder=0)
    despine(ax)
    return [Patch(facecolor=c, label=l)
            for c, l in zip((C1, C2, C3), MECHANISMS)]


def panel_reach(ax):
    labels = [SHORT[r[0]] for r in REACH]
    pct = [100 * r[2] / r[1] for r in REACH]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, pct, 0.62, color=[C1 if p > 0 else C3 for p in pct], zorder=3)
    for yi, p in zip(y, pct):
        ax.text(p + 0.8, yi, f"{p:.1f}%", va="center", fontsize=num(), color=LAB)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 42); ax.set_xticks([0, 20, 40])
    ax.set_xlabel("Trials raising an exception (%)", labelpad=2)
    ax.set_title("Exception reachability", loc="left", pad=6)
    ax.grid(True, axis="x", ls=":", lw=0.6, c="#d0d0d0", zorder=0)
    despine(ax)
    return []


def panel_outcomes(ax):
    labels = [SHORT[r[0]] for r in OUTCOMES]
    data = np.array([r[1:] for r in OUTCOMES], float)
    frac = 100 * data / data.sum(axis=1, keepdims=True)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for j, (lab, col) in enumerate(zip(OUT_LABELS, OUT_COLS)):
        ax.barh(y, frac[:, j], 0.62, left=left, color=col, zorder=3,
                edgecolor="white", linewidth=0.5)
        for yi, f, l in zip(y, frac[:, j], left):
            if f >= 15:
                ax.text(l + f / 2, yi, f"{f:.0f}", ha="center", va="center",
                        fontsize=num(), color="white" if j in (1, 3, 4) else "#3a3a3a")
        left += frac[:, j]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Share of injected faults (%)", labelpad=2)
    ax.set_title("Fault outcome composition", loc="left", pad=6)
    despine(ax)
    hs = [Patch(facecolor=c, label=l) for l, c in zip(OUT_LABELS, OUT_COLS)]
    return [hs[0], hs[3], hs[1], hs[4], hs[2]]   # column-filled -> reads row-wise


def panel_visibility(ax):
    labels = [SHORT[r[0]] for r in VISIBILITY]
    y = np.arange(len(labels))[::-1]; h = 0.36
    for yi, (_, (fc, ft), (qc, qt)) in zip(y, VISIBILITY):
        ax.barh(yi - h / 2, 100 * qt / qc, h, color=C2, zorder=3)
        ax.text(100 * qt / qc + 3, yi - h / 2, f"{qt} of {qc}", va="center",
                fontsize=num(), color=LAB)
        if fc:
            ax.barh(yi + h / 2, 100 * ft / fc, h, color=C1, zorder=3)
            ax.text(103, yi + h / 2, f"{ft} of {fc}", va="center", fontsize=num(), color=LAB)
        else:
            ax.text(103, yi + h / 2, "no float32", va="center", fontsize=num(), color="#999")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 150); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Activations retained (%)", labelpad=2)
    ax.set_title("Observability under delegation", loc="left", pad=6)
    ax.grid(True, axis="x", ls=":", lw=0.6, c="#d0d0d0", zorder=0)
    despine(ax)
    return [Patch(facecolor=C1, label="float32"),
            Patch(facecolor=C2, label="quantized")]


def panel_validation(ax):
    classes = [r[0] for r in REJECT_CPU]
    yb = np.arange(len(classes))[::-1]; w = 0.36
    for k, (model, colour, col_i) in enumerate((("SSD", C1, 1), ("EffDet", C2, 2))):
        vals = [100 * r[col_i] / TRIALS_PER_CLASS for r in REJECT_CPU]
        pos = yb + (0.5 - k) * w
        ax.barh(pos, vals, w * 0.9, color=colour, zorder=3)
        for yp, v in zip(pos, vals):
            ax.text(v + 3, yp, f"{v:.0f}", va="center", fontsize=num(), color=LAB)
    ax.set_yticks(yb); ax.set_yticklabels(classes)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 135); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Injections rejected (%)", labelpad=2)
    ax.set_title("Metadata validation, CPU path", loc="left", pad=6)
    ax.grid(True, axis="x", ls=":", lw=0.6, c="#d0d0d0", zorder=0)
    despine(ax)
    return [Patch(facecolor=C1, label="SSD"), Patch(facecolor=C2, label="EffDet")]


def _single(outdir, name, painter, adj, ncol, legend_y):
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    handles = painter(ax)
    if handles:
        legend_below(fig, handles, ncol, legend_y)
    fig.subplots_adjust(**adj)
    save(fig, outdir, name)


# margins are for a 6.3 x 4.6 in panel at 24 pt, the poster's three-per-row slot
def fig_overhead(outdir):
    _single(outdir, "p1_overhead", panel_overhead,
            dict(left=0.19, right=0.97, top=0.90, bottom=0.46), 2, 0.28)


def fig_detection(outdir):
    _single(outdir, "p2_detection", panel_detection,
            dict(left=0.17, right=0.98, top=0.88, bottom=0.44), 2, 0.24)
    print("\nWilson 95% intervals, for the caption")
    for lab, n, ks in DETECTION:
        cells = []
        for mech, k in zip(MECHANISMS, ks):
            p, lo, hi = wilson(k, n)
            cells.append(f"{mech} {p*100:.1f}% [{lo*100:.0f},{hi*100:.0f}]")
        print(f"  {lab.replace(chr(10), ' '):<12} n={n:<4} " + " | ".join(cells))
    print()


def fig_reach(outdir):
    _single(outdir, "p3_reachability", panel_reach,
            dict(left=0.31, right=0.96, top=0.88, bottom=0.24), 1, 0.0)


def fig_outcomes(outdir):
    _single(outdir, "p4_outcomes", panel_outcomes,
            dict(left=0.31, right=0.96, top=0.88, bottom=0.38), 3, 0.23)


def fig_visibility(outdir):
    _single(outdir, "p5_visibility", panel_visibility,
            dict(left=0.31, right=0.97, top=0.88, bottom=0.30), 2, 0.14)


def fig_validation(outdir):
    _single(outdir, "p6_validation", panel_validation,
            dict(left=0.30, right=0.97, top=0.88, bottom=0.30), 2, 0.14)


FIGURES = {
    "overhead":    fig_overhead,
    "detection":   fig_detection,
    "reachability": fig_reach,
    "outcomes":    fig_outcomes,
    "visibility":  fig_visibility,
    "validation":  fig_validation,
}


def main():
    global FONT
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figs_poster")
    ap.add_argument("--width", type=float, default=W,
                    help="panel width in inches, as placed on the poster")
    ap.add_argument("--height", type=float, default=H)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--font", type=float, default=FONT,
                    help="pt for every label; 28 = poster body, 24 = poster caption")
    ap.add_argument("--only", choices=sorted(FIGURES),
                    help="generate a single figure instead of all three")
    args = ap.parse_args()

    _set_geometry(args.width, args.height, args.dpi)
    FONT = args.font
    style()
    names = [args.only] if args.only else list(FIGURES)
    for n in names:
        FIGURES[n](args.outdir)


if __name__ == "__main__":
    main()