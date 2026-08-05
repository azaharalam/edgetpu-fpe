import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


C1 = "#1d3f6e"   # dark navy
C2 = "#4a89c8"   # mid blue
C3 = "#c4bfae"   # warm grey
C_EXTRA_A = "#9c6b2f"   # ochre, for the CPU path
C_EXTRA_B = "#1f7a68"   # teal, for the TPU path
C_EXTRA_C = "#c2703f"   # rust, for the silent-corruption segment
C_TRACK = "#f0eee8"     # inert background track
C_RED = "#a32d2d"       # fitted curve


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

LABEL_OFF = {
    ("MobileNetV2",  "cpu"): (  2, -16),
    ("MobileNetV2",  "tpu"): (  8,  -3),
    ("DeepLabV3",    "cpu"): (  7,   3),
    ("DeepLabV3",    "tpu"): (-24,   8),
    ("SSD MBv2",     "cpu"): (  8,  -4),
    ("SSD MBv2",     "tpu"): (  8,  -3),
    ("EfficientDet", "cpu"): (-34,   7),
    ("EfficientDet", "tpu"): (-27, -13),
}

DETECTION = [
    ("SSD\nCPU",    87, (87, 86, 27)),
    ("SSD\nTPU",    99, (99, 96, 24)),
    ("EffDet\nCPU", 60, (60, 60,  0)),
    ("EffDet\nTPU", 92, (92, 92, 32)),
]
MECHANISMS = ["Flag polling", "Tensor scan", "Output only"]

REACH = [
    ("SSD MBv2",    240, 59),
    ("EffDet",      240, 60),
    ("MobileNetV2", 150,  1),
    ("MoveNet",     150,  0),
    ("DeepLabV3",   150,  0),
]

OUTCOMES = [
    ("SSD MBv2",    88,  2,  39, 52, 59),
    ("EffDet",       2,  8, 114, 56, 60),
    ("MobileNetV2", 79,  3,  39, 28,  1),
    ("MoveNet",      8, 11,  63, 68,  0),
    ("DeepLabV3",   75,  2,  15, 58,  0),
]
OUT_LABELS = ["Rejected", "Crashed", "Inert", "Silent", "Exception"]
OUT_COLS = [C3, C_EXTRA_A, "#dfd9c8", C_EXTRA_C, C1]

VISIBILITY = [
    ("SSD",       ( 6,  6), (108,  2)),
    ("EffDet",    ( 6,  6), (264, 10)),
    ("MoveNet",   (11, 11), (148, 55)),
    ("DeepLab",   ( 0,  0), ( 72, 10)),
    ("MobileNet", ( 0,  0), ( 66,  1)),
]

REJECT_CPU = [
    ("overflow",  16, 30),
    ("underflow", 19, 30),
    ("scale NaN", 15, 30),
    ("zp shift",   8, 14),
]

TRIALS_PER_CLASS = 30

FIT_DEG = 2


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,     # embed TrueType rather than Type 3
        "ps.fonttype": 42,
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def wilson(k, n, z=1.96):

    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, **kw)
    print(f"wrote {outdir}/{name}.pdf and .png")
    plt.close(fig)




def panel_overhead(ax):
    """Model is encoded by marker, execution path by colour. Only the
    overhead percentage is annotated; the mid-range points are too close
    together to carry model names as well."""
    for name, arm, _, el, inv, scn in OVERHEAD:
        ax.scatter([el], [scn], s=34, marker=MARKERS[name],
                   facecolors="none", linewidths=1.4, zorder=3,
                   edgecolors=(C_EXTRA_A if arm == "cpu" else C_EXTRA_B))
        ax.annotate(f"{100 * scn / inv:.0f}%", (el, scn),
                    textcoords="offset points", xytext=LABEL_OFF[(name, arm)],
                    fontsize=8.5, color="#333")

    x = np.array([r[3] for r in OVERHEAD], float)
    y = np.array([r[5] for r in OVERHEAD], float)
    coef = np.polyfit(np.log10(x), np.log10(y), FIT_DEG)
    grid = np.logspace(np.log10(x.min() * 0.55), np.log10(x.max() * 2.2), 300)
    ax.plot(grid, 10 ** np.polyval(coef, np.log10(grid)),
            "-", c=C_RED, lw=1.3, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x.min() * 0.22, x.max() * 9.0)
    ax.set_ylim(y.min() * 0.24, y.max() * 5.5)
    ax.set_xlabel("Elements scanned per inference", labelpad=2)
    ax.set_ylabel("Scan time (ms)", labelpad=2)
    ax.set_title("(a) Instrumentation cost", fontsize=9.5, pad=5, loc="left")
    ax.grid(True, which="major", ls=":", lw=0.5, c="#d0d0d0", zorder=0)

    handles = [Line2D([], [], ls="", marker=MARKERS[m], mfc="none",
                      mec="#555", mew=1.3, ms=5, label=m) for m in MARKERS]
    handles += [Line2D([], [], ls="", marker="o", mfc="none",
                       mec=C_EXTRA_A, mew=1.3, ms=5, label="CPU path"),
                Line2D([], [], ls="", marker="o", mfc="none",
                       mec=C_EXTRA_B, mew=1.3, ms=5, label="TPU path")]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.46),
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
              labelspacing=0.35)
    despine(ax)


def panel_detection(ax):
    idx = np.arange(len(DETECTION))
    width = 0.26
    for j, (mech, colour) in enumerate(zip(MECHANISMS, (C1, C2, C3))):
        rates = [wilson(ks[j], n)[0] * 100 for _, n, ks in DETECTION]
        pos = idx + (j - 1) * width
        ax.bar(pos, rates, width * 0.9, color=colour, zorder=3,
               label=mech, edgecolor="white", linewidth=0.5)
        # adjacent bars sit close; stagger the labels so the three-digit
        # ones do not collide
        dy = (3.0, 9.5, 3.0)[j]
        for xp, r in zip(pos, rates):
            ax.text(xp, r + dy, f"{r:.0f}", ha="center", fontsize=9,
                    color="#333")

    ax.set_xticks(idx)
    ax.set_xticklabels([lab for lab, _, _ in DETECTION], fontsize=9.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 126)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Detected (%)", labelpad=2)
    ax.set_title("(b) Detection rate by mechanism", fontsize=9.5, pad=5,
                 loc="left")
    ax.grid(True, axis="y", ls=":", lw=0.5, c="#d0d0d0", zorder=0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.46), handlelength=1.0,
              handletextpad=0.35, columnspacing=0.7)
    despine(ax)


def fig_overhead_detection(outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.25))
    panel_overhead(a)
    panel_detection(b)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.925, bottom=0.31,
                        wspace=0.26)
    save(fig, outdir, "fig_overhead_detection")

    print("\nWilson 95% intervals, for the caption")
    for lab, n, ks in DETECTION:
        cells = []
        for mech, k in zip(MECHANISMS, ks):
            p, lo, hi = wilson(k, n)
            cells.append(f"{mech} {p*100:.1f}% [{lo*100:.0f},{hi*100:.0f}]")
        flat = lab.replace("\n", " ")
        print(f"  {flat:<12} n={n:<4} " + " | ".join(cells))
    print()




def panel_reach(ax):
    """Normalised: the detection models ran 240 trials and the controls
    150, so raw counts would not be comparable."""
    labels = [r[0] for r in REACH]
    pct = [100 * r[2] / r[1] for r in REACH]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, pct, 0.62, color=[C1 if p > 0 else C3 for p in pct], zorder=3)
    for yi, p in zip(y, pct):
        ax.text(p + 0.7, yi, f"{p:.1f}%", va="center", fontsize=9,
                color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 31)
    ax.set_xlabel("Trials raising an exception (%)", labelpad=2)
    ax.set_title("(a) Exception reachability", fontsize=9.5, pad=5,
                 loc="left")
    ax.grid(True, axis="x", ls=":", lw=0.5, c="#d0d0d0", zorder=0)
    despine(ax)


def panel_outcomes(ax):
    labels = [r[0] for r in OUTCOMES]
    data = np.array([r[1:] for r in OUTCOMES], float)
    frac = 100 * data / data.sum(axis=1, keepdims=True)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for j, (lab, col) in enumerate(zip(OUT_LABELS, OUT_COLS)):
        ax.barh(y, frac[:, j], 0.62, left=left, color=col, zorder=3,
                label=lab, edgecolor="white", linewidth=0.5)
        for yi, f, l in zip(y, frac[:, j], left):
            if f >= 11:
                ax.text(l + f / 2, yi, f"{f:.0f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if j in (1, 3, 4) else "#3a3a3a")
        left += frac[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of injected faults (%)", labelpad=2)
    ax.set_title("(b) Fault outcome composition", fontsize=9.5, pad=5,
                 loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.36, -0.40), handlelength=0.9,
              handletextpad=0.3, columnspacing=0.6)
    despine(ax)


def fig_reachability(outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.9))
    panel_reach(a)
    panel_outcomes(b)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.92, bottom=0.30,
                        wspace=0.46)
    save(fig, outdir, "fig_reachability")




def panel_visibility(ax):
    """Retention of each activation class under delegation. Float
    activations are unaffected in every model, which is why the two
    floating-point mechanisms are unchanged on the accelerator; the
    quantized interior is what the fused partition absorbs, and with it
    the saturation surface."""
    labels = [r[0] for r in VISIBILITY]
    y = np.arange(len(labels))[::-1]
    h = 0.34

    for yi, (_, (fc, ft), (qc, qt)) in zip(y, VISIBILITY):
        qpct = 100 * qt / qc
        ax.barh(yi - h / 2, qpct, h, color=C2, zorder=3)
        ax.text(qpct + 2.5, yi - h / 2, f"{qt} of {qc}", va="center",
                fontsize=8.5, color="#333")
        if fc:
            ax.barh(yi + h / 2, 100 * ft / fc, h, color=C1, zorder=3)
            ax.text(102.5, yi + h / 2, f"{ft} of {fc}", va="center",
                    fontsize=8.5, color="#333")
        else:
            ax.text(2.0, yi + h / 2, "no float activations", va="center",
                    fontsize=8.5, color="#999")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Activations retained (%)", labelpad=2)
    ax.set_title("(a) Observability under delegation", fontsize=9.5, pad=5,
                 loc="left")
    ax.grid(True, axis="x", ls=":", lw=0.5, c="#d0d0d0", zorder=0)
    ax.legend(handles=[Patch(facecolor=C1, label="float32"),
                       Patch(facecolor=C2, label="quantized")],
              frameon=False, fontsize=9, ncol=2, loc="lower center",
              bbox_to_anchor=(0.42, -0.40), handlelength=1.1,
              handletextpad=0.4, columnspacing=1.0)
    despine(ax)


def panel_validation(ax):
    """Rejection rate on the CPU path, grouped by fault class. The rate on
    the compiled model is zero in every case, so only the CPU-path bars are
    drawn and the zero is stated in the caption."""
    classes = [r[0] for r in REJECT_CPU]
    idx = np.arange(len(classes))
    w = 0.34
    for k, (model, colour, col_i) in enumerate((("SSD", C1, 1),
                                                ("EffDet", C2, 2))):
        vals = [100 * r[col_i] / TRIALS_PER_CLASS for r in REJECT_CPU]
        pos = idx + (k - 0.5) * w
        ax.bar(pos, vals, w * 0.88, color=colour, zorder=3, label=model)
        for xp, v in zip(pos, vals):
            ax.text(xp, v + 2.5, f"{v:.0f}", ha="center", fontsize=8.5,
                    color="#333")

    ax.set_xticks(idx)
    ax.set_xticklabels(classes, fontsize=8.2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Injections rejected (%)", labelpad=2)
    ax.set_title("(b) Metadata validation, CPU path", fontsize=9.5, pad=5,
                 loc="left")
    ax.grid(True, axis="y", ls=":", lw=0.5, c="#d0d0d0", zorder=0)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.40), handlelength=1.1,
              handletextpad=0.4, columnspacing=1.2)
    despine(ax)


def fig_delegation(outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.1))
    panel_visibility(a)
    panel_validation(b)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.92, bottom=0.29,
                        wspace=0.55)
    save(fig, outdir, "fig_delegation")


# ==========================================================================

FIGURES = {
    "overhead": fig_overhead_detection,
    "reachability": fig_reachability,
    "delegation": fig_delegation,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--only", choices=sorted(FIGURES),
                    help="generate a single figure instead of all three")
    args = ap.parse_args()

    style()
    names = [args.only] if args.only else list(FIGURES)
    for n in names:
        FIGURES[n](args.outdir)


if __name__ == "__main__":
    main()