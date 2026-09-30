#!/usr/bin/env python3
"""
make_paper_figures.py -- the three two-panel figures for the IEEEtran paper.

    python3 scripts/make_paper_figures.py --outdir figures

Geometry: one column wide (3.5 in), two panels side by side. All text is
the paper's 10 pt body size; the in-plot value labels are 8 pt (IEEE
caption size), the one concession that lets the numbers fit. Include with

    \\includegraphics[width=\\columnwidth]{figures/fig_overhead_detection.pdf}

so LaTeX scales by 1.0.

Every data table is imported from make_poster_figure.py, so the paper and
the poster are drawn from one copy of the numbers.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import make_poster_figure as P     # data tables and palette only

FONT = 8.0      # pt, IEEEtran caption size: titles, axes, ticks, legends
NUM = 7.0       # pt, in-plot value labels only
W = 3.5         # in, \columnwidth
H = 1.75        # in
LAB = "#333"


def style():
    plt.rcParams.update({
        # the paper is set in Times; STIX is metric-compatible and ships
        # with matplotlib, so the figure text matches the body text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
        "axes.titleweight": "bold", "xtick.labelsize": FONT,
        "ytick.labelsize": FONT, "legend.fontsize": FONT,
        "lines.linewidth": 1.2, "patch.linewidth": 0.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def title(ax, text):
    # one-line titles hang from the panel edge nearest the figure edge, so a
    # long (b) title grows into the gap between panels instead of off the
    # figure; a two-line title is centred over its panel
    if "\n" in text:
        ax.set_title(text, loc="center", pad=3, multialignment="center")
    else:
        ax.set_title(text, loc="right" if text.startswith("(b)") else "left", pad=3)


def shared_legend(fig, handles, ncol, y):
    fig.legend(handles=handles, frameon=False, ncol=ncol, loc="upper center",
               bbox_to_anchor=(0.5, y), handlelength=1.0, handletextpad=0.4,
               columnspacing=0.9, labelspacing=0.2)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=600)
    print(f"wrote {outdir}/{name}.pdf and .png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_overhead_detection(outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H + 0.1))

    # (a) scan cost; path by colour, one marker
    off = {("MobileNetV2", "cpu"): (3, -10), ("MobileNetV2", "tpu"): (4, -3),
           ("DeepLabV3", "cpu"): (4, -2), ("DeepLabV3", "tpu"): (-21, -8),
           ("SSD MBv2", "cpu"): (4, -7), ("SSD MBv2", "tpu"): (-21, 1),
           ("EfficientDet", "cpu"): (-19, 5), ("EfficientDet", "tpu"): (-21, 3)}
    for name, arm, _, el, inv, scn in P.OVERHEAD:
        col = P.C_EXTRA_A if arm == "cpu" else P.C_EXTRA_B
        a.scatter([el], [scn], s=14, marker="o", facecolors="none",
                  edgecolors=col, linewidths=1.0, zorder=3)
        a.annotate(f"{100 * scn / inv:.0f}%", (el, scn), fontsize=NUM,
                   color=LAB, textcoords="offset points", xytext=off[(name, arm)])
    x = np.array([r[3] for r in P.OVERHEAD], float)
    y = np.array([r[5] for r in P.OVERHEAD], float)
    coef = np.polyfit(np.log10(x), np.log10(y), P.FIT_DEG)
    grid = np.logspace(np.log10(x.min() * 0.55), np.log10(x.max() * 2.2), 300)
    a.plot(grid, 10 ** np.polyval(coef, np.log10(grid)), c=P.C_RED, lw=1.2, zorder=2)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(x.min() * 0.16, x.max() * 30)
    a.set_ylim(y.min() * 0.18, y.max() * 9.0)
    a.set_xticks([1e4, 1e6, 1e8]); a.set_yticks([1e-1, 1e1, 1e3])
    a.set_xlabel("Elements scanned", labelpad=1)
    a.set_ylabel("Scan time (ms)", labelpad=1)
    a.grid(True, which="major", ls=":", lw=0.4, c="#d0d0d0", zorder=0)
    title(a, "(a) Instrumentation cost")
    despine(a)

    # (b) detection rate, horizontal so the four arm labels have room
    idx = np.arange(len(P.DETECTION))[::-1]; width = 0.30
    for j, colour in enumerate((P.C1, P.C2, P.C3)):
        rates = [P.wilson(ks[j], n)[0] * 100 for _, n, ks in P.DETECTION]
        pos = idx + (1 - j) * width
        b.barh(pos, rates, width * 0.9, color=colour, zorder=3)
        for yp, r in zip(pos, rates):
            b.text(r + 3, yp, f"{r:.0f}", va="center", fontsize=NUM, color=LAB)
    b.set_yticks(idx)
    b.set_yticklabels([lab.replace("\n", " ") for lab, _, _ in P.DETECTION])
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 135); b.set_xticks([0, 50, 100])
    b.set_xlabel("Detected (%)", labelpad=1)
    b.grid(True, axis="x", ls=":", lw=0.4, c="#d0d0d0", zorder=0)
    title(b, "(b) Detection rate by mechanism")
    despine(b)

    handles = [Line2D([], [], ls="", marker="o", mfc="none", mec=P.C_EXTRA_A,
                      mew=1.0, ms=4, label="CPU path"),
               Line2D([], [], ls="", marker="o", mfc="none", mec=P.C_EXTRA_B,
                      mew=1.0, ms=4, label="TPU path"),
               Patch(facecolor=P.C1, label="Flag polling"),
               Patch(facecolor=P.C2, label="Tensor scan"),
               Patch(facecolor=P.C3, label="Output only")]
    shared_legend(fig, handles, 3, 0.21)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.40, wspace=0.62)
    save(fig, outdir, "fig_overhead_detection")


# ---------------------------------------------------------------- figure 3
SHORT = {"SSD MBv2": "SSD", "EffDet": "EffDet", "MobileNetV2": "MNv2",
         "MoveNet": "MoveNet", "DeepLabV3": "DeepLab"}


def fig_reachability(outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    labels = [SHORT[r[0]] for r in P.REACH]
    y = np.arange(len(labels))[::-1]

    # (a) reachability
    pct = [100 * r[2] / r[1] for r in P.REACH]
    a.barh(y, pct, 0.62, color=[P.C1 if p > 0 else P.C3 for p in pct], zorder=3)
    for yi, p in zip(y, pct):
        a.text(p + 0.8, yi, f"{p:.1f}%", va="center", fontsize=NUM, color=LAB)
    a.set_yticks(y); a.set_yticklabels(labels)
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, 42); a.set_xticks([0, 20, 40])
    a.set_xlabel("Exceptions (%)", labelpad=1)
    a.grid(True, axis="x", ls=":", lw=0.4, c="#d0d0d0", zorder=0)
    title(a, "(a) Exception reachability")
    despine(a)

    # (b) outcome composition, rows shared with (a)
    data = np.array([r[1:] for r in P.OUTCOMES], float)
    frac = 100 * data / data.sum(axis=1, keepdims=True)
    left = np.zeros(len(labels))
    for j, col in enumerate(P.OUT_COLS):
        b.barh(y, frac[:, j], 0.62, left=left, color=col, zorder=3,
               edgecolor="white", linewidth=0.4)
        for yi, f, l in zip(y, frac[:, j], left):
            if f >= 15:
                b.text(l + f / 2, yi, f"{f:.0f}", ha="center", va="center",
                       fontsize=NUM, color="white" if j in (1, 3, 4) else "#3a3a3a")
        left += frac[:, j]
    b.set_xlim(0, 100); b.set_xticks([0, 50, 100])
    b.set_xlabel("Faults (%)", labelpad=1)
    b.tick_params(axis="y", length=0)
    title(b, "(b) Fault outcome composition")
    despine(b)

    handles = [Patch(facecolor=c, label=l) for l, c in zip(P.OUT_LABELS, P.OUT_COLS)]
    shared_legend(fig, handles, 5, 0.14)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.36, wspace=0.10)
    save(fig, outdir, "fig_reachability")


# ---------------------------------------------------------------- figure 4
def fig_delegation(outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H + 0.32))

    # (a) activation retention
    short = {"SSD": "SSD", "EffDet": "EffDet", "MoveNet": "MoveNet",
             "DeepLab": "DeepLab", "MobileNet": "MNv2"}
    labels = [short[r[0]] for r in P.VISIBILITY]
    y = np.arange(len(labels))[::-1]; h = 0.36
    for yi, (_, (fc, ft), (qc, qt)) in zip(y, P.VISIBILITY):
        a.barh(yi - h / 2, 100 * qt / qc, h, color=P.C2, zorder=3)
        a.text(100 * qt / qc + 3, yi - h / 2, f"{qt}/{qc}", va="center",
               fontsize=NUM, color=LAB)
        if fc:
            a.barh(yi + h / 2, 100 * ft / fc, h, color=P.C1, zorder=3)
            a.text(103, yi + h / 2, f"{ft}/{fc}", va="center", fontsize=NUM, color=LAB)
        else:
            a.text(3, yi + h / 2, "none", va="center", fontsize=NUM, color="#999")
    a.set_yticks(y); a.set_yticklabels(labels)
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, 150); a.set_xticks([0, 50, 100])
    a.set_xlabel("Retained (%)", labelpad=1)
    a.grid(True, axis="x", ls=":", lw=0.4, c="#d0d0d0", zorder=0)
    title(a, "(a) Observability\nunder delegation")
    despine(a)

    # (b) metadata validation on the CPU path, horizontal
    classes = [r[0] for r in P.REJECT_CPU]
    yb = np.arange(len(classes))[::-1]; w = 0.36
    for k, (model, colour, col_i) in enumerate((("SSD", P.C1, 1), ("EffDet", P.C2, 2))):
        vals = [100 * r[col_i] / P.TRIALS_PER_CLASS for r in P.REJECT_CPU]
        pos = yb + (0.5 - k) * w
        b.barh(pos, vals, w * 0.9, color=colour, zorder=3)
        for yp, v in zip(pos, vals):
            b.text(v + 3, yp, f"{v:.0f}", va="center", fontsize=NUM, color=LAB)
    b.set_yticks(yb); b.set_yticklabels(classes)
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 135); b.set_xticks([0, 50, 100])
    b.set_xlabel("Rejected (%)", labelpad=1)
    b.grid(True, axis="x", ls=":", lw=0.4, c="#d0d0d0", zorder=0)
    title(b, "(b) Metadata validation,\nCPU path")
    despine(b)

    handles = [Patch(facecolor=P.C1, label="float32"),
               Patch(facecolor=P.C2, label="quantized"),
               Patch(facecolor=P.C1, label="SSD"),
               Patch(facecolor=P.C2, label="EffDet")]
    shared_legend(fig, handles, 4, 0.11)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.28, wspace=0.55)
    save(fig, outdir, "fig_delegation")


FIGURES = {"overhead": fig_overhead_detection,
           "reachability": fig_reachability,
           "delegation": fig_delegation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", choices=sorted(FIGURES))
    args = ap.parse_args()
    style()
    for n in ([args.only] if args.only else FIGURES):
        FIGURES[n](args.outdir)


if __name__ == "__main__":
    main()