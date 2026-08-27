from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.figure_style import COLORS, apply_publication_style, clean_axes, panel_label

apply_publication_style()


def _find_csv(source: Path, suffix: str) -> pd.DataFrame:
    """Read a Phase-VI CSV from either the output ZIP or an extracted directory."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            matches = [n for n in zf.namelist() if n.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one *{suffix} in {source}; found {matches}")
            return pd.read_csv(io.BytesIO(zf.read(matches[0])))
    if source.is_dir():
        matches = list(source.rglob(suffix))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {suffix} below {source}; found {matches}")
        return pd.read_csv(matches[0])
    raise FileNotFoundError(source)


def load_phase6(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = _find_csv(source, "phase6_head_to_head_results.csv")
    deltas = _find_csv(source, "phase6_head_to_head_deltas.csv")
    required_r = {"dataset", "experiment", "macro_f1_oof", "macro_f1_ci_low", "macro_f1_ci_high"}
    required_d = {"dataset", "contrast", "delta_macro_f1", "delta_ci_low", "delta_ci_high"}
    if not required_r.issubset(results):
        raise RuntimeError(f"Results CSV missing columns: {sorted(required_r - set(results.columns))}")
    if not required_d.issubset(deltas):
        raise RuntimeError(f"Deltas CSV missing columns: {sorted(required_d - set(deltas.columns))}")
    return results, deltas


def draw_concept_panel(ax):
    ax.set_axis_off()
    panel_label(ax, "A")

    ax.text(0.02, 0.96, "Complementary descriptions of local image structure",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.4,
            fontweight="bold", color=COLORS["charcoal"])

    boxes = [
        (0.03, 0.36, 0.25, 0.36, COLORS["olive"],
         "Ordinal patterns\n(OP75)",
         "2×2 rank-order relations\nwith explicit intensity ties"),
        (0.36, 0.36, 0.27, 0.36, COLORS["cobalt"],
         "Level-set geometry\n(K40)",
         "multiscale curvature of\niso-luminance contours"),
        (0.72, 0.36, 0.25, 0.36, COLORS["wine"],
         "Joint representation",
         "OP75 + K40"),
    ]
    for x, y, w, h, c, title, body in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                               transform=ax.transAxes, linewidth=1.1, edgecolor=c,
                               facecolor="white")
        ax.add_patch(patch)
        ax.text(x + w/2, y + 0.245, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.6, fontweight="bold", color=c, linespacing=1.05)
        ax.text(x + w/2, y + 0.11, body, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.4, color=COLORS["charcoal"], linespacing=1.2)

    ax.text(0.315, 0.54, "+", transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color=COLORS["gray"])
    ax.add_patch(FancyArrowPatch((0.64, 0.54), (0.71, 0.54), transform=ax.transAxes,
                                arrowstyle="-|>", mutation_scale=10,
                                linewidth=0.9, color=COLORS["gray"]))

    ax.text(0.50, 0.16,
            "Question: does continuous multiscale geometry add information beyond local ordinal structure?",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.0,
            color=COLORS["charcoal"])


def plot_family_comparison(ax, results: pd.DataFrame):
    panel_label(ax, "B")
    clean_axes(ax)
    order = ["OP_HC", "OP11", "OP24", "OP75", "K40_curvature", "OP75_K40"]
    labels = ["H,C", "OP11", "OP24", "OP75", "K40", "OP75+K40"]
    x = np.arange(len(order))
    offsets = {"artbench10_all": -0.11, "artbench10_wikiart8": 0.11}
    ds_label = {"artbench10_all": "ArtBench-10", "artbench10_wikiart8": "WikiArt-8"}
    ds_marker = {"artbench10_all": "o", "artbench10_wikiart8": "s"}
    ds_color = {"artbench10_all": COLORS["charcoal"], "artbench10_wikiart8": COLORS["gray"]}

    for ds in offsets:
        sub = results[results["dataset"].eq(ds)].set_index("experiment")
        yy, lo, hi = [], [], []
        for exp in order:
            r = sub.loc[exp]
            yy.append(float(r["macro_f1_oof"]))
            lo.append(float(r["macro_f1_ci_low"]))
            hi.append(float(r["macro_f1_ci_high"]))
        yy = np.asarray(yy)
        err = np.vstack([yy - np.asarray(lo), np.asarray(hi) - yy])
        ax.errorbar(x + offsets[ds], yy, yerr=err, fmt=ds_marker[ds], ms=4.4,
                    lw=0.9, capsize=2.2, color=ds_color[ds], ecolor=ds_color[ds],
                    label=ds_label[ds], zorder=3)

    ax.axvspan(2.55, 3.45, color=COLORS["olive"], alpha=0.055, linewidth=0)
    ax.axvspan(3.55, 4.45, color=COLORS["cobalt"], alpha=0.055, linewidth=0)
    ax.axvspan(4.55, 5.45, color=COLORS["wine"], alpha=0.055, linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Artist-disjoint macro-F1")
    ax.set_ylim(0.06, 0.34)
    ax.set_title("Predictive information across ordinal and geometric representations", loc="left", pad=7)
    ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.5, columnspacing=1.1)
    ax.text(0.98, 0.05, "95% artist-group bootstrap CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.9, color=COLORS["gray"])


def plot_delta_forest(ax, deltas: pd.DataFrame):
    panel_label(ax, "C")
    clean_axes(ax)
    wanted = [
        ("artbench10_all", "curvature_increment_over_ordinal_full", "ArtBench-10  OP75 + K40 − OP75"),
        ("artbench10_all", "curvature_increment_over_ordinal_k40", "ArtBench-10  matched k=40"),
        ("artbench10_wikiart8", "curvature_increment_over_ordinal_full", "WikiArt-8  OP75 + K40 − OP75"),
        ("artbench10_wikiart8", "curvature_increment_over_ordinal_k40", "WikiArt-8  matched k=40"),
        ("artbench10_wikiart8", "curvature_increment_over_B_plus_ordinal_full", "WikiArt-8  B90+OP75+K40 − B90+OP75"),
        ("artbench10_wikiart8", "curvature_increment_over_B_plus_ordinal_k90", "WikiArt-8  matched k=90"),
    ]
    rows = []
    for ds, contrast, label in wanted:
        hit = deltas[(deltas["dataset"] == ds) & (deltas["contrast"] == contrast)]
        if len(hit) != 1:
            raise RuntimeError(f"Missing/duplicate contrast: {ds} {contrast}")
        r = hit.iloc[0]
        rows.append((label, float(r.delta_macro_f1), float(r.delta_ci_low), float(r.delta_ci_high),
                     float(r.get("bootstrap_q_bh_primary", np.nan))))

    y = np.arange(len(rows))[::-1]
    for yi, (label, d, lo, hi, q) in zip(y, rows):
        if "matched" in label:
            color = COLORS["gray"]
            marker = "s"
        elif "B90" in label:
            color = COLORS["wine"]
            marker = "D"
        else:
            color = COLORS["cobalt"]
            marker = "o"
        ax.hlines(yi, lo, hi, color=color, lw=1.4, zorder=2)
        ax.plot(d, yi, marker=marker, ms=5.2, color=color, zorder=3)
        qtxt = "" if not np.isfinite(q) else f"q={q:.3g}"
        ax.text(max(0.082, hi + 0.003), yi, qtxt, va="center", ha="left", fontsize=6.8,
                color=COLORS["gray"])

    ax.axvline(0, color=COLORS["charcoal"], lw=0.75, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.0)
    ax.set_xlabel("Paired Δ macro-F1")
    ax.set_xlim(-0.012, 0.10)
    ax.set_ylim(-0.7, len(rows)-0.3)
    ax.set_title("Pre-specified increments from level-set geometry", loc="left", pad=7)
    ax.grid(axis="x", color=COLORS["lightgray"], linewidth=0.45, alpha=0.55)
    ax.grid(axis="y", visible=False)


def plot_strict_control(ax, results: pd.DataFrame, deltas: pd.DataFrame):
    panel_label(ax, "D")
    clean_axes(ax)
    sub = results[results["dataset"].eq("artbench10_wikiart8")].set_index("experiment")

    groups = [
        ("Full representation", ["B90_strong", "B90_OP75", "B90_OP75_K40"],
         ["B90", "B90+OP75", "B90+OP75+K40"]),
        ("Dimension-matched", ["B90_strong", "B90_OP75_k90", "B90_OP75_K40_k90"],
         ["B90", "(B90+OP75) k=90", "(B90+OP75+K40) k=90"]),
    ]
    xbase = [0, 1]
    for gi, (gname, exps, labs) in enumerate(groups):
        xs = np.asarray([xbase[gi] - 0.18, xbase[gi], xbase[gi] + 0.18])
        cols = [COLORS["charcoal"], COLORS["olive"], COLORS["wine"]]
        for x, exp, lab, c in zip(xs, exps, labs, cols):
            r = sub.loc[exp]
            y = float(r.macro_f1_oof)
            lo = float(r.macro_f1_ci_low)
            hi = float(r.macro_f1_ci_high)
            ax.errorbar([x], [y], yerr=[[y-lo], [hi-y]], fmt="o", ms=5.0, lw=1.0,
                        capsize=2.2, color=c, ecolor=c, zorder=3)
        ax.plot(xs, [float(sub.loc[e].macro_f1_oof) for e in exps], color=COLORS["lightgray"], lw=0.8, zorder=1)

    for contrast, x, ytxt in [
        ("curvature_increment_over_B_plus_ordinal_full", 0.09, 0.319),
        ("curvature_increment_over_B_plus_ordinal_k90", 1.09, 0.306),
    ]:
        r = deltas[(deltas.dataset == "artbench10_wikiart8") & (deltas.contrast == contrast)].iloc[0]
        ax.text(x, ytxt, f"Δ={r.delta_macro_f1:+.3f}\n95% CI [{r.delta_ci_low:.3f}, {r.delta_ci_high:.3f}]",
                ha="center", va="top", fontsize=6.9, color=COLORS["charcoal"])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full representation", "Dimension-matched\n(k=90)"])
    ax.set_ylabel("WikiArt-8 macro-F1")
    ax.set_ylim(0.215, 0.325)
    ax.set_xlim(-0.45, 1.45)
    ax.set_title("Geometry after strong appearance + ordinal structure", loc="left", pad=7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["charcoal"], markeredgecolor=COLORS["charcoal"], label="B90"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["olive"], markeredgecolor=COLORS["olive"], label="+ OP75"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["wine"], markeredgecolor=COLORS["wine"], label="+ K40"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", ncol=3, handletextpad=0.4, columnspacing=0.9)


def make_figure(source: Path, out_dir: Path):
    results, deltas = load_phase6(source)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.6, 8.1), facecolor="white")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.88, 1.12], width_ratios=[1, 1],
                           hspace=0.38, wspace=0.36)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    draw_concept_panel(axA)
    plot_family_comparison(axB, results)
    plot_delta_forest(axC, deltas)
    plot_strict_control(axD, results, deltas)

    fig.suptitle("Ordinal patterns and multiscale level-set geometry encode complementary style information",
                 x=0.05, y=0.987, ha="left", va="top", fontsize=12.1, fontweight="bold",
                 color=COLORS["charcoal"])
    fig.text(0.05, 0.956,
             "Same ArtBench pilot, same artist-disjoint nested cross-validation; uncertainty from artist-group bootstrap",
             ha="left", va="top", fontsize=7.6, color=COLORS["gray"])

    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.08, top=0.91)

    stems = [
        (out_dir / "Figure5_ordinal_vs_geometry.pdf", None),
        (out_dir / "Figure5_ordinal_vs_geometry.svg", None),
        (out_dir / "Figure5_ordinal_vs_geometry.png", 450),
    ]
    for path, dpi in stems:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 5. Ordinal patterns and multiscale level-set geometry encode complementary style information. "
        "(A) Conceptual distinction between local tie-aware 2×2 ordinal structure (OP75) and continuous multiscale "
        "level-set curvature (K40). (B) Artist-disjoint style-recognition performance for compact ordinal summaries, "
        "the full 75-pattern representation, curvature, and their combination in ArtBench-10 and WikiArt-8. Points "
        "show out-of-fold macro-F1 and intervals show 95% artist-group bootstrap confidence intervals. (C) Paired "
        "bootstrap increments attributable to curvature, including dimension-matched controls and the stricter test "
        "after strong appearance descriptors plus OP75 are already included. (D) WikiArt-8 control showing the "
        "increment from curvature after conventional appearance and ordinal descriptors. Full-representation gains are "
        "supported, whereas the stricter k=90 dimension-matched increment is positive but its 95% interval includes zero."
    )
    (out_dir / "Figure5_caption.txt").write_text(caption, encoding="utf-8")

    key = results[results.experiment.isin(["OP75", "K40_curvature", "OP75_K40", "B90_strong", "B90_OP75", "B90_OP75_K40"])].copy()
    key.to_csv(out_dir / "Figure5_key_values.csv", index=False)
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Make manuscript Figure 5: ordinal patterns vs multiscale level-set geometry.")
    p.add_argument("--phase6", type=Path, required=True, help="Phase-VI output ZIP or extracted directory")
    p.add_argument("--out-dir", type=Path, default=Path("figures/main"))
    args = p.parse_args()
    make_figure(args.phase6, args.out_dir)


if __name__ == "__main__":
    main()
