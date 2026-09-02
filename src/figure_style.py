from __future__ import annotations

import matplotlib.pyplot as plt

COLORS = {
    "charcoal": "#2D2D2D",
    "gray": "#6B6E73",
    "lightgray": "#D7D9DC",
    "olive": "#6D7F3F",
    "cobalt": "#2E59A8",
    "wine": "#7D3F67",
    "ochre": "#B58A3A",
    "teal": "#2B6F73",
    "terracotta": "#B55D42",
    "offwhite": "#FAFAF8",
}

SCALE_COLORS = {
    1.0: COLORS["teal"],
    2.0: COLORS["cobalt"],
    4.0: COLORS["ochre"],
    8.0: COLORS["terracotta"],
}


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["gray"])
    ax.spines["bottom"].set_color(COLORS["gray"])
    ax.tick_params(colors=COLORS["charcoal"], width=0.6, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["lightgray"], linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", ha="left", va="top", color=COLORS["charcoal"])
