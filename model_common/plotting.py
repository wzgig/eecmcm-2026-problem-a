from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


CSEE_COLORS = {
    "blue": "#1f4e79",
    "red": "#a61c2a",
    "green": "#2f6b4f",
    "orange": "#c55a11",
    "purple": "#5b4b8a",
    "gray": "#666666",
    "light_blue": "#9dc3e6",
    "light_red": "#f4b6b2",
}


def set_csee_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["SimSun", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.2,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output_base: Path, formats: Iterable[str] = ("png", "pdf", "svg")) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = 600
        fig.savefig(output_base.with_suffix(f".{fmt}"), **kwargs)


def plot_q1_power_balance(df: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    x = df["hour"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, df["total_load_mw"], color=CSEE_COLORS["blue"], marker="o", label="总用电负荷")
    ax.plot(x, df["renewable_mw"], color=CSEE_COLORS["green"], marker="s", label="风光发电合计")
    ax.bar(x - 0.18, df["grid_purchase_mw"], width=0.34, color=CSEE_COLORS["light_red"], edgecolor=CSEE_COLORS["red"], linewidth=0.5, label="购电功率")
    ax.bar(x + 0.18, df["grid_sale_mw"], width=0.34, color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.5, label="上网功率")
    ax.set_xlabel("时段/h")
    ax.set_ylabel("功率/MW")
    ax.set_title("典型日功率平衡与购售电功率")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.8)
    save_figure(fig, output_dir / "q1_fig1_power_balance_csee")
    plt.close(fig)


def plot_q1_components(df: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    x = df["hour"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    ax1.plot(x, df["regular_load_mw"], color=CSEE_COLORS["gray"], marker="o", label="常规电负荷")
    ax1.plot(x, df["process_load_mw"], color=CSEE_COLORS["red"], marker="s", label="制氢制氨负荷")
    ax1.plot(x, df["total_load_mw"], color=CSEE_COLORS["blue"], marker="^", label="总负荷")
    ax1.set_ylabel("负荷功率/MW")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax1.legend(ncol=3, frameon=False, loc="upper left")

    ax2.stackplot(
        x,
        df["wind_mw"],
        df["pv_mw"],
        colors=[CSEE_COLORS["light_blue"], "#f7c97f"],
        labels=["风电", "光伏"],
        alpha=0.85,
    )
    ax2.plot(x, df["renewable_mw"], color=CSEE_COLORS["green"], marker="o", label="风光合计")
    ax2.set_xlabel("时段/h")
    ax2.set_ylabel("发电功率/MW")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.2, 23.2)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax2.legend(ncol=3, frameon=False, loc="upper left")
    fig.suptitle("典型日负荷与新能源出力分项曲线", y=0.995, fontsize=10)
    fig.tight_layout(pad=0.8)
    save_figure(fig, output_dir / "q1_fig2_components_csee")
    plt.close(fig)


def plot_q1_indicator_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    metrics = [
        ("新能源自发自用比例（物理/政策口径）", 0.60, "≥60%"),
        ("总用电量绿电比例", 0.30, "≥30%"),
        ("新能源上网电量比例", 0.20, "≤20%"),
    ]
    values = [float(summary.loc[summary["metric"] == m, "value"].iloc[0]) for m, _, _ in metrics]
    thresholds = [thr for _, thr, _ in metrics]
    labels = ["自发自用比例", "绿电比例", "上网比例"]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = [CSEE_COLORS["green"], CSEE_COLORS["blue"], CSEE_COLORS["red"]]
    bars = ax.bar(labels, [v * 100 for v in values], color=colors, alpha=0.82, width=0.56)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{value:.2%}", ha="center", va="bottom", fontsize=8)
    for i, (_, thr, text) in enumerate(metrics):
        ax.hlines(thr * 100, i - 0.35, i + 0.35, colors="black", linestyles="--", linewidth=0.9)
        ax.text(i + 0.38, thr * 100, text, va="center", fontsize=8)
    ax.set_ylabel("指标值/%")
    ax.set_ylim(0, max(80, max(values) * 120))
    ax.set_title("绿电直连指标合格性判定")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(pad=0.8)
    save_figure(fig, output_dir / "q1_fig3_indicator_check_csee")
    plt.close(fig)


def plot_q1_cost_breakdown(summary: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    cost_metrics = [
        "风电成本",
        "光伏成本",
        "购电成本",
        "碱性电解槽运维成本",
        "PEM电解槽运维成本",
        "合成氨装置运维成本",
        "余电上网收益",
    ]
    data = summary[summary["metric"].isin(cost_metrics)].copy()
    data["plot_value"] = data["value"]
    data.loc[data["metric"] == "余电上网收益", "plot_value"] *= -1
    labels = ["风电", "光伏", "购电", "ALK运维", "PEM运维", "合成氨运维", "售电收益"]
    colors = [
        CSEE_COLORS["light_blue"],
        "#f7c97f",
        CSEE_COLORS["light_red"],
        CSEE_COLORS["purple"],
        CSEE_COLORS["blue"],
        CSEE_COLORS["gray"],
        CSEE_COLORS["green"],
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, data["plot_value"] / 10000, color=colors, edgecolor="black", linewidth=0.45, width=0.62)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, data["plot_value"] / 10000):
        va = "bottom" if value >= 0 else "top"
        offset = 0.25 if value >= 0 else -0.25
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=8)
    ax.set_ylabel("金额/万元")
    ax.set_title("典型日成本与收益分项")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(pad=0.8)
    save_figure(fig, output_dir / "q1_fig4_cost_breakdown_csee")
    plt.close(fig)
