from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
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


def legend_below(ax: plt.Axes, ncol: int, y: float = -0.22) -> None:
    ax.legend(
        frameon=False,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0,
        handlelength=2.4,
        columnspacing=1.2,
    )


def figure_legend_below(fig: plt.Figure, handles: list, labels: list, ncol: int, y: float = 0.01) -> None:
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0,
        handlelength=2.4,
        columnspacing=1.2,
    )


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", dy_frac: float = 0.015) -> None:
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * dy_frac
    for bar in bars:
        value = bar.get_height()
        if value >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + dy, fmt.format(value), ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value - dy, fmt.format(value), ha="center", va="top", fontsize=7.5)


def plot_q1_power_balance(df: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    x = df["hour"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
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
    legend_below(ax, ncol=4, y=-0.18)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q1_fig1_power_balance_csee")
    plt.close(fig)


def plot_q1_components(df: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    x = df["hour"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    ax1.plot(x, df["regular_load_mw"], color=CSEE_COLORS["gray"], marker="o", label="常规电负荷")
    ax1.plot(x, df["process_load_mw"], color=CSEE_COLORS["red"], marker="s", label="制氢制氨负荷")
    ax1.plot(x, df["total_load_mw"], color=CSEE_COLORS["blue"], marker="^", label="总负荷")
    ax1.set_ylabel("负荷功率/MW")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
    legend_below(ax1, ncol=3, y=-0.09)

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
    legend_below(ax2, ncol=3, y=-0.22)
    fig.suptitle("典型日负荷与新能源出力分项曲线", y=0.98, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.18, hspace=0.36)
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

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = [CSEE_COLORS["green"], CSEE_COLORS["blue"], CSEE_COLORS["red"]]
    bars = ax.bar(labels, [v * 100 for v in values], color=colors, alpha=0.82, width=0.56)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{value:.2%}", ha="center", va="bottom", fontsize=8)
    for i, (_, thr, text) in enumerate(metrics):
        ax.hlines(thr * 100, i - 0.35, i + 0.35, colors="black", linestyles="--", linewidth=0.9)
        ax.text(i + 0.38, thr * 100, text, va="center", fontsize=8)
    ax.set_ylabel("指标值/%")
    ax.set_ylim(0, max(82, max(values) * 125))
    ax.set_title("绿电直连指标合格性判定")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.15)
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

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(labels, data["plot_value"] / 10000, color=colors, edgecolor="black", linewidth=0.45, width=0.62)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, data["plot_value"] / 10000):
        va = "bottom" if value >= 0 else "top"
        offset = 0.25 if value >= 0 else -0.25
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=8)
    ax.set_ylabel("金额/万元")
    ax.set_title("典型日成本与收益分项")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16)
    save_figure(fig, output_dir / "q1_fig4_cost_breakdown_csee")
    plt.close(fig)


def plot_q1_net_exchange(df: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    x = df["hour"]
    net_surplus = df["renewable_mw"] - df["total_load_mw"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(
        x,
        net_surplus.clip(lower=0),
        color=CSEE_COLORS["light_blue"],
        edgecolor=CSEE_COLORS["blue"],
        linewidth=0.5,
        label="风光富余",
    )
    ax.bar(
        x,
        net_surplus.clip(upper=0),
        color=CSEE_COLORS["light_red"],
        edgecolor=CSEE_COLORS["red"],
        linewidth=0.5,
        label="用电缺口",
    )
    ax.plot(x, net_surplus, color=CSEE_COLORS["gray"], marker="o", linewidth=1.2, label="源荷差额")
    ax.set_xlabel("时段/h")
    ax.set_ylabel("源荷差额/MW")
    ax.set_title("典型日源荷差额与购售电方向")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    legend_below(ax, ncol=3, y=-0.22)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q1_fig5_net_exchange_csee")
    plt.close(fig)


def plot_q3_typical_load_factor(typical_hourly: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    renewable = typical_hourly.drop_duplicates("hour").sort_values("hour")
    heat = typical_hourly.pivot_table(index="production_t_per_day", columns="hour", values="production_load_factor", aggfunc="first")
    heat = heat.sort_index(ascending=False) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.3]})
    ax1.stackplot(
        renewable["hour"],
        renewable["wind_mw"],
        renewable["pv_mw"],
        colors=[CSEE_COLORS["light_blue"], "#f7c97f"],
        labels=["风电", "光伏"],
        alpha=0.82,
    )
    ax1.plot(
        renewable["hour"],
        renewable["renewable_mw"],
        color=CSEE_COLORS["green"],
        marker="o",
        linewidth=1.2,
        label="风光合计",
    )
    ax1.set_ylabel("风光出力/MW")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    legend_below(ax1, ncol=3, y=-0.08)

    im = ax2.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax2.set_yticks(np.arange(len(heat.index)))
    ax2.set_yticklabels([f"{int(v)}" for v in heat.index])
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlabel("时段/h")
    ax2.set_ylabel("日产量/t")
    ax2.set_title("连续制氨负荷率/%", fontsize=9, pad=4)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.values[i, j]
            if value <= 15 or value >= 95:
                text_color = "white" if value >= 60 else "#222222"
                ax2.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color=text_color)
    cbar = fig.colorbar(im, ax=ax2, orientation="vertical", fraction=0.024, pad=0.014)
    cbar.set_label("负荷率/%")
    fig.suptitle("典型风光场景下风光出力与连续制氨负荷率", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.90, bottom=0.12, hspace=0.38)
    save_figure(fig, output_dir / "q3_fig1_typical_load_factor_csee")
    plt.close(fig)


def plot_q3_typical_load_factor_lines(typical_hourly: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    for production, group in typical_hourly.groupby("production_t_per_day", sort=False):
        group = group.sort_values("hour")
        ax1.plot(group["hour"], group["production_load_factor"] * 100, marker="o", linewidth=1.4, label=f"{int(production)} t/d")
    ax1.set_xlabel("时段/h")
    ax1.set_ylabel("连续负荷率/%")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(-0.4, 23.4)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    renewable = typical_hourly.drop_duplicates("hour").sort_values("hour")
    ax2 = ax1.twinx()
    ax2.fill_between(renewable["hour"], renewable["renewable_mw"], color=CSEE_COLORS["light_blue"], alpha=0.25, label="风光合计")
    ax2.plot(renewable["hour"], renewable["renewable_mw"], color=CSEE_COLORS["green"], linewidth=1.2)
    ax2.set_ylabel("风光出力/MW")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    figure_legend_below(fig, handles1 + handles2[:1], labels1 + labels2[:1], ncol=3, y=0.01)
    ax1.set_title("典型风光场景下连续制氨负荷率曲线")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q3_fig1b_typical_load_factor_lines_csee")
    plt.close(fig)


def plot_q3_q2_cost_compare(compare: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    df = compare.sort_values("production_t_per_day")
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bars1 = ax.bar(
        x - width / 2,
        df["q2_annual_average_ton_cost_yuan_per_t"],
        width=width,
        color=CSEE_COLORS["light_blue"],
        edgecolor=CSEE_COLORS["blue"],
        linewidth=0.6,
        label="问题二：离散开停",
    )
    bars2 = ax.bar(
        x + width / 2,
        df["q3_annual_average_ton_cost_yuan_per_t"],
        width=width,
        color="#f7c97f",
        edgecolor=CSEE_COLORS["orange"],
        linewidth=0.6,
        label="问题三：连续调节",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in df["production_t_per_day"]])
    ax.set_xlabel("日产量/t")
    ax.set_ylabel("年均吨氨成本/(元/t)")
    ax.set_title("离散开停与连续调节的年均吨氨成本对比")
    ax.set_ylim(0, max(df["q2_annual_average_ton_cost_yuan_per_t"].max(), df["q3_annual_average_ton_cost_yuan_per_t"].max()) * 1.16)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    annotate_bars(ax, bars1, fmt="{:.0f}", dy_frac=0.01)
    annotate_bars(ax, bars2, fmt="{:.0f}", dy_frac=0.01)
    legend_below(ax, ncol=2, y=-0.20)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.23)
    save_figure(fig, output_dir / "q3_fig2_compare_q2_cost_csee")
    plt.close(fig)


def plot_q3_grid_indicator_compare(compare: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    df = compare.sort_values("production_t_per_day")
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)

    width = 0.18
    ax1.bar(x - 1.5 * width, df["q2_annual_purchase_mwh"], width=width, color=CSEE_COLORS["light_red"], edgecolor=CSEE_COLORS["red"], linewidth=0.5, label="离散购电")
    ax1.bar(x - 0.5 * width, df["q3_annual_purchase_mwh"], width=width, color="#f4d2cf", edgecolor=CSEE_COLORS["red"], linewidth=0.5, hatch="//", label="连续购电")
    ax1.bar(x + 0.5 * width, df["q2_annual_sale_mwh"], width=width, color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.5, label="离散上网")
    ax1.bar(x + 1.5 * width, df["q3_annual_sale_mwh"], width=width, color="#c9dff2", edgecolor=CSEE_COLORS["blue"], linewidth=0.5, hatch="\\\\", label="连续上网")
    ax1.set_ylabel("年度电量/MWh")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    legend_below(ax1, ncol=4, y=-0.10)

    ax2.plot(x, df["q2_annual_self_use_ratio"] * 100, marker="o", color=CSEE_COLORS["green"], linestyle="--", label="离散自发自用")
    ax2.plot(x, df["q3_annual_self_use_ratio"] * 100, marker="o", color=CSEE_COLORS["green"], label="连续自发自用")
    ax2.plot(x, df["q2_annual_sale_ratio"] * 100, marker="^", color=CSEE_COLORS["red"], linestyle="--", label="离散上网比例")
    ax2.plot(x, df["q3_annual_sale_ratio"] * 100, marker="^", color=CSEE_COLORS["red"], label="连续上网比例")
    ax2.axhline(60, color=CSEE_COLORS["green"], linestyle=":", linewidth=0.8)
    ax2.axhline(20, color=CSEE_COLORS["red"], linestyle=":", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(v)}" for v in df["production_t_per_day"]])
    ax2.set_xlabel("日产量/t")
    ax2.set_ylabel("指标值/%")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    legend_below(ax2, ncol=2, y=-0.24)

    fig.suptitle("离散开停与连续调节的购售电和绿电指标对比", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.20, hspace=0.34)
    save_figure(fig, output_dir / "q3_fig3_compare_q2_grid_indicator_csee")
    plt.close(fig)


def plot_q3_scenario_cost_box(scenario_summary: pd.DataFrame, output_dir: Path, production_levels: Iterable[int]) -> None:
    set_csee_style()
    ordered = sorted(production_levels)
    data = [scenario_summary.loc[scenario_summary["production_t_per_day"] == d, "吨氨成本"] for d in ordered]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bp = ax.boxplot(data, tick_labels=[str(d) for d in ordered], patch_artist=True, widths=0.58)
    for patch in bp["boxes"]:
        patch.set_facecolor("#f7c97f")
        patch.set_edgecolor(CSEE_COLORS["orange"])
    ax.set_xlabel("日产量/t")
    ax.set_ylabel("吨氨成本/(元/t)")
    ax.set_title("连续调节下24种风光场景吨氨成本分布")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.15)
    save_figure(fig, output_dir / "q3_fig4_scenario_cost_box_csee")
    plt.close(fig)


def plot_q3_cost_duration_curve(
    scenario_summary: pd.DataFrame,
    output_dir: Path,
    production_levels: Iterable[int],
    scenario_days: int = 15,
) -> None:
    set_csee_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for production in sorted(production_levels, reverse=True):
        vals = np.sort(scenario_summary.loc[scenario_summary["production_t_per_day"] == production, "吨氨成本"].to_numpy())
        x = np.arange(1, len(vals) + 1) * scenario_days
        ax.step(x, vals, where="post", linewidth=1.4, label=f"{production} t/d")
    ax.set_xlabel("累计代表天数/d")
    ax.set_ylabel("吨氨成本/(元/t)")
    ax.set_title("连续调节下全年代表场景吨氨成本分布曲线")
    ax.set_xlim(0, max(360, len(vals) * scenario_days))
    ax.grid(True, linestyle="--", alpha=0.4)
    legend_below(ax, ncol=5, y=-0.20)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q3_fig7_annual_cost_duration_csee")
    plt.close(fig)


def plot_q3_cost_delta_heatmaps(q2_summary: pd.DataFrame, q3_summary: pd.DataFrame, output_dir: Path, production_levels: Iterable[int]) -> None:
    set_csee_style()
    key = ["wind_scenario", "pv_scenario", "production_t_per_day"]
    merged = q2_summary[key + ["吨氨成本"]].merge(
        q3_summary[key + ["吨氨成本"]],
        on=key,
        suffixes=("_q2", "_q3"),
    )
    merged["cost_delta"] = merged["吨氨成本_q3"] - merged["吨氨成本_q2"]
    vmax = float(np.ceil(max(abs(merged["cost_delta"].min()), abs(merged["cost_delta"].max())) / 50) * 50)
    levels = sorted(production_levels, reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.4), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    image = None
    for ax, production in zip(axes_flat, levels):
        matrix = (
            merged.loc[merged["production_t_per_day"] == production]
            .pivot_table(index="wind_scenario", columns="pv_scenario", values="cost_delta", aggfunc="first")
            .sort_index()
        )
        image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{production} t/d", fontsize=9)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([str(int(c)) for c in matrix.columns])
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([str(int(i)) for i in matrix.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.values[i, j]
                text_color = "white" if abs(value) > 0.55 * vmax else "#222222"
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color=text_color)
    axes_flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("风电场景")
    for ax in axes[-1, :2]:
        ax.set_xlabel("光伏场景")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_flat[:-1], orientation="vertical", fraction=0.026, pad=0.018)
        cbar.set_label("Q3-Q2成本变化/(元/t)")
    fig.suptitle("连续调节相对离散开停的场景吨氨成本变化", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.10, wspace=0.14, hspace=0.26)
    save_figure(fig, output_dir / "q3_fig8_cost_delta_heatmap_csee")
    plt.close(fig)


def plot_q3_satisfaction_stacked(annual: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    df = annual.sort_values("production_t_per_day")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x, df["full_satisfied_days"], color=CSEE_COLORS["green"], label="全满足")
    ax.bar(x, df["partial_satisfied_days"], bottom=df["full_satisfied_days"], color="#f7c97f", label="部分满足")
    bottom = df["full_satisfied_days"] + df["partial_satisfied_days"]
    ax.bar(x, df["none_satisfied_days"], bottom=bottom, color=CSEE_COLORS["light_red"], label="全不满足")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in df["production_t_per_day"]])
    ax.set_xlabel("日产量/t")
    ax.set_ylabel("年度代表天数/d")
    ax.set_title("连续调节下年度绿电指标合格类型统计")
    legend_below(ax, ncol=3, y=-0.20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q3_fig5_satisfaction_stacked_csee")
    plt.close(fig)


def plot_q3_typical_power_balance(selected_hourly: pd.DataFrame, output_dir: Path, production_t_per_day: float) -> None:
    set_csee_style()
    df = selected_hourly.sort_values("hour")
    x = df["hour"]
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.stackplot(
        x,
        df["wind_mw"],
        df["pv_mw"],
        colors=[CSEE_COLORS["light_blue"], "#f7c97f"],
        labels=["风电", "光伏"],
        alpha=0.82,
    )
    ax1.plot(x, df["total_load_mw"], color=CSEE_COLORS["red"], marker="o", label="总负荷")
    ax1.bar(x - 0.18, df["grid_purchase_mw"], width=0.34, color=CSEE_COLORS["light_red"], edgecolor=CSEE_COLORS["red"], linewidth=0.5, label="购电")
    ax1.bar(x + 0.18, df["grid_sale_mw"], width=0.34, color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.5, label="上网")
    ax1.set_xlabel("时段/h")
    ax1.set_ylabel("功率/MW")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(-0.5, 23.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, df["production_load_factor"] * 100, color=CSEE_COLORS["purple"], marker="s", linewidth=1.2, label="负荷率")
    ax2.set_ylabel("制氨负荷率/%")
    ax2.set_ylim(0, 105)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0,
        columnspacing=1.1,
    )
    ax1.set_title(f"典型场景连续调节功率平衡（{production_t_per_day:.0f} t/d）")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q3_fig6_typical_power_balance_csee")
    plt.close(fig)
