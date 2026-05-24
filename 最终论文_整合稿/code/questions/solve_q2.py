from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_common.data import SystemParams, load_typical_day, load_wind_solar_scenarios
from model_common.metrics import get_metric, q1_summary
from model_common.plotting import CSEE_COLORS, annotate_bars, legend_below, save_figure, set_csee_style


OUT_DIR = Path(__file__).resolve().parent / "outputs"
PRODUCTION_LEVELS_T = [72, 63, 54, 45, 36]
SCENARIO_DAYS = 15
CAPACITY_DAILY_AMMONIA_T = 72.0


def add_regular_load_to_scenarios(params: SystemParams) -> pd.DataFrame:
    scenario_df = load_wind_solar_scenarios(params=params, root=PROJECT_ROOT)
    regular = load_typical_day(params=params, root=PROJECT_ROOT)[["hour", "regular_load_mw"]]
    return scenario_df.merge(regular, on="hour", how="left")


def net_grid_cost_yuan(load_mw: pd.Series, renewable_mw: pd.Series, buy_price: pd.Series, sell_price: float) -> pd.Series:
    deficit = (load_mw - renewable_mw).clip(lower=0)
    surplus = (renewable_mw - load_mw).clip(lower=0)
    return deficit * 1000 * buy_price - surplus * 1000 * sell_price


def optimize_discrete_schedule(
    base_df: pd.DataFrame,
    params: SystemParams,
    production_t_per_day: float,
    case_name: str,
    wind_scenario: int | None = None,
    pv_scenario: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    hourly = base_df.copy().sort_values("hour").reset_index(drop=True)
    full_output_t_per_h = params.ammonia_output_t_per_h(CAPACITY_DAILY_AMMONIA_T)
    required_hours = production_t_per_day / full_output_t_per_h
    if abs(required_hours - round(required_hours)) > 1e-9:
        raise ValueError(f"Production {production_t_per_day} t/d cannot be represented by full-load hours.")
    required_hours = int(round(required_hours))

    full_alk_mw = params.alk_power_mw(CAPACITY_DAILY_AMMONIA_T)
    full_pem_mw = params.pem_power_mw(CAPACITY_DAILY_AMMONIA_T)
    full_ammonia_mw = params.ammonia_power_mw(CAPACITY_DAILY_AMMONIA_T)
    full_process_mw = full_alk_mw + full_pem_mw + full_ammonia_mw
    process_om_yuan_per_h = (
        full_alk_mw * 1000 * params.alk_om_yuan_per_kwh
        + full_pem_mw * 1000 * params.pem_om_yuan_per_kwh
        + full_ammonia_mw * 1000 * params.ammonia_om_yuan_per_kwh
    )

    off_load = hourly["regular_load_mw"]
    on_load = hourly["regular_load_mw"] + full_process_mw
    renewable = hourly["renewable_mw"]
    buy_price = hourly["purchase_price_yuan_per_kwh"]

    off_grid_cost = net_grid_cost_yuan(off_load, renewable, buy_price, params.sell_price_yuan_per_kwh)
    on_grid_cost = net_grid_cost_yuan(on_load, renewable, buy_price, params.sell_price_yuan_per_kwh)
    marginal_on_cost = on_grid_cost - off_grid_cost + process_om_yuan_per_h

    order = np.lexsort((hourly["hour"].to_numpy(), marginal_on_cost.to_numpy()))
    on = np.zeros(len(hourly), dtype=int)
    on[order[:required_hours]] = 1

    hourly["production_on"] = on
    hourly["marginal_on_cost_yuan"] = marginal_on_cost
    hourly["alk_load_mw"] = full_alk_mw * hourly["production_on"]
    hourly["pem_load_mw"] = full_pem_mw * hourly["production_on"]
    hourly["ammonia_load_mw"] = full_ammonia_mw * hourly["production_on"]
    hourly["process_load_mw"] = full_process_mw * hourly["production_on"]
    hourly["ammonia_output_t"] = full_output_t_per_h * hourly["production_on"]
    hourly["total_load_mw"] = hourly["regular_load_mw"] + hourly["process_load_mw"]

    net_deficit = hourly["total_load_mw"] - hourly["renewable_mw"]
    hourly["grid_purchase_mw"] = net_deficit.clip(lower=0)
    hourly["grid_sale_mw"] = (-net_deficit).clip(lower=0)
    hourly["grid_exchange_mw"] = hourly["grid_purchase_mw"] - hourly["grid_sale_mw"]
    hourly["case_name"] = case_name
    hourly["production_t_per_day"] = production_t_per_day
    hourly["required_on_hours"] = required_hours
    hourly["equipment_utilization"] = production_t_per_day / CAPACITY_DAILY_AMMONIA_T
    hourly["wind_scenario"] = -1 if wind_scenario is None else wind_scenario
    hourly["pv_scenario"] = -1 if pv_scenario is None else pv_scenario

    summary = q1_summary(hourly, params)
    record = {
        "case_name": case_name,
        "wind_scenario": -1 if wind_scenario is None else wind_scenario,
        "pv_scenario": -1 if pv_scenario is None else pv_scenario,
        "production_t_per_day": production_t_per_day,
        "required_on_hours": required_hours,
        "equipment_utilization": production_t_per_day / CAPACITY_DAILY_AMMONIA_T,
        "on_hours": ",".join(str(int(h)) for h in hourly.loc[hourly["production_on"] == 1, "hour"]),
    }
    for _, row in summary.iterrows():
        record[str(row["metric"])] = float(row["value"])
    record["satisfied_count"] = int(
        record["新能源自发自用比例是否达标"] + record["总用电量绿电比例是否达标"] + record["新能源上网电量比例是否达标"]
    )
    if record["satisfied_count"] == 3:
        record["satisfaction_type"] = "全满足"
    elif record["satisfied_count"] == 0:
        record["satisfaction_type"] = "全不满足"
    else:
        record["satisfaction_type"] = "部分满足"
    return hourly, record


def run_typical_cases(params: SystemParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    typical = load_typical_day(params=params, root=PROJECT_ROOT)
    hourly_parts = []
    records = []
    for production in PRODUCTION_LEVELS_T:
        hourly, record = optimize_discrete_schedule(typical, params, production, "typical")
        hourly_parts.append(hourly)
        records.append(record)
    return pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(records)


def run_scenario_cases(params: SystemParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = add_regular_load_to_scenarios(params)
    hourly_parts = []
    records = []
    for (scenario, wind_id, pv_id), group in scenarios.groupby(["scenario", "wind_scenario", "pv_scenario"], sort=True):
        for production in PRODUCTION_LEVELS_T:
            hourly, record = optimize_discrete_schedule(
                group,
                params,
                production,
                str(scenario),
                wind_scenario=int(wind_id),
                pv_scenario=int(pv_id),
            )
            hourly_parts.append(hourly)
            records.append(record)
    return pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(records)


def build_annual_summary(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for production, group in scenario_summary.groupby("production_t_per_day", sort=False):
        annual_total_cost = float((group["日总成本（扣除售电收益后）"] * SCENARIO_DAYS).sum())
        annual_ammonia = float((group["日产氨量"] * SCENARIO_DAYS).sum())
        annual_purchase = float((group["网购电量"] * SCENARIO_DAYS).sum())
        annual_sale = float((group["上网电量"] * SCENARIO_DAYS).sum())
        type_counts = group["satisfaction_type"].value_counts().to_dict()
        rows.append(
            {
                "production_t_per_day": production,
                "annual_days": int(len(group) * SCENARIO_DAYS),
                "annual_ammonia_t": annual_ammonia,
                "annual_total_cost_yuan": annual_total_cost,
                "annual_average_ton_cost_yuan_per_t": annual_total_cost / annual_ammonia,
                "annual_purchase_mwh": annual_purchase,
                "annual_sale_mwh": annual_sale,
                "full_satisfied_scenarios": int(type_counts.get("全满足", 0)),
                "partial_satisfied_scenarios": int(type_counts.get("部分满足", 0)),
                "none_satisfied_scenarios": int(type_counts.get("全不满足", 0)),
                "full_satisfied_days": int(type_counts.get("全满足", 0) * SCENARIO_DAYS),
                "partial_satisfied_days": int(type_counts.get("部分满足", 0) * SCENARIO_DAYS),
                "none_satisfied_days": int(type_counts.get("全不满足", 0) * SCENARIO_DAYS),
            }
        )
    annual = pd.DataFrame(rows)
    return annual.sort_values("production_t_per_day", ascending=False).reset_index(drop=True)


def build_distribution_summary(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for production, group in scenario_summary.groupby("production_t_per_day", sort=False):
        rows.append(
            {
                "production_t_per_day": production,
                "ton_cost_min": float(group["吨氨成本"].min()),
                "ton_cost_mean": float(group["吨氨成本"].mean()),
                "ton_cost_median": float(group["吨氨成本"].median()),
                "ton_cost_max": float(group["吨氨成本"].max()),
                "purchase_mwh_mean": float(group["网购电量"].mean()),
                "purchase_mwh_min": float(group["网购电量"].min()),
                "purchase_mwh_max": float(group["网购电量"].max()),
                "sale_mwh_mean": float(group["上网电量"].mean()),
                "sale_mwh_min": float(group["上网电量"].min()),
                "sale_mwh_max": float(group["上网电量"].max()),
                "self_use_ratio_mean": float(group["新能源自发自用比例（物理/政策口径）"].mean()),
                "green_power_ratio_mean": float(group["总用电量绿电比例"].mean()),
                "sale_ratio_mean": float(group["新能源上网电量比例"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("production_t_per_day", ascending=False).reset_index(drop=True)


def write_markdown_summary(typical_summary: pd.DataFrame, scenario_summary: pd.DataFrame, annual: pd.DataFrame) -> None:
    best_typical = typical_summary.loc[typical_summary["吨氨成本"].idxmin()]
    best_annual = annual.loc[annual["annual_average_ton_cost_yuan_per_t"].idxmin()]

    lines = [
        "# 问题二结果摘要",
        "",
        "## 典型风光场景",
        "",
        "| 日产量/t | 开机小时数 | 设备利用率 | 吨氨成本/(元/t) | 自发自用比例 | 绿电比例 | 上网比例 | 合格类型 | 开机时段 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in typical_summary.sort_values("production_t_per_day", ascending=False).iterrows():
        lines.append(
            "| {production:.0f} | {hours:.0f} | {util:.2%} | {cost:,.2f} | {self:.2%} | {green:.2%} | {sale:.2%} | {stype} | {on_hours} |".format(
                production=row["production_t_per_day"],
                hours=row["required_on_hours"],
                util=row["equipment_utilization"],
                cost=row["吨氨成本"],
                self=row["新能源自发自用比例（物理/政策口径）"],
                green=row["总用电量绿电比例"],
                sale=row["新能源上网电量比例"],
                stype=row["satisfaction_type"],
                on_hours=row["on_hours"],
            )
        )

    lines.extend(
        [
            "",
            f"典型风光场景下吨氨成本最低的日产量为 **{best_typical['production_t_per_day']:.0f} t/d**，吨氨成本为 **{best_typical['吨氨成本']:,.2f} 元/t**。",
            "",
            "## 24种风光场景年度加权统计",
            "",
            "| 日产量/t | 年产氨量/t | 年总成本/万元 | 年均吨氨成本/(元/t) | 年购电量/MWh | 年上网电量/MWh | 全满足/天 | 部分满足/天 | 全不满足/天 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in annual.iterrows():
        lines.append(
            "| {production:.0f} | {ammonia:,.0f} | {total_cost:,.2f} | {avg_cost:,.2f} | {purchase:,.2f} | {sale:,.2f} | {full_days} | {partial_days} | {none_days} |".format(
                production=row["production_t_per_day"],
                ammonia=row["annual_ammonia_t"],
                total_cost=row["annual_total_cost_yuan"] / 10000,
                avg_cost=row["annual_average_ton_cost_yuan_per_t"],
                purchase=row["annual_purchase_mwh"],
                sale=row["annual_sale_mwh"],
                full_days=int(row["full_satisfied_days"]),
                partial_days=int(row["partial_satisfied_days"]),
                none_days=int(row["none_satisfied_days"]),
            )
        )

    lines.extend(
        [
            "",
            f"24场景年度加权口径下，年均吨氨成本最低的日产量为 **{best_annual['production_t_per_day']:.0f} t/d**，年均吨氨成本为 **{best_annual['annual_average_ton_cost_yuan_per_t']:,.2f} 元/t**。",
            "",
            "说明：每种风光场景代表15天，年度统计按360天计算；年均吨氨成本按全年总成本除以全年产氨量计算。",
        ]
    )
    (OUT_DIR / "q2_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_typical_schedule(typical_hourly: pd.DataFrame) -> None:
    set_csee_style()
    pivot = typical_hourly.pivot_table(index="production_t_per_day", columns="hour", values="production_on", aggfunc="first")
    pivot = pivot.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.imshow(pivot.values, cmap=plt.cm.Greens, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{int(v)}" for v in pivot.index])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("时段/h")
    ax.set_ylabel("日产量/t")
    ax.set_title("典型风光场景下不同日产量的最优开机时段")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if pivot.values[i, j] > 0.5:
                ax.text(j, i, "1", ha="center", va="center", fontsize=6, color="white")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16)
    save_figure(fig, OUT_DIR / "q2_fig1_typical_schedule_heatmap_csee")
    plt.close(fig)


def plot_typical_cost_indicator(typical_summary: pd.DataFrame) -> None:
    set_csee_style()
    df = typical_summary.sort_values("production_t_per_day", ascending=True)
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    bars = ax1.bar(x, df["吨氨成本"], color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.6, label="吨氨成本")
    ax1.set_ylabel("吨氨成本/(元/t)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(v)}" for v in df["production_t_per_day"]])
    ax1.set_xlabel("日产量/t")
    ax1.set_ylim(0, df["吨氨成本"].max() * 1.18)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    annotate_bars(ax1, bars, fmt="{:.0f}", dy_frac=0.01)

    ax2 = ax1.twinx()
    ax2.plot(x, df["新能源自发自用比例（物理/政策口径）"] * 100, marker="o", color=CSEE_COLORS["green"], label="自发自用比例")
    ax2.plot(x, df["总用电量绿电比例"] * 100, marker="s", color=CSEE_COLORS["blue"], label="绿电比例")
    ax2.plot(x, df["新能源上网电量比例"] * 100, marker="^", color=CSEE_COLORS["red"], label="上网比例")
    ax2.axhline(60, color=CSEE_COLORS["green"], linestyle="--", linewidth=0.8)
    ax2.axhline(30, color=CSEE_COLORS["blue"], linestyle="--", linewidth=0.8)
    ax2.axhline(20, color=CSEE_COLORS["red"], linestyle="--", linewidth=0.8)
    ax2.set_ylabel("指标值/%")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0,
        columnspacing=1.1,
    )
    ax1.set_title("典型风光场景下吨氨成本与绿电指标")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.24)
    save_figure(fig, OUT_DIR / "q2_fig2_typical_cost_indicator_csee")
    plt.close(fig)


def plot_scenario_cost_box(scenario_summary: pd.DataFrame) -> None:
    set_csee_style()
    ordered = sorted(PRODUCTION_LEVELS_T)
    data = [scenario_summary.loc[scenario_summary["production_t_per_day"] == d, "吨氨成本"] for d in ordered]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bp = ax.boxplot(data, tick_labels=[str(d) for d in ordered], patch_artist=True, widths=0.58)
    for patch in bp["boxes"]:
        patch.set_facecolor(CSEE_COLORS["light_blue"])
        patch.set_edgecolor(CSEE_COLORS["blue"])
    ax.set_xlabel("日产量/t")
    ax.set_ylabel("吨氨成本/(元/t)")
    ax.set_title("24种风光场景下吨氨成本分布")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.15)
    save_figure(fig, OUT_DIR / "q2_fig3_scenario_cost_box_csee")
    plt.close(fig)


def plot_satisfaction_stacked(annual: pd.DataFrame) -> None:
    set_csee_style()
    df = annual.sort_values("production_t_per_day", ascending=True)
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
    ax.set_title("24场景年度绿电指标合格类型统计")
    legend_below(ax, ncol=3, y=-0.20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, OUT_DIR / "q2_fig4_satisfaction_stacked_csee")
    plt.close(fig)


def plot_cost_duration_curve(scenario_summary: pd.DataFrame) -> None:
    set_csee_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for production in sorted(PRODUCTION_LEVELS_T, reverse=True):
        vals = np.sort(scenario_summary.loc[scenario_summary["production_t_per_day"] == production, "吨氨成本"].to_numpy())
        x = np.arange(1, len(vals) + 1) * SCENARIO_DAYS
        ax.step(x, vals, where="post", linewidth=1.4, label=f"{production} t/d")
    ax.set_xlabel("累计代表天数/d")
    ax.set_ylabel("吨氨成本/(元/t)")
    ax.set_title("全年代表场景吨氨成本分布曲线")
    ax.set_xlim(0, 360)
    ax.grid(True, linestyle="--", alpha=0.4)
    legend_below(ax, ncol=5, y=-0.20)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, OUT_DIR / "q2_fig5_annual_cost_duration_csee")
    plt.close(fig)


def plot_purchase_sale_distribution(scenario_summary: pd.DataFrame) -> None:
    set_csee_style()
    df = scenario_summary.sort_values("production_t_per_day")
    levels = sorted(PRODUCTION_LEVELS_T)
    purchase = [df.loc[df["production_t_per_day"] == d, "网购电量"] for d in levels]
    sale = [df.loc[df["production_t_per_day"] == d, "上网电量"] for d in levels]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bp1 = ax.boxplot(purchase, positions=x - 0.16, widths=0.26, patch_artist=True)
    bp2 = ax.boxplot(sale, positions=x + 0.16, widths=0.26, patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor(CSEE_COLORS["light_red"])
        patch.set_edgecolor(CSEE_COLORS["red"])
    for patch in bp2["boxes"]:
        patch.set_facecolor(CSEE_COLORS["light_blue"])
        patch.set_edgecolor(CSEE_COLORS["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in levels])
    ax.set_xlabel("日产量/t")
    ax.set_ylabel("日电量/MWh")
    ax.set_title("24场景日购电量与上网电量分布")
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ["网购电量", "上网电量"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, OUT_DIR / "q2_fig6_purchase_sale_box_csee")
    plt.close(fig)


def plot_scenario_cost_heatmaps(scenario_summary: pd.DataFrame) -> None:
    set_csee_style()
    levels = sorted(PRODUCTION_LEVELS_T, reverse=True)
    vmin = float(scenario_summary["吨氨成本"].min())
    vmax = float(scenario_summary["吨氨成本"].max())
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.4), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    image = None
    for ax, production in zip(axes_flat, levels):
        matrix = (
            scenario_summary.loc[scenario_summary["production_t_per_day"] == production]
            .pivot_table(index="wind_scenario", columns="pv_scenario", values="吨氨成本", aggfunc="first")
            .sort_index()
        )
        image = ax.imshow(matrix.values, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{production} t/d", fontsize=9)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([str(int(c)) for c in matrix.columns])
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([str(int(i)) for i in matrix.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.values[i, j]
                text_color = "white" if (value - vmin) / (vmax - vmin) > 0.62 else "#222222"
                ax.text(j, i, f"{value / 1000:.1f}", ha="center", va="center", fontsize=6.5, color=text_color)
    axes_flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("风电场景")
    for ax in axes[-1, :2]:
        ax.set_xlabel("光伏场景")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_flat[:-1], orientation="vertical", fraction=0.026, pad=0.018)
        cbar.set_label("吨氨成本/(元/t)")
    fig.suptitle("离散开停模式下24场景吨氨成本热力图（格内为千元/t）", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.10, wspace=0.14, hspace=0.26)
    save_figure(fig, OUT_DIR / "q2_fig7_scenario_cost_heatmap_csee")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = SystemParams()

    typical_hourly, typical_summary = run_typical_cases(params)
    scenario_hourly, scenario_summary = run_scenario_cases(params)
    annual = build_annual_summary(scenario_summary)
    distribution = build_distribution_summary(scenario_summary)

    typical_hourly.to_csv(OUT_DIR / "q2_typical_hourly_schedule.csv", index=False, encoding="utf-8-sig")
    typical_summary.to_csv(OUT_DIR / "q2_typical_summary.csv", index=False, encoding="utf-8-sig")
    scenario_hourly.to_csv(OUT_DIR / "q2_all_scenario_hourly_schedule.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(OUT_DIR / "q2_all_scenario_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT_DIR / "q2_annual_summary.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(OUT_DIR / "q2_distribution_summary.csv", index=False, encoding="utf-8-sig")

    write_markdown_summary(typical_summary, scenario_summary, annual)
    plot_typical_schedule(typical_hourly)
    plot_typical_cost_indicator(typical_summary)
    plot_scenario_cost_box(scenario_summary)
    plot_satisfaction_stacked(annual)
    plot_cost_duration_curve(scenario_summary)
    plot_purchase_sale_distribution(scenario_summary)
    plot_scenario_cost_heatmaps(scenario_summary)

    print("Typical summary:")
    print(typical_summary[["production_t_per_day", "required_on_hours", "吨氨成本", "新能源自发自用比例（物理/政策口径）", "总用电量绿电比例", "新能源上网电量比例", "satisfaction_type", "on_hours"]].to_string(index=False))
    print("\nAnnual summary:")
    print(annual.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
