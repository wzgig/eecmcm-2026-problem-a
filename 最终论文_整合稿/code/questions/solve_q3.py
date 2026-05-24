from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_common.data import SystemParams, load_typical_day, load_wind_solar_scenarios
from model_common.metrics import add_process_load, q1_summary
from model_common.plotting import (
    plot_q3_cost_duration_curve,
    plot_q3_cost_delta_heatmaps,
    plot_q3_grid_indicator_compare,
    plot_q3_q2_cost_compare,
    plot_q3_satisfaction_stacked,
    plot_q3_scenario_cost_box,
    plot_q3_typical_load_factor,
    plot_q3_typical_load_factor_lines,
    plot_q3_typical_power_balance,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs"
Q2_OUT_DIR = PROJECT_ROOT / "问题2_离散制氨调节优化" / "outputs"
PRODUCTION_LEVELS_T = [72, 63, 54, 45, 36]
SCENARIO_DAYS = 15
CAPACITY_DAILY_AMMONIA_T = 72.0
MIN_LOAD_FACTOR = 0.10


def add_regular_load_to_scenarios(params: SystemParams) -> pd.DataFrame:
    scenario_df = load_wind_solar_scenarios(params=params, root=PROJECT_ROOT)
    regular = load_typical_day(params=params, root=PROJECT_ROOT)[["hour", "regular_load_mw"]]
    return scenario_df.merge(regular, on="hour", how="left")


def solve_load_factor_milp(
    hourly: pd.DataFrame,
    params: SystemParams,
    production_t_per_day: float,
    big_m_mw: float = 200.0,
) -> np.ndarray:
    n_hour = len(hourly)
    if n_hour != 24:
        raise ValueError(f"Expected 24 hourly rows, got {n_hour}.")

    full_process_mw = params.process_power_mw(CAPACITY_DAILY_AMMONIA_T)
    full_output_t_per_h = params.ammonia_output_t_per_h(CAPACITY_DAILY_AMMONIA_T)
    required_load_factor_sum = production_t_per_day / full_output_t_per_h

    u = np.arange(0, n_hour)
    buy = np.arange(n_hour, 2 * n_hour)
    sell = np.arange(2 * n_hour, 3 * n_hour)
    binary = np.arange(3 * n_hour, 4 * n_hour)
    n_var = 4 * n_hour

    c = np.zeros(n_var)
    full_om_yuan_per_h = (
        params.alk_power_mw(CAPACITY_DAILY_AMMONIA_T) * 1000 * params.alk_om_yuan_per_kwh
        + params.pem_power_mw(CAPACITY_DAILY_AMMONIA_T) * 1000 * params.pem_om_yuan_per_kwh
        + params.ammonia_power_mw(CAPACITY_DAILY_AMMONIA_T) * 1000 * params.ammonia_om_yuan_per_kwh
    )
    c[u] = full_om_yuan_per_h
    c[buy] = hourly["purchase_price_yuan_per_kwh"].to_numpy() * 1000
    c[sell] = -params.sell_price_yuan_per_kwh * 1000

    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    lower[u] = MIN_LOAD_FACTOR
    upper[u] = 1.0
    upper[buy] = big_m_mw
    upper[sell] = big_m_mw
    upper[binary] = 1.0

    rows = []
    lb = []
    ub = []

    renewable = hourly["renewable_mw"].to_numpy()
    regular = hourly["regular_load_mw"].to_numpy()
    for t in range(n_hour):
        row = np.zeros(n_var)
        row[u[t]] = full_process_mw
        row[buy[t]] = -1.0
        row[sell[t]] = 1.0
        rows.append(row)
        rhs = renewable[t] - regular[t]
        lb.append(rhs)
        ub.append(rhs)

    row = np.zeros(n_var)
    row[u] = 1.0
    rows.append(row)
    lb.append(required_load_factor_sum)
    ub.append(required_load_factor_sum)

    for t in range(n_hour):
        row = np.zeros(n_var)
        row[buy[t]] = 1.0
        row[binary[t]] = -big_m_mw
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    for t in range(n_hour):
        row = np.zeros(n_var)
        row[sell[t]] = 1.0
        row[binary[t]] = big_m_mw
        rows.append(row)
        lb.append(-np.inf)
        ub.append(big_m_mw)

    integrality = np.zeros(n_var)
    integrality[binary] = 1

    constraints = LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub))
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": 30, "mip_rel_gap": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed for {production_t_per_day} t/d: {result.message}")

    load_factor = np.asarray(result.x[u], dtype=float)
    load_factor[np.isclose(load_factor, MIN_LOAD_FACTOR, atol=1e-8)] = MIN_LOAD_FACTOR
    load_factor[np.isclose(load_factor, 1.0, atol=1e-8)] = 1.0
    return load_factor


def classify_satisfaction(record: dict[str, float | int | str]) -> None:
    record["satisfied_count"] = int(
        record["新能源自发自用比例是否达标"] + record["总用电量绿电比例是否达标"] + record["新能源上网电量比例是否达标"]
    )
    if record["satisfied_count"] == 3:
        record["satisfaction_type"] = "全满足"
    elif record["satisfied_count"] == 0:
        record["satisfaction_type"] = "全不满足"
    else:
        record["satisfaction_type"] = "部分满足"


def optimize_continuous_schedule(
    base_df: pd.DataFrame,
    params: SystemParams,
    production_t_per_day: float,
    case_name: str,
    wind_scenario: int | None = None,
    pv_scenario: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    hourly = base_df.copy().sort_values("hour").reset_index(drop=True)
    load_factor = solve_load_factor_milp(hourly, params, production_t_per_day)

    hourly["production_load_factor"] = load_factor
    hourly = add_process_load(
        hourly,
        params=params,
        daily_ammonia_t=CAPACITY_DAILY_AMMONIA_T,
        load_factor_col="production_load_factor",
    )
    net_deficit = hourly["total_load_mw"] - hourly["renewable_mw"]
    hourly["grid_purchase_mw"] = net_deficit.clip(lower=0)
    hourly["grid_sale_mw"] = (-net_deficit).clip(lower=0)
    hourly["grid_exchange_mw"] = hourly["grid_purchase_mw"] - hourly["grid_sale_mw"]
    hourly["case_name"] = case_name
    hourly["production_t_per_day"] = production_t_per_day
    hourly["full_load_equivalent_hours"] = production_t_per_day / params.ammonia_output_t_per_h(CAPACITY_DAILY_AMMONIA_T)
    hourly["equipment_utilization"] = production_t_per_day / CAPACITY_DAILY_AMMONIA_T
    hourly["wind_scenario"] = -1 if wind_scenario is None else wind_scenario
    hourly["pv_scenario"] = -1 if pv_scenario is None else pv_scenario

    summary = q1_summary(hourly, params)
    record: dict[str, float | int | str] = {
        "case_name": case_name,
        "wind_scenario": -1 if wind_scenario is None else wind_scenario,
        "pv_scenario": -1 if pv_scenario is None else pv_scenario,
        "production_t_per_day": production_t_per_day,
        "full_load_equivalent_hours": production_t_per_day / params.ammonia_output_t_per_h(CAPACITY_DAILY_AMMONIA_T),
        "equipment_utilization": production_t_per_day / CAPACITY_DAILY_AMMONIA_T,
        "load_factor_min": float(hourly["production_load_factor"].min()),
        "load_factor_mean": float(hourly["production_load_factor"].mean()),
        "load_factor_max": float(hourly["production_load_factor"].max()),
        "load_factor_std": float(hourly["production_load_factor"].std(ddof=0)),
    }
    for _, row in summary.iterrows():
        record[str(row["metric"])] = float(row["value"])
    classify_satisfaction(record)
    return hourly, record


def run_typical_cases(params: SystemParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    typical = load_typical_day(params=params, root=PROJECT_ROOT)
    hourly_parts = []
    records = []
    for production in PRODUCTION_LEVELS_T:
        hourly, record = optimize_continuous_schedule(typical, params, production, "typical")
        hourly_parts.append(hourly)
        records.append(record)
    return pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(records)


def run_scenario_cases(params: SystemParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = add_regular_load_to_scenarios(params)
    hourly_parts = []
    records = []
    for (scenario, wind_id, pv_id), group in scenarios.groupby(["scenario", "wind_scenario", "pv_scenario"], sort=True):
        for production in PRODUCTION_LEVELS_T:
            hourly, record = optimize_continuous_schedule(
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
        annual_total_load = float((group["典型日总用电量"] * SCENARIO_DAYS).sum())
        annual_renewable = float((group["新能源发电量"] * SCENARIO_DAYS).sum())
        annual_purchase = float((group["网购电量"] * SCENARIO_DAYS).sum())
        annual_sale = float((group["上网电量"] * SCENARIO_DAYS).sum())
        annual_self_use = annual_renewable - annual_sale
        type_counts = group["satisfaction_type"].value_counts().to_dict()
        rows.append(
            {
                "production_t_per_day": production,
                "annual_days": int(len(group) * SCENARIO_DAYS),
                "annual_ammonia_t": annual_ammonia,
                "annual_total_cost_yuan": annual_total_cost,
                "annual_average_ton_cost_yuan_per_t": annual_total_cost / annual_ammonia,
                "annual_total_load_mwh": annual_total_load,
                "annual_renewable_mwh": annual_renewable,
                "annual_purchase_mwh": annual_purchase,
                "annual_sale_mwh": annual_sale,
                "annual_self_use_mwh": annual_self_use,
                "annual_self_use_ratio": annual_self_use / annual_renewable,
                "annual_green_power_ratio": annual_self_use / annual_total_load,
                "annual_sale_ratio": annual_sale / annual_renewable,
                "equipment_utilization": production / CAPACITY_DAILY_AMMONIA_T,
                "full_satisfied_scenarios": int(type_counts.get("全满足", 0)),
                "partial_satisfied_scenarios": int(type_counts.get("部分满足", 0)),
                "none_satisfied_scenarios": int(type_counts.get("全不满足", 0)),
                "full_satisfied_days": int(type_counts.get("全满足", 0) * SCENARIO_DAYS),
                "partial_satisfied_days": int(type_counts.get("部分满足", 0) * SCENARIO_DAYS),
                "none_satisfied_days": int(type_counts.get("全不满足", 0) * SCENARIO_DAYS),
            }
        )
    return pd.DataFrame(rows).sort_values("production_t_per_day", ascending=False).reset_index(drop=True)


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


def build_weighted_annual_from_scenario(scenario_summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for production, group in scenario_summary.groupby("production_t_per_day", sort=False):
        total_cost = float((group["日总成本（扣除售电收益后）"] * SCENARIO_DAYS).sum())
        ammonia = float((group["日产氨量"] * SCENARIO_DAYS).sum())
        total_load = float((group["典型日总用电量"] * SCENARIO_DAYS).sum())
        renewable = float((group["新能源发电量"] * SCENARIO_DAYS).sum())
        purchase = float((group["网购电量"] * SCENARIO_DAYS).sum())
        sale = float((group["上网电量"] * SCENARIO_DAYS).sum())
        self_use = renewable - sale
        type_counts = group["satisfaction_type"].value_counts().to_dict()
        rows.append(
            {
                "production_t_per_day": production,
                f"{prefix}_annual_total_cost_yuan": total_cost,
                f"{prefix}_annual_average_ton_cost_yuan_per_t": total_cost / ammonia,
                f"{prefix}_annual_purchase_mwh": purchase,
                f"{prefix}_annual_sale_mwh": sale,
                f"{prefix}_annual_self_use_ratio": self_use / renewable,
                f"{prefix}_annual_green_power_ratio": self_use / total_load,
                f"{prefix}_annual_sale_ratio": sale / renewable,
                f"{prefix}_full_satisfied_days": int(type_counts.get("全满足", 0) * SCENARIO_DAYS),
                f"{prefix}_partial_satisfied_days": int(type_counts.get("部分满足", 0) * SCENARIO_DAYS),
                f"{prefix}_none_satisfied_days": int(type_counts.get("全不满足", 0) * SCENARIO_DAYS),
            }
        )
    return pd.DataFrame(rows)


def build_q2_compare(q3_scenario_summary: pd.DataFrame) -> pd.DataFrame:
    q2_path = Q2_OUT_DIR / "q2_all_scenario_summary.csv"
    if not q2_path.exists():
        raise FileNotFoundError(f"Cannot find Q2 scenario summary: {q2_path}")
    q2_scenario_summary = pd.read_csv(q2_path)
    q2 = build_weighted_annual_from_scenario(q2_scenario_summary, "q2")
    q3 = build_weighted_annual_from_scenario(q3_scenario_summary, "q3")
    compare = q2.merge(q3, on="production_t_per_day", how="inner")
    compare["equipment_utilization"] = compare["production_t_per_day"] / CAPACITY_DAILY_AMMONIA_T
    compare["ton_cost_delta_yuan_per_t"] = (
        compare["q3_annual_average_ton_cost_yuan_per_t"] - compare["q2_annual_average_ton_cost_yuan_per_t"]
    )
    compare["ton_cost_delta_percent"] = (
        compare["q3_annual_average_ton_cost_yuan_per_t"] / compare["q2_annual_average_ton_cost_yuan_per_t"] - 1.0
    )
    compare["purchase_delta_mwh"] = compare["q3_annual_purchase_mwh"] - compare["q2_annual_purchase_mwh"]
    compare["sale_delta_mwh"] = compare["q3_annual_sale_mwh"] - compare["q2_annual_sale_mwh"]
    compare["self_use_ratio_delta"] = compare["q3_annual_self_use_ratio"] - compare["q2_annual_self_use_ratio"]
    compare["green_power_ratio_delta"] = compare["q3_annual_green_power_ratio"] - compare["q2_annual_green_power_ratio"]
    compare["sale_ratio_delta"] = compare["q3_annual_sale_ratio"] - compare["q2_annual_sale_ratio"]
    compare["full_satisfied_days_delta"] = compare["q3_full_satisfied_days"] - compare["q2_full_satisfied_days"]
    compare["partial_satisfied_days_delta"] = compare["q3_partial_satisfied_days"] - compare["q2_partial_satisfied_days"]
    compare["none_satisfied_days_delta"] = compare["q3_none_satisfied_days"] - compare["q2_none_satisfied_days"]
    return compare.sort_values("production_t_per_day", ascending=False).reset_index(drop=True)


def write_markdown_summary(
    typical_summary: pd.DataFrame,
    annual: pd.DataFrame,
    distribution: pd.DataFrame,
    compare: pd.DataFrame,
) -> None:
    best_typical = typical_summary.loc[typical_summary["吨氨成本"].idxmin()]
    typical_full = typical_summary[typical_summary["satisfaction_type"] == "全满足"]
    best_typical_full = typical_full.loc[typical_full["吨氨成本"].idxmin()] if not typical_full.empty else None
    best_annual = annual.loc[annual["annual_average_ton_cost_yuan_per_t"].idxmin()]

    lines = [
        "# 问题三结果摘要",
        "",
        "## 典型风光场景",
        "",
        "| 日产量/t | 等效满负荷小时 | 平均负荷率 | 负荷率范围 | 吨氨成本/(元/t) | 自发自用比例 | 绿电比例 | 上网比例 | 合格类型 |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in typical_summary.sort_values("production_t_per_day", ascending=False).iterrows():
        lines.append(
            "| {production:.0f} | {hours:.2f} | {mean:.2%} | {vmin:.1%}-{vmax:.1%} | {cost:,.2f} | {self:.2%} | {green:.2%} | {sale:.2%} | {stype} |".format(
                production=row["production_t_per_day"],
                hours=row["full_load_equivalent_hours"],
                mean=row["load_factor_mean"],
                vmin=row["load_factor_min"],
                vmax=row["load_factor_max"],
                cost=row["吨氨成本"],
                self=row["新能源自发自用比例（物理/政策口径）"],
                green=row["总用电量绿电比例"],
                sale=row["新能源上网电量比例"],
                stype=row["satisfaction_type"],
            )
        )

    lines.extend(
        [
            "",
            f"典型风光场景下，吨氨成本最低的连续调节方案为 **{best_typical['production_t_per_day']:.0f} t/d**，吨氨成本为 **{best_typical['吨氨成本']:,.2f} 元/t**。",
        ]
    )
    if best_typical_full is not None:
        lines.append(
            f"若要求三项绿电指标全部满足，典型场景下最低成本方案为 **{best_typical_full['production_t_per_day']:.0f} t/d**，吨氨成本为 **{best_typical_full['吨氨成本']:,.2f} 元/t**。"
        )
    else:
        lines.append("典型场景下没有日产量方案三项绿电指标全部满足。")

    lines.extend(
        [
            "",
            "## 24种风光场景年度加权统计",
            "",
            "| 日产量/t | 年产氨量/t | 年总成本/万元 | 年均吨氨成本/(元/t) | 年购电量/MWh | 年上网电量/MWh | 年度自发自用比例 | 年度绿电比例 | 年度上网比例 | 全满足/天 | 部分满足/天 | 全不满足/天 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in annual.iterrows():
        lines.append(
            "| {production:.0f} | {ammonia:,.0f} | {total_cost:,.2f} | {avg_cost:,.2f} | {purchase:,.2f} | {sale:,.2f} | {self:.2%} | {green:.2%} | {sale_ratio:.2%} | {full_days} | {partial_days} | {none_days} |".format(
                production=row["production_t_per_day"],
                ammonia=row["annual_ammonia_t"],
                total_cost=row["annual_total_cost_yuan"] / 10000,
                avg_cost=row["annual_average_ton_cost_yuan_per_t"],
                purchase=row["annual_purchase_mwh"],
                sale=row["annual_sale_mwh"],
                self=row["annual_self_use_ratio"],
                green=row["annual_green_power_ratio"],
                sale_ratio=row["annual_sale_ratio"],
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
            "## 与问题二离散开停机结果对比",
            "",
            "| 日产量/t | 设备利用率 | Q2吨氨成本 | Q3吨氨成本 | 成本变化 | Q2购电/MWh | Q3购电/MWh | Q2上网/MWh | Q3上网/MWh | Q2全满足/天 | Q3全满足/天 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in compare.iterrows():
        lines.append(
            "| {production:.0f} | {util:.2%} | {q2_cost:,.2f} | {q3_cost:,.2f} | {delta:+,.2f} | {q2_buy:,.2f} | {q3_buy:,.2f} | {q2_sale:,.2f} | {q3_sale:,.2f} | {q2_full} | {q3_full} |".format(
                production=row["production_t_per_day"],
                util=row["equipment_utilization"],
                q2_cost=row["q2_annual_average_ton_cost_yuan_per_t"],
                q3_cost=row["q3_annual_average_ton_cost_yuan_per_t"],
                delta=row["ton_cost_delta_yuan_per_t"],
                q2_buy=row["q2_annual_purchase_mwh"],
                q3_buy=row["q3_annual_purchase_mwh"],
                q2_sale=row["q2_annual_sale_mwh"],
                q3_sale=row["q3_annual_sale_mwh"],
                q2_full=int(row["q2_full_satisfied_days"]),
                q3_full=int(row["q3_full_satisfied_days"]),
            )
        )

    lines.extend(
        [
            "",
            "## 24场景分布特征",
            "",
            "| 日产量/t | 吨氨成本最小值 | 吨氨成本均值 | 吨氨成本最大值 | 平均日购电量/MWh | 平均日上网电量/MWh | 平均上网比例 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in distribution.iterrows():
        lines.append(
            "| {production:.0f} | {min_cost:,.2f} | {mean_cost:,.2f} | {max_cost:,.2f} | {purchase:,.2f} | {sale:,.2f} | {sale_ratio:.2%} |".format(
                production=row["production_t_per_day"],
                min_cost=row["ton_cost_min"],
                mean_cost=row["ton_cost_mean"],
                max_cost=row["ton_cost_max"],
                purchase=row["purchase_mwh_mean"],
                sale=row["sale_mwh_mean"],
                sale_ratio=row["sale_ratio_mean"],
            )
        )

    lines.extend(
        [
            "",
            "说明：问题三主模型采用全天连续运行假设，制氨负荷率满足10%-100%；每种风光场景代表15天，年度统计按360天计算。模型通过购售电互斥二进制变量避免同一小时同时购电和售电。",
        ]
    )
    (OUT_DIR / "q3_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = SystemParams()

    typical_hourly, typical_summary = run_typical_cases(params)
    scenario_hourly, scenario_summary = run_scenario_cases(params)
    annual = build_annual_summary(scenario_summary)
    distribution = build_distribution_summary(scenario_summary)
    compare = build_q2_compare(scenario_summary)

    typical_hourly.to_csv(OUT_DIR / "q3_typical_hourly_schedule.csv", index=False, encoding="utf-8-sig")
    typical_summary.to_csv(OUT_DIR / "q3_typical_summary.csv", index=False, encoding="utf-8-sig")
    scenario_hourly.to_csv(OUT_DIR / "q3_all_scenario_hourly_schedule.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(OUT_DIR / "q3_all_scenario_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT_DIR / "q3_annual_summary.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(OUT_DIR / "q3_distribution_summary.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(OUT_DIR / "q3_compare_q2_summary.csv", index=False, encoding="utf-8-sig")

    write_markdown_summary(typical_summary, annual, distribution, compare)

    plot_q3_typical_load_factor(typical_hourly, OUT_DIR)
    plot_q3_typical_load_factor_lines(typical_hourly, OUT_DIR)
    plot_q3_q2_cost_compare(compare, OUT_DIR)
    plot_q3_grid_indicator_compare(compare, OUT_DIR)
    plot_q3_scenario_cost_box(scenario_summary, OUT_DIR, PRODUCTION_LEVELS_T)
    plot_q3_satisfaction_stacked(annual, OUT_DIR)
    plot_q3_cost_duration_curve(scenario_summary, OUT_DIR, PRODUCTION_LEVELS_T, SCENARIO_DAYS)
    q2_scenario_summary = pd.read_csv(Q2_OUT_DIR / "q2_all_scenario_summary.csv")
    plot_q3_cost_delta_heatmaps(q2_scenario_summary, scenario_summary, OUT_DIR, PRODUCTION_LEVELS_T)

    typical_full = typical_summary[typical_summary["satisfaction_type"] == "全满足"]
    selected = typical_full.loc[typical_full["吨氨成本"].idxmin()] if not typical_full.empty else typical_summary.loc[typical_summary["吨氨成本"].idxmin()]
    selected_hourly = typical_hourly[typical_hourly["production_t_per_day"] == selected["production_t_per_day"]]
    plot_q3_typical_power_balance(selected_hourly, OUT_DIR, float(selected["production_t_per_day"]))

    print("Typical summary:")
    print(
        typical_summary[
            [
                "production_t_per_day",
                "full_load_equivalent_hours",
                "load_factor_min",
                "load_factor_mean",
                "load_factor_max",
                "吨氨成本",
                "新能源自发自用比例（物理/政策口径）",
                "总用电量绿电比例",
                "新能源上网电量比例",
                "satisfaction_type",
            ]
        ].to_string(index=False)
    )
    print("\nAnnual summary:")
    print(annual.to_string(index=False))
    print("\nQ2/Q3 comparison:")
    print(
        compare[
            [
                "production_t_per_day",
                "q2_annual_average_ton_cost_yuan_per_t",
                "q3_annual_average_ton_cost_yuan_per_t",
                "ton_cost_delta_yuan_per_t",
                "q2_annual_purchase_mwh",
                "q3_annual_purchase_mwh",
                "q2_annual_sale_mwh",
                "q3_annual_sale_mwh",
                "q2_full_satisfied_days",
                "q3_full_satisfied_days",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
