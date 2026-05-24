from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_common.data import SystemParams, load_typical_day, load_wind_solar_scenarios
from model_common.metrics import add_process_load
from model_common.plotting import CSEE_COLORS, figure_legend_below, save_figure, set_csee_style


OUT_DIR = Path(__file__).resolve().parent / "outputs"
SCENARIO_DAYS = 15
CAPACITY_DAILY_AMMONIA_T = 72.0
MIN_LOAD_FACTOR = 0.10
BATTERY_OM_YUAN_PER_KWH = 0.01
BATTERY_C_RATE = 1.0
CAPACITY_SCAN_MAX_MWH = 300.0
CAPACITY_SCAN_STEP_MWH = 5.0

# Lexicographic-like weights for the off-grid storage dispatch MILP.
# Regular-load shedding must be avoided first; ammonia output is then maximized.
SHED_PENALTY_YUAN_PER_MWH = 1.0e7
AMMONIA_REWARD_YUAN_PER_T = 2.0e5
CURTAILMENT_TIE_BREAK_YUAN_PER_MWH = 1.0


def add_regular_load_to_scenarios(params: SystemParams) -> pd.DataFrame:
    scenario_df = load_wind_solar_scenarios(params=params, root=PROJECT_ROOT)
    regular = load_typical_day(params=params, root=PROJECT_ROOT)[["hour", "regular_load_mw"]]
    return scenario_df.merge(regular, on="hour", how="left")


def full_process_om_yuan_per_h(params: SystemParams) -> float:
    return (
        params.alk_power_mw(CAPACITY_DAILY_AMMONIA_T) * 1000 * params.alk_om_yuan_per_kwh
        + params.pem_power_mw(CAPACITY_DAILY_AMMONIA_T) * 1000 * params.pem_om_yuan_per_kwh
        + params.ammonia_power_mw(CAPACITY_DAILY_AMMONIA_T) * 1000 * params.ammonia_om_yuan_per_kwh
    )


def add_process_columns(hourly: pd.DataFrame, params: SystemParams, load_factor: np.ndarray) -> pd.DataFrame:
    out = hourly.copy()
    out["production_load_factor"] = np.asarray(load_factor, dtype=float)
    return add_process_load(
        out,
        params=params,
        daily_ammonia_t=CAPACITY_DAILY_AMMONIA_T,
        load_factor_col="production_load_factor",
    )


def compute_offgrid_summary(
    hourly: pd.DataFrame,
    params: SystemParams,
    storage_capacity_mwh: float,
    mode: str,
    scenario: str,
    wind_scenario: int,
    pv_scenario: int,
) -> dict[str, float | int | str]:
    wind_mwh = float(hourly["wind_mw"].sum())
    pv_mwh = float(hourly["pv_mw"].sum())
    renewable_mwh = wind_mwh + pv_mwh
    regular_demand_mwh = float(hourly["regular_load_mw"].sum())
    regular_shed_mwh = float(hourly["regular_shed_mw"].sum())
    regular_served_mwh = regular_demand_mwh - regular_shed_mwh
    process_load_mwh = float(hourly["process_load_mw"].sum())
    served_load_mwh = regular_served_mwh + process_load_mwh
    demanded_load_mwh = regular_demand_mwh + process_load_mwh
    curtailment_mwh = float(hourly["curtailment_mw"].sum())
    renewable_used_mwh = renewable_mwh - curtailment_mwh
    storage_charge_mwh = float(hourly["storage_charge_mw"].sum())
    storage_discharge_mwh = float(hourly["storage_discharge_mw"].sum())
    storage_throughput_mwh = storage_charge_mwh + storage_discharge_mwh
    ammonia_t = float(hourly["ammonia_output_t"].sum())

    wind_cost = wind_mwh * 1000 * params.wind_lcoe_yuan_per_kwh
    pv_cost = pv_mwh * 1000 * params.pv_lcoe_yuan_per_kwh
    alk_om = float((hourly["alk_load_mw"] * 1000 * params.alk_om_yuan_per_kwh).sum())
    pem_om = float((hourly["pem_load_mw"] * 1000 * params.pem_om_yuan_per_kwh).sum())
    ammonia_om = float((hourly["ammonia_load_mw"] * 1000 * params.ammonia_om_yuan_per_kwh).sum())
    storage_daily_capital = storage_capacity_mwh * 1000 * params.storage_capex_yuan_per_kwh / (
        params.storage_life_year * 360
    )
    storage_om = storage_throughput_mwh * 1000 * BATTERY_OM_YUAN_PER_KWH
    total_cost = wind_cost + pv_cost + alk_om + pem_om + ammonia_om + storage_daily_capital + storage_om

    storage_loss_mwh = max(renewable_used_mwh - served_load_mwh, 0.0)
    return {
        "mode": mode,
        "scenario": scenario,
        "wind_scenario": int(wind_scenario),
        "pv_scenario": int(pv_scenario),
        "storage_capacity_mwh": float(storage_capacity_mwh),
        "storage_power_mw": float(storage_capacity_mwh * BATTERY_C_RATE),
        "日产氨量": ammonia_t,
        "设备利用率": ammonia_t / CAPACITY_DAILY_AMMONIA_T,
        "常规负荷需求电量": regular_demand_mwh,
        "常规负荷缺供电量": regular_shed_mwh,
        "常规负荷供电保障率": regular_served_mwh / regular_demand_mwh,
        "制氨工艺用电量": process_load_mwh,
        "离网服务负荷电量": served_load_mwh,
        "离网需求电量": demanded_load_mwh,
        "风电发电量": wind_mwh,
        "光伏发电量": pv_mwh,
        "新能源可发电量": renewable_mwh,
        "新能源实际利用电量": renewable_used_mwh,
        "弃风弃光电量": curtailment_mwh,
        "弃风弃光率": curtailment_mwh / renewable_mwh if renewable_mwh > 0 else np.nan,
        "风光利用率": renewable_used_mwh / renewable_mwh if renewable_mwh > 0 else np.nan,
        "储能充电量": storage_charge_mwh,
        "储能放电量": storage_discharge_mwh,
        "储能等效吞吐量": storage_throughput_mwh,
        "储能损耗及自损耗估计": storage_loss_mwh,
        "新能源上网电量比例": 0.0,
        "新能源自发自用比例（离网实际消纳口径）": renewable_used_mwh / renewable_mwh if renewable_mwh > 0 else np.nan,
        "总用电量绿电比例（离网需求保障口径）": served_load_mwh / demanded_load_mwh if demanded_load_mwh > 0 else np.nan,
        "风电成本": wind_cost,
        "光伏成本": pv_cost,
        "碱性电解槽运维成本": alk_om,
        "PEM电解槽运维成本": pem_om,
        "合成氨装置运维成本": ammonia_om,
        "储能年化投资成本": storage_daily_capital,
        "储能充放电运维成本": storage_om,
        "日总成本": total_cost,
        "吨氨成本": total_cost / ammonia_t if ammonia_t > 1e-9 else np.nan,
        "能源自治是否满足常规负荷": int(regular_shed_mwh <= 1e-7),
    }


def solve_no_storage_offgrid(
    hourly: pd.DataFrame,
    params: SystemParams,
    scenario: str,
    wind_scenario: int,
    pv_scenario: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    full_process_mw = params.process_power_mw(CAPACITY_DAILY_AMMONIA_T)
    min_process_mw = MIN_LOAD_FACTOR * full_process_mw
    hourly = hourly.sort_values("hour").reset_index(drop=True).copy()
    available = hourly["renewable_mw"].to_numpy() - hourly["regular_load_mw"].to_numpy()

    process = np.zeros(len(hourly))
    curtailment = np.zeros(len(hourly))
    shed = np.zeros(len(hourly))
    for i, value in enumerate(available):
        if value < 0:
            shed[i] = -value
        elif value < min_process_mw:
            curtailment[i] = value
        else:
            process[i] = min(value, full_process_mw)
            curtailment[i] = max(value - full_process_mw, 0.0)

    load_factor = process / full_process_mw
    hourly = add_process_columns(hourly, params, load_factor)
    hourly["production_on"] = (hourly["production_load_factor"] > 1e-9).astype(int)
    hourly["storage_charge_mw"] = 0.0
    hourly["storage_discharge_mw"] = 0.0
    hourly["storage_soc_mwh"] = 0.0
    hourly["storage_mode_binary"] = 0
    hourly["curtailment_mw"] = curtailment
    hourly["regular_shed_mw"] = shed
    hourly["offgrid_balance_error_mw"] = (
        hourly["renewable_mw"]
        + hourly["storage_discharge_mw"]
        + hourly["regular_shed_mw"]
        - hourly["regular_load_mw"]
        - hourly["process_load_mw"]
        - hourly["storage_charge_mw"]
        - hourly["curtailment_mw"]
    )
    hourly["scenario"] = scenario
    hourly["wind_scenario"] = int(wind_scenario)
    hourly["pv_scenario"] = int(pv_scenario)
    hourly["mode"] = "offgrid_no_storage"

    summary = compute_offgrid_summary(
        hourly,
        params=params,
        storage_capacity_mwh=0.0,
        mode="offgrid_no_storage",
        scenario=scenario,
        wind_scenario=wind_scenario,
        pv_scenario=pv_scenario,
    )
    return hourly, summary


def solve_storage_offgrid(
    hourly: pd.DataFrame,
    params: SystemParams,
    storage_capacity_mwh: float,
    scenario: str,
    wind_scenario: int,
    pv_scenario: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    n_hour = len(hourly)
    if n_hour != 24:
        raise ValueError(f"Expected 24 hourly rows, got {n_hour}.")

    hourly = hourly.sort_values("hour").reset_index(drop=True).copy()
    full_process_mw = params.process_power_mw(CAPACITY_DAILY_AMMONIA_T)
    full_output_t_per_h = params.ammonia_output_t_per_h(CAPACITY_DAILY_AMMONIA_T)
    storage_power_mw = storage_capacity_mwh * BATTERY_C_RATE

    idx_u = np.arange(0, n_hour)
    idx_z = np.arange(n_hour, 2 * n_hour)
    idx_ch = np.arange(2 * n_hour, 3 * n_hour)
    idx_dis = np.arange(3 * n_hour, 4 * n_hour)
    idx_soc = np.arange(4 * n_hour, 5 * n_hour)
    idx_cur = np.arange(5 * n_hour, 6 * n_hour)
    idx_shed = np.arange(6 * n_hour, 7 * n_hour)
    idx_y = np.arange(7 * n_hour, 8 * n_hour)
    n_var = 8 * n_hour

    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[idx_u] = 1.0
    upper[idx_z] = 1.0
    upper[idx_ch] = storage_power_mw
    upper[idx_dis] = storage_power_mw
    upper[idx_soc] = storage_capacity_mwh
    upper[idx_shed] = hourly["regular_load_mw"].to_numpy()
    upper[idx_y] = 1.0

    c = np.zeros(n_var)
    c[idx_u] = -AMMONIA_REWARD_YUAN_PER_T * full_output_t_per_h
    c[idx_shed] = SHED_PENALTY_YUAN_PER_MWH
    c[idx_cur] = CURTAILMENT_TIE_BREAK_YUAN_PER_MWH
    c[idx_ch] = BATTERY_OM_YUAN_PER_KWH * 1000
    c[idx_dis] = BATTERY_OM_YUAN_PER_KWH * 1000

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    renewable = hourly["renewable_mw"].to_numpy()
    regular = hourly["regular_load_mw"].to_numpy()

    for t in range(n_hour):
        row = np.zeros(n_var)
        row[idx_u[t]] = full_process_mw
        row[idx_ch[t]] = 1.0
        row[idx_cur[t]] = 1.0
        row[idx_dis[t]] = -1.0
        row[idx_shed[t]] = -1.0
        rows.append(row)
        rhs = renewable[t] - regular[t]
        lb.append(rhs)
        ub.append(rhs)

    for t in range(n_hour):
        row = np.zeros(n_var)
        row[idx_u[t]] = 1.0
        row[idx_z[t]] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

        row = np.zeros(n_var)
        row[idx_u[t]] = -1.0
        row[idx_z[t]] = MIN_LOAD_FACTOR
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    for t in range(n_hour):
        row = np.zeros(n_var)
        row[idx_ch[t]] = 1.0
        row[idx_y[t]] = -storage_power_mw
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

        row = np.zeros(n_var)
        row[idx_dis[t]] = 1.0
        row[idx_y[t]] = storage_power_mw
        rows.append(row)
        lb.append(-np.inf)
        ub.append(storage_power_mw)

    eta_ch = params.storage_charge_eff
    eta_dis = params.storage_discharge_eff
    rho = params.storage_self_loss_ratio
    for t in range(n_hour):
        prev = (t - 1) % n_hour
        row = np.zeros(n_var)
        row[idx_soc[t]] = 1.0
        row[idx_soc[prev]] = -(1.0 - rho)
        row[idx_ch[t]] = -eta_ch
        row[idx_dis[t]] = 1.0 / eta_dis
        rows.append(row)
        lb.append(0.0)
        ub.append(0.0)

    integrality = np.zeros(n_var)
    integrality[idx_z] = 1
    integrality[idx_y] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"time_limit": 30, "mip_rel_gap": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Storage off-grid MILP failed for {scenario}, E={storage_capacity_mwh:.1f} MWh: {result.message}")

    x = np.asarray(result.x)
    load_factor = np.clip(x[idx_u], 0, 1)
    hourly = add_process_columns(hourly, params, load_factor)
    hourly["production_on"] = np.rint(np.clip(x[idx_z], 0, 1)).astype(int)
    hourly["storage_charge_mw"] = np.clip(x[idx_ch], 0, None)
    hourly["storage_discharge_mw"] = np.clip(x[idx_dis], 0, None)
    hourly["storage_soc_mwh"] = np.clip(x[idx_soc], 0, storage_capacity_mwh)
    hourly["curtailment_mw"] = np.clip(x[idx_cur], 0, None)
    hourly["regular_shed_mw"] = np.clip(x[idx_shed], 0, None)
    hourly["storage_mode_binary"] = np.rint(np.clip(x[idx_y], 0, 1)).astype(int)
    hourly["offgrid_balance_error_mw"] = (
        hourly["renewable_mw"]
        + hourly["storage_discharge_mw"]
        + hourly["regular_shed_mw"]
        - hourly["regular_load_mw"]
        - hourly["process_load_mw"]
        - hourly["storage_charge_mw"]
        - hourly["curtailment_mw"]
    )
    hourly["scenario"] = scenario
    hourly["wind_scenario"] = int(wind_scenario)
    hourly["pv_scenario"] = int(pv_scenario)
    hourly["mode"] = "offgrid_storage"
    hourly["storage_capacity_mwh"] = storage_capacity_mwh

    summary = compute_offgrid_summary(
        hourly,
        params=params,
        storage_capacity_mwh=storage_capacity_mwh,
        mode="offgrid_storage",
        scenario=scenario,
        wind_scenario=wind_scenario,
        pv_scenario=pv_scenario,
    )
    return hourly, summary


def compute_grid_summary(
    hourly: pd.DataFrame,
    params: SystemParams,
    mode: str,
    scenario: str,
    wind_scenario: int,
    pv_scenario: int,
) -> dict[str, float | int | str]:
    total_load_mwh = float(hourly["total_load_mw"].sum())
    regular_load_mwh = float(hourly["regular_load_mw"].sum())
    process_load_mwh = float(hourly["process_load_mw"].sum())
    wind_mwh = float(hourly["wind_mw"].sum())
    pv_mwh = float(hourly["pv_mw"].sum())
    renewable_mwh = wind_mwh + pv_mwh
    purchase_mwh = float(hourly["grid_purchase_mw"].sum())
    sale_mwh = float(hourly["grid_sale_mw"].sum())
    ammonia_t = float(hourly["ammonia_output_t"].sum())
    self_use_mwh = renewable_mwh - sale_mwh

    wind_cost = wind_mwh * 1000 * params.wind_lcoe_yuan_per_kwh
    pv_cost = pv_mwh * 1000 * params.pv_lcoe_yuan_per_kwh
    purchase_cost = float((hourly["grid_purchase_mw"] * 1000 * hourly["purchase_price_yuan_per_kwh"]).sum())
    alk_om = float((hourly["alk_load_mw"] * 1000 * params.alk_om_yuan_per_kwh).sum())
    pem_om = float((hourly["pem_load_mw"] * 1000 * params.pem_om_yuan_per_kwh).sum())
    ammonia_om = float((hourly["ammonia_load_mw"] * 1000 * params.ammonia_om_yuan_per_kwh).sum())
    sale_revenue = sale_mwh * 1000 * params.sell_price_yuan_per_kwh
    total_cost = wind_cost + pv_cost + purchase_cost + alk_om + pem_om + ammonia_om - sale_revenue

    return {
        "mode": mode,
        "scenario": scenario,
        "wind_scenario": int(wind_scenario),
        "pv_scenario": int(pv_scenario),
        "日产氨量": ammonia_t,
        "设备利用率": ammonia_t / CAPACITY_DAILY_AMMONIA_T,
        "典型日总用电量": total_load_mwh,
        "常规电负荷用电量": regular_load_mwh,
        "制氨工艺用电量": process_load_mwh,
        "风电发电量": wind_mwh,
        "光伏发电量": pv_mwh,
        "新能源发电量": renewable_mwh,
        "网购电量": purchase_mwh,
        "上网电量": sale_mwh,
        "新能源自发自用电量（物理/政策口径）": self_use_mwh,
        "新能源自发自用比例（物理/政策口径）": self_use_mwh / renewable_mwh if renewable_mwh > 0 else np.nan,
        "总用电量绿电比例": self_use_mwh / total_load_mwh if total_load_mwh > 0 else np.nan,
        "新能源上网电量比例": sale_mwh / renewable_mwh if renewable_mwh > 0 else np.nan,
        "风电成本": wind_cost,
        "光伏成本": pv_cost,
        "购电成本": purchase_cost,
        "碱性电解槽运维成本": alk_om,
        "PEM电解槽运维成本": pem_om,
        "合成氨装置运维成本": ammonia_om,
        "余电上网收益": sale_revenue,
        "日总成本（扣除售电收益后）": total_cost,
        "吨氨成本": total_cost / ammonia_t if ammonia_t > 1e-9 else np.nan,
    }


def solve_grid_connected_same_production(
    hourly: pd.DataFrame,
    params: SystemParams,
    target_ammonia_t: float,
    scenario: str,
    wind_scenario: int,
    pv_scenario: int,
    big_m_mw: float = 250.0,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    n_hour = len(hourly)
    if n_hour != 24:
        raise ValueError(f"Expected 24 hourly rows, got {n_hour}.")

    hourly = hourly.sort_values("hour").reset_index(drop=True).copy()
    full_process_mw = params.process_power_mw(CAPACITY_DAILY_AMMONIA_T)
    full_output_t_per_h = params.ammonia_output_t_per_h(CAPACITY_DAILY_AMMONIA_T)

    idx_u = np.arange(0, n_hour)
    idx_z = np.arange(n_hour, 2 * n_hour)
    idx_buy = np.arange(2 * n_hour, 3 * n_hour)
    idx_sell = np.arange(3 * n_hour, 4 * n_hour)
    idx_b = np.arange(4 * n_hour, 5 * n_hour)
    n_var = 5 * n_hour

    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[idx_u] = 1.0
    upper[idx_z] = 1.0
    upper[idx_buy] = big_m_mw
    upper[idx_sell] = big_m_mw
    upper[idx_b] = 1.0

    c = np.zeros(n_var)
    c[idx_u] = full_process_om_yuan_per_h(params)
    c[idx_buy] = hourly["purchase_price_yuan_per_kwh"].to_numpy() * 1000
    c[idx_sell] = -params.sell_price_yuan_per_kwh * 1000

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    renewable = hourly["renewable_mw"].to_numpy()
    regular = hourly["regular_load_mw"].to_numpy()
    for t in range(n_hour):
        row = np.zeros(n_var)
        row[idx_u[t]] = full_process_mw
        row[idx_buy[t]] = -1.0
        row[idx_sell[t]] = 1.0
        rows.append(row)
        rhs = renewable[t] - regular[t]
        lb.append(rhs)
        ub.append(rhs)

    row = np.zeros(n_var)
    row[idx_u] = full_output_t_per_h
    rows.append(row)
    lb.append(target_ammonia_t)
    ub.append(target_ammonia_t)

    for t in range(n_hour):
        row = np.zeros(n_var)
        row[idx_u[t]] = 1.0
        row[idx_z[t]] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

        row = np.zeros(n_var)
        row[idx_u[t]] = -1.0
        row[idx_z[t]] = MIN_LOAD_FACTOR
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

        row = np.zeros(n_var)
        row[idx_buy[t]] = 1.0
        row[idx_b[t]] = -big_m_mw
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

        row = np.zeros(n_var)
        row[idx_sell[t]] = 1.0
        row[idx_b[t]] = big_m_mw
        rows.append(row)
        lb.append(-np.inf)
        ub.append(big_m_mw)

    integrality = np.zeros(n_var)
    integrality[idx_z] = 1
    integrality[idx_b] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"time_limit": 30, "mip_rel_gap": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Grid-connected MILP failed for {scenario}, target={target_ammonia_t:.4f} t: {result.message}")

    x = np.asarray(result.x)
    load_factor = np.clip(x[idx_u], 0, 1)
    hourly = add_process_columns(hourly, params, load_factor)
    hourly["production_on"] = np.rint(np.clip(x[idx_z], 0, 1)).astype(int)
    hourly["grid_purchase_mw"] = np.clip(x[idx_buy], 0, None)
    hourly["grid_sale_mw"] = np.clip(x[idx_sell], 0, None)
    hourly["grid_exchange_mw"] = hourly["grid_purchase_mw"] - hourly["grid_sale_mw"]
    hourly["purchase_sale_binary"] = np.rint(np.clip(x[idx_b], 0, 1)).astype(int)
    hourly["target_ammonia_t"] = target_ammonia_t
    hourly["scenario"] = scenario
    hourly["wind_scenario"] = int(wind_scenario)
    hourly["pv_scenario"] = int(pv_scenario)
    hourly["mode"] = "grid_connected_same_production"

    summary = compute_grid_summary(
        hourly,
        params=params,
        mode="grid_connected_same_production",
        scenario=scenario,
        wind_scenario=wind_scenario,
        pv_scenario=pv_scenario,
    )
    summary["target_ammonia_t"] = target_ammonia_t
    return hourly, summary


def run_no_storage_cases(params: SystemParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = add_regular_load_to_scenarios(params)
    hourly_parts = []
    records = []
    for (scenario, wind_id, pv_id), group in scenarios.groupby(["scenario", "wind_scenario", "pv_scenario"], sort=True):
        hourly, record = solve_no_storage_offgrid(group, params, str(scenario), int(wind_id), int(pv_id))
        hourly_parts.append(hourly)
        records.append(record)
    return pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(records)


def scan_storage_capacity(
    base_hourly: pd.DataFrame,
    params: SystemParams,
    scenario: str,
    wind_scenario: int,
    pv_scenario: int,
) -> pd.DataFrame:
    capacities = np.arange(0.0, CAPACITY_SCAN_MAX_MWH + 0.1, CAPACITY_SCAN_STEP_MWH)
    records = []
    for capacity in capacities:
        _, record = solve_storage_offgrid(base_hourly, params, float(capacity), scenario, wind_scenario, pv_scenario)
        records.append(record)
    scan = pd.DataFrame(records)
    scan["ammonia_gain_t"] = scan["日产氨量"] - float(scan.loc[scan["storage_capacity_mwh"] == 0.0, "日产氨量"].iloc[0])
    scan["curtailment_reduction_mwh"] = float(scan.loc[scan["storage_capacity_mwh"] == 0.0, "弃风弃光电量"].iloc[0]) - scan["弃风弃光电量"]
    return scan


def choose_storage_capacity(scan: pd.DataFrame) -> tuple[float, pd.Series, str]:
    base_ammonia = float(scan.loc[scan["storage_capacity_mwh"] == 0.0, "日产氨量"].iloc[0])
    positive = scan[(scan["storage_capacity_mwh"] > 0) & (scan["日产氨量"] > base_ammonia + 1e-5)].copy()
    if positive.empty:
        row = scan.loc[scan["吨氨成本"].idxmin()]
        return float(row["storage_capacity_mwh"]), row, "未找到可提升日产氨量的正储能容量，按全扫描吨氨成本最小选择。"
    row = positive.loc[positive["吨氨成本"].idxmin()]
    return float(row["storage_capacity_mwh"]), row, "在能够提升无储能日产氨量的正储能容量中，选择吨氨成本最低方案。"


def build_storage_pareto_summary(scan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    thresholds = [0.20, 0.15, 0.10, 0.05, 0.01, 0.0]
    for eps in thresholds:
        if eps == 0.0:
            feasible = scan[scan["弃风弃光率"] <= 1e-9].copy()
            label = "零弃电"
        else:
            feasible = scan[scan["弃风弃光率"] <= eps].copy()
            label = f"弃电率不超过{eps:.0%}"
        if feasible.empty:
            continue
        row = feasible.loc[feasible["吨氨成本"].idxmin()]
        rows.append(
            {
                "scheme": label,
                "epsilon_curtailment_rate": eps,
                "storage_capacity_mwh": float(row["storage_capacity_mwh"]),
                "storage_power_mw": float(row["storage_power_mw"]),
                "daily_ammonia_t": float(row["日产氨量"]),
                "curtailment_mwh": float(row["弃风弃光电量"]),
                "curtailment_rate": float(row["弃风弃光率"]),
                "renewable_utilization": float(row["风光利用率"]),
                "ton_cost_yuan_per_t": float(row["吨氨成本"]),
            }
        )

    for label, idx in [
        ("经济最优正储能", scan[scan["storage_capacity_mwh"] > 0]["吨氨成本"].idxmin()),
        ("最大日产氨/消纳优先", scan["日产氨量"].idxmax()),
    ]:
        row = scan.loc[idx]
        rows.append(
            {
                "scheme": label,
                "epsilon_curtailment_rate": np.nan,
                "storage_capacity_mwh": float(row["storage_capacity_mwh"]),
                "storage_power_mw": float(row["storage_power_mw"]),
                "daily_ammonia_t": float(row["日产氨量"]),
                "curtailment_mwh": float(row["弃风弃光电量"]),
                "curtailment_rate": float(row["弃风弃光率"]),
                "renewable_utilization": float(row["风光利用率"]),
                "ton_cost_yuan_per_t": float(row["吨氨成本"]),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["scheme"], keep="first")


def run_storage_cases(params: SystemParams, storage_capacity_mwh: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = add_regular_load_to_scenarios(params)
    hourly_parts = []
    records = []
    for (scenario, wind_id, pv_id), group in scenarios.groupby(["scenario", "wind_scenario", "pv_scenario"], sort=True):
        hourly, record = solve_storage_offgrid(group, params, storage_capacity_mwh, str(scenario), int(wind_id), int(pv_id))
        hourly_parts.append(hourly)
        records.append(record)
    return pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(records)


def run_grid_same_production_cases(
    params: SystemParams,
    storage_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = add_regular_load_to_scenarios(params)
    targets = storage_summary.set_index("scenario")["日产氨量"].to_dict()
    hourly_parts = []
    records = []
    for (scenario, wind_id, pv_id), group in scenarios.groupby(["scenario", "wind_scenario", "pv_scenario"], sort=True):
        target = float(targets[str(scenario)])
        hourly, record = solve_grid_connected_same_production(group, params, target, str(scenario), int(wind_id), int(pv_id))
        hourly_parts.append(hourly)
        records.append(record)
    return pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(records)


def build_offgrid_annual_summary(summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    total_cost = float((summary["日总成本"] * SCENARIO_DAYS).sum())
    ammonia = float((summary["日产氨量"] * SCENARIO_DAYS).sum())
    renewable = float((summary["新能源可发电量"] * SCENARIO_DAYS).sum())
    curtailment = float((summary["弃风弃光电量"] * SCENARIO_DAYS).sum())
    shed = float((summary["常规负荷缺供电量"] * SCENARIO_DAYS).sum())
    regular = float((summary["常规负荷需求电量"] * SCENARIO_DAYS).sum())
    served_load = float((summary["离网服务负荷电量"] * SCENARIO_DAYS).sum())
    demand_load = float((summary["离网需求电量"] * SCENARIO_DAYS).sum())
    storage_capacity = float(summary["storage_capacity_mwh"].iloc[0])
    return pd.DataFrame(
        [
            {
                "mode": mode,
                "annual_days": int(len(summary) * SCENARIO_DAYS),
                "storage_capacity_mwh": storage_capacity,
                "storage_power_mw": storage_capacity * BATTERY_C_RATE,
                "annual_ammonia_t": ammonia,
                "annual_capacity_utilization": ammonia / (CAPACITY_DAILY_AMMONIA_T * len(summary) * SCENARIO_DAYS),
                "annual_total_cost_yuan": total_cost,
                "annual_average_ton_cost_yuan_per_t": total_cost / ammonia,
                "annual_renewable_mwh": renewable,
                "annual_curtailment_mwh": curtailment,
                "annual_renewable_utilization": (renewable - curtailment) / renewable,
                "annual_regular_shed_mwh": shed,
                "annual_regular_supply_ratio": (regular - shed) / regular,
                "annual_served_load_mwh": served_load,
                "annual_demand_load_mwh": demand_load,
                "annual_green_demand_supply_ratio": served_load / demand_load,
                "energy_autonomous_days": int((summary["常规负荷缺供电量"] <= 1e-7).sum() * SCENARIO_DAYS),
                "full_production_days": int((summary["日产氨量"] >= CAPACITY_DAILY_AMMONIA_T - 1e-7).sum() * SCENARIO_DAYS),
            }
        ]
    )


def build_grid_annual_summary(summary: pd.DataFrame) -> pd.DataFrame:
    total_cost = float((summary["日总成本（扣除售电收益后）"] * SCENARIO_DAYS).sum())
    ammonia = float((summary["日产氨量"] * SCENARIO_DAYS).sum())
    renewable = float((summary["新能源发电量"] * SCENARIO_DAYS).sum())
    purchase = float((summary["网购电量"] * SCENARIO_DAYS).sum())
    sale = float((summary["上网电量"] * SCENARIO_DAYS).sum())
    total_load = float((summary["典型日总用电量"] * SCENARIO_DAYS).sum())
    self_use = renewable - sale
    return pd.DataFrame(
        [
            {
                "mode": "grid_connected_same_production",
                "annual_days": int(len(summary) * SCENARIO_DAYS),
                "annual_ammonia_t": ammonia,
                "annual_capacity_utilization": ammonia / (CAPACITY_DAILY_AMMONIA_T * len(summary) * SCENARIO_DAYS),
                "annual_total_cost_yuan": total_cost,
                "annual_average_ton_cost_yuan_per_t": total_cost / ammonia,
                "annual_renewable_mwh": renewable,
                "annual_purchase_mwh": purchase,
                "annual_sale_mwh": sale,
                "annual_self_use_mwh": self_use,
                "annual_self_use_ratio": self_use / renewable,
                "annual_green_power_ratio": self_use / total_load,
                "annual_sale_ratio": sale / renewable,
            }
        ]
    )


def build_mode_comparison(
    no_storage_annual: pd.DataFrame,
    storage_annual: pd.DataFrame,
    grid_annual: pd.DataFrame,
    no_storage_summary: pd.DataFrame,
    storage_summary: pd.DataFrame,
    grid_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, annual, summary, cost_col in [
        ("离网无储能", no_storage_annual, no_storage_summary, "日总成本"),
        ("离网+储能", storage_annual, storage_summary, "日总成本"),
        ("并网同产量", grid_annual, grid_summary, "日总成本（扣除售电收益后）"),
    ]:
        row = annual.iloc[0].to_dict()
        row["display_mode"] = label
        row["annual_wind_cost_yuan"] = float((summary["风电成本"] * SCENARIO_DAYS).sum())
        row["annual_pv_cost_yuan"] = float((summary["光伏成本"] * SCENARIO_DAYS).sum())
        row["annual_alk_om_yuan"] = float((summary["碱性电解槽运维成本"] * SCENARIO_DAYS).sum())
        row["annual_pem_om_yuan"] = float((summary["PEM电解槽运维成本"] * SCENARIO_DAYS).sum())
        row["annual_ammonia_om_yuan"] = float((summary["合成氨装置运维成本"] * SCENARIO_DAYS).sum())
        row["annual_total_cost_check_yuan"] = float((summary[cost_col] * SCENARIO_DAYS).sum())
        if "储能年化投资成本" in summary.columns:
            row["annual_storage_capital_yuan"] = float((summary["储能年化投资成本"] * SCENARIO_DAYS).sum())
            row["annual_storage_om_yuan"] = float((summary["储能充放电运维成本"] * SCENARIO_DAYS).sum())
        else:
            row["annual_storage_capital_yuan"] = 0.0
            row["annual_storage_om_yuan"] = 0.0
        if "购电成本" in summary.columns:
            row["annual_purchase_cost_yuan"] = float((summary["购电成本"] * SCENARIO_DAYS).sum())
            row["annual_sale_revenue_yuan"] = float((summary["余电上网收益"] * SCENARIO_DAYS).sum())
        else:
            row["annual_purchase_cost_yuan"] = 0.0
            row["annual_sale_revenue_yuan"] = 0.0
        rows.append(row)
    comp = pd.DataFrame(rows)
    off = comp.loc[comp["display_mode"] == "离网+储能", "annual_total_cost_yuan"].iloc[0]
    grid = comp.loc[comp["display_mode"] == "并网同产量", "annual_total_cost_yuan"].iloc[0]
    ammonia = comp.loc[comp["display_mode"] == "离网+储能", "annual_ammonia_t"].iloc[0]
    comp["grid_support_value_total_yuan"] = off - grid
    comp["grid_support_value_yuan_per_t"] = (off - grid) / ammonia
    return comp


def estimate_min_capacity_for_full_autonomy(params: SystemParams) -> pd.DataFrame:
    scenarios = add_regular_load_to_scenarios(params)
    full_process_mw = params.process_power_mw(CAPACITY_DAILY_AMMONIA_T)
    demand = scenarios["regular_load_mw"].to_numpy() + full_process_mw
    a_ub = -scenarios[["wind_pu", "pv_pu"]].to_numpy()
    b_ub = -demand
    result = linprog(c=np.array([1.0, 1.0]), A_ub=a_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method="highs")
    if not result.success:
        raise RuntimeError(f"Minimum capacity LP failed: {result.message}")
    wind_cap, pv_cap = map(float, result.x)
    min_margin = float((scenarios["wind_pu"] * wind_cap + scenarios["pv_pu"] * pv_cap - demand).min())
    current_mix_supply = scenarios["wind_mw"].to_numpy() + scenarios["pv_mw"].to_numpy()
    fixed_ratio_scale = float(np.max(demand / current_mix_supply))
    fixed_wind_cap = params.wind_capacity_mw * fixed_ratio_scale
    fixed_pv_cap = params.pv_capacity_mw * fixed_ratio_scale
    fixed_margin = float((current_mix_supply * fixed_ratio_scale - demand).min())
    return pd.DataFrame(
        [
            {
                "target": "min_total_capacity_unconstrained_mix",
                "description": "不固定风光比例、以总装机最小为目标",
                "wind_capacity_mw": wind_cap,
                "pv_capacity_mw": pv_cap,
                "total_capacity_mw": wind_cap + pv_cap,
                "wind_capacity_multiple": wind_cap / params.wind_capacity_mw,
                "pv_capacity_multiple": pv_cap / params.pv_capacity_mw if params.pv_capacity_mw > 0 else np.nan,
                "total_capacity_multiple_vs_current": (wind_cap + pv_cap) / (params.wind_capacity_mw + params.pv_capacity_mw),
                "minimum_hourly_margin_mw": min_margin,
            },
            {
                "target": "fixed_current_wind_pv_ratio",
                "description": "保持当前40MW风电与64MW光伏比例",
                "wind_capacity_mw": fixed_wind_cap,
                "pv_capacity_mw": fixed_pv_cap,
                "total_capacity_mw": fixed_wind_cap + fixed_pv_cap,
                "wind_capacity_multiple": fixed_ratio_scale,
                "pv_capacity_multiple": fixed_ratio_scale,
                "total_capacity_multiple_vs_current": fixed_ratio_scale,
                "minimum_hourly_margin_mw": fixed_margin,
            },
        ]
    )


def plot_no_storage_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    metrics = [
        ("日产氨量", "日产氨量/t", "YlGnBu", "{:.1f}"),
        ("弃风弃光电量", "弃风弃光/MWh", "YlOrRd", "{:.0f}"),
        ("常规负荷缺供电量", "常规缺供/MWh", "Reds", "{:.1f}"),
        ("风光利用率", "风光利用率/%", "YlGn", "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True, sharey=True)
    for ax, (metric, title, cmap, fmt) in zip(axes.ravel(), metrics):
        matrix = summary.pivot_table(index="wind_scenario", columns="pv_scenario", values=metric, aggfunc="first").sort_index()
        values = matrix.values * 100 if "率" in metric else matrix.values
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([str(int(c)) for c in matrix.columns])
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([str(int(i)) for i in matrix.index])
        vmax = np.nanmax(values)
        vmin = np.nanmin(values)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = values[i, j]
                color = "white" if vmax > vmin and val > vmin + 0.62 * (vmax - vmin) else "#222222"
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6.6, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.018)
        cbar.ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("风电场景")
    for ax in axes[-1, :]:
        ax.set_xlabel("光伏场景")
    fig.suptitle("离网无储能下24种风光场景运行结果", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.09, wspace=0.22, hspace=0.28)
    save_figure(fig, output_dir / "q4_fig1_no_storage_heatmaps_csee")
    plt.close(fig)


def plot_storage_capacity_scan(scan: pd.DataFrame, selected_capacity: float, output_dir: Path) -> None:
    set_csee_style()
    df = scan.sort_values("storage_capacity_mwh")
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
    ax = axes[0]
    ax.plot(df["storage_capacity_mwh"], df["日产氨量"], color=CSEE_COLORS["green"], marker="o", label="日产氨量")
    ax.set_ylabel("日产氨量/t")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2 = ax.twinx()
    ax2.plot(df["storage_capacity_mwh"], df["弃风弃光电量"], color=CSEE_COLORS["red"], marker="s", label="弃风弃光")
    ax2.set_ylabel("弃风弃光/MWh")
    ax.axvline(selected_capacity, color="black", linestyle="--", linewidth=0.9)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    figure_legend_below(fig, handles1 + handles2, labels1 + labels2, ncol=2, y=0.49)

    ax = axes[1]
    ax.plot(df["storage_capacity_mwh"], df["吨氨成本"], color=CSEE_COLORS["blue"], marker="^", label="吨氨成本")
    ax.axvline(selected_capacity, color="black", linestyle="--", linewidth=0.9, label="选定容量")
    ax.set_xlabel("储能容量/MWh（功率按1C配置）")
    ax.set_ylabel("吨氨成本/(元/t)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("最大弃电场景下储能容量扫描结果", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.16, hspace=0.42)
    save_figure(fig, output_dir / "q4_fig2_storage_capacity_scan_csee")
    plt.close(fig)


def plot_storage_pareto(scan: pd.DataFrame, pareto: pd.DataFrame, selected_capacity: float, output_dir: Path) -> None:
    set_csee_style()
    df = scan.sort_values("storage_capacity_mwh")
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    sc = ax1.scatter(
        df["弃风弃光率"] * 100,
        df["吨氨成本"],
        c=df["storage_capacity_mwh"],
        cmap="YlGnBu",
        s=34,
        edgecolor="#444444",
        linewidth=0.35,
        label="容量扫描点",
    )
    selected = df.loc[df["storage_capacity_mwh"].eq(selected_capacity)].iloc[0]
    ax1.scatter(
        [selected["弃风弃光率"] * 100],
        [selected["吨氨成本"]],
        s=70,
        marker="*",
        color=CSEE_COLORS["red"],
        edgecolor="black",
        linewidth=0.5,
        label="经济方案",
        zorder=4,
    )
    zero = df.loc[df["弃风弃光电量"].idxmin()]
    ax1.scatter(
        [zero["弃风弃光率"] * 100],
        [zero["吨氨成本"]],
        s=64,
        marker="D",
        color=CSEE_COLORS["green"],
        edgecolor="black",
        linewidth=0.5,
        label="消纳优先",
        zorder=4,
    )
    label_offsets = {
        "弃电率不超过20%": (-8, 14, "center"),
        "弃电率不超过15%": (-6, 18, "center"),
        "弃电率不超过10%": (0, 18, "center"),
        "弃电率不超过5%": (0, 22, "center"),
        "弃电率不超过1%": (22, 36, "left"),
        "零弃电": (10, 8, "left"),
    }
    for _, row in pareto[pareto["scheme"].str.contains("弃电率不超过|零弃电", regex=True)].iterrows():
        label = row["scheme"].replace("弃电率不超过", "≤")
        dx, dy, ha = label_offsets.get(row["scheme"], (0, 16, "center"))
        ax1.annotate(
            label,
            xy=(row["curtailment_rate"] * 100, row["ton_cost_yuan_per_t"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.45, "shrinkA": 0, "shrinkB": 4},
        )
    ax1.set_xlabel("弃风弃光率/%")
    ax1.set_ylabel("吨氨成本/(元/t)")
    ax1.set_title("最大弃电场景储能容量的成本-消纳权衡")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.035, pad=0.02)
    cbar.set_label("储能容量/MWh")
    fig.subplots_adjust(left=0.11, right=0.92, top=0.88, bottom=0.24)
    save_figure(fig, output_dir / "q4_fig6_storage_pareto_csee")
    plt.close(fig)


def plot_selected_storage_dispatch(hourly: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    df = hourly.sort_values("hour")
    x = df["hour"]
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0, 0.85]})

    ax = axes[0]
    ax.stackplot(x, df["wind_mw"], df["pv_mw"], colors=[CSEE_COLORS["light_blue"], "#f7c97f"], labels=["风电", "光伏"], alpha=0.82)
    ax.plot(x, df["regular_load_mw"] + df["process_load_mw"], color=CSEE_COLORS["red"], marker="o", label="总用电")
    ax.plot(x, df["process_load_mw"], color=CSEE_COLORS["purple"], marker="s", label="制氨负荷")
    ax.set_ylabel("功率/MW")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))

    ax = axes[1]
    ax.bar(x - 0.18, df["storage_charge_mw"], width=0.34, color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.5, label="充电")
    ax.bar(x + 0.18, -df["storage_discharge_mw"], width=0.34, color=CSEE_COLORS["light_red"], edgecolor=CSEE_COLORS["red"], linewidth=0.5, label="放电")
    ax.plot(x, df["curtailment_mw"], color=CSEE_COLORS["orange"], marker="^", label="弃电")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("功率/MW")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    ax = axes[2]
    ax.plot(x, df["storage_soc_mwh"], color=CSEE_COLORS["green"], marker="o", label="SOC")
    ax2 = ax.twinx()
    ax2.plot(x, df["production_load_factor"] * 100, color=CSEE_COLORS["purple"], linestyle="--", marker="s", label="负荷率")
    ax.set_xlabel("时段/h")
    ax.set_ylabel("SOC/MWh")
    ax2.set_ylabel("制氨负荷率/%")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.4, 23.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))

    scenario = str(df["scenario"].iloc[0])
    cap = float(df["storage_capacity_mwh"].iloc[0])
    fig.suptitle(f"最大弃电场景{scenario}的储能调度结果（{cap:.0f} MWh）", y=0.99, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.91, bottom=0.12, hspace=0.42)
    save_figure(fig, output_dir / "q4_fig3_selected_storage_dispatch_csee")
    plt.close(fig)


def plot_no_storage_vs_storage(no_storage: pd.DataFrame, storage: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    left = no_storage[["scenario", "日产氨量", "弃风弃光电量", "吨氨成本"]].rename(
        columns={"日产氨量": "no_storage_ammonia", "弃风弃光电量": "no_storage_curtailment", "吨氨成本": "no_storage_ton_cost"}
    )
    right = storage[["scenario", "日产氨量", "弃风弃光电量", "吨氨成本"]].rename(
        columns={"日产氨量": "storage_ammonia", "弃风弃光电量": "storage_curtailment", "吨氨成本": "storage_ton_cost"}
    )
    df = left.merge(right, on="scenario").sort_values("scenario")
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.0), sharex=True)
    width = 0.36
    axes[0].bar(x - width / 2, df["no_storage_ammonia"], width, color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.5, label="无储能")
    axes[0].bar(x + width / 2, df["storage_ammonia"], width, color="#f7c97f", edgecolor=CSEE_COLORS["orange"], linewidth=0.5, label="有储能")
    axes[0].set_ylabel("日产氨量/t")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))

    axes[1].bar(x - width / 2, df["no_storage_curtailment"], width, color=CSEE_COLORS["light_blue"], edgecolor=CSEE_COLORS["blue"], linewidth=0.5, label="无储能")
    axes[1].bar(x + width / 2, df["storage_curtailment"], width, color="#f7c97f", edgecolor=CSEE_COLORS["orange"], linewidth=0.5, label="有储能")
    axes[1].set_ylabel("弃风弃光/MWh")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[2].plot(x, df["no_storage_ton_cost"], color=CSEE_COLORS["blue"], marker="o", linestyle="--", label="无储能")
    axes[2].plot(x, df["storage_ton_cost"], color=CSEE_COLORS["orange"], marker="s", label="有储能")
    axes[2].set_ylabel("吨氨成本/(元/t)")
    axes[2].set_xlabel("风光组合场景")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["scenario"], rotation=45, ha="right")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.36))
    fig.suptitle("离网无储能与有储能运行结果对比", y=0.99, fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.24, hspace=0.35)
    save_figure(fig, output_dir / "q4_fig4_no_storage_vs_storage_csee")
    plt.close(fig)


def plot_mode_cost_breakdown(mode_compare: pd.DataFrame, output_dir: Path) -> None:
    set_csee_style()
    df = mode_compare.copy()
    labels = df["display_mode"].tolist()
    x = np.arange(len(df))
    components = [
        ("annual_wind_cost_yuan", "风电", CSEE_COLORS["light_blue"]),
        ("annual_pv_cost_yuan", "光伏", "#f7c97f"),
        ("annual_purchase_cost_yuan", "购电", CSEE_COLORS["light_red"]),
        ("annual_alk_om_yuan", "ALK运维", CSEE_COLORS["purple"]),
        ("annual_pem_om_yuan", "PEM运维", CSEE_COLORS["blue"]),
        ("annual_ammonia_om_yuan", "合成氨运维", CSEE_COLORS["gray"]),
        ("annual_storage_capital_yuan", "储能年化", "#b7b7b7"),
        ("annual_storage_om_yuan", "储能运维", CSEE_COLORS["green"]),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    bottom = np.zeros(len(df))
    for col, name, color in components:
        values = df[col].fillna(0).to_numpy() / 10000
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="black", linewidth=0.35, width=0.58, label=name)
        bottom += values
    sale = df["annual_sale_revenue_yuan"].fillna(0).to_numpy() / 10000
    ax.bar(x, -sale, color="#d9ead3", edgecolor=CSEE_COLORS["green"], linewidth=0.35, width=0.58, label="售电收益")
    ax.axhline(0, color="black", linewidth=0.8)
    positive_limit = max(float(np.max(bottom)) * 1.14, 1.0)
    negative_limit = min(float(np.min(-sale)) * 1.18, -1.0) if np.max(sale) > 0 else -1.0
    ax.set_ylim(negative_limit, positive_limit)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("年度成本/万元")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2 = ax.twinx()
    ax2.plot(x, df["annual_average_ton_cost_yuan_per_t"], color=CSEE_COLORS["red"], marker="o", label="吨氨成本")
    ax2.set_ylabel("吨氨成本/(元/t)")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    figure_legend_below(fig, handles1 + handles2, labels1 + labels2, ncol=5, y=0.01)
    ax.set_title("离网与并网同产量方案年度成本分项对比")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.25)
    save_figure(fig, output_dir / "q4_fig5_mode_cost_breakdown_csee")
    plt.close(fig)


def write_markdown_summary(
    no_storage_summary: pd.DataFrame,
    no_storage_annual: pd.DataFrame,
    capacity_scan: pd.DataFrame,
    pareto_summary: pd.DataFrame,
    selected_capacity: float,
    selected_row: pd.Series,
    selection_note: str,
    storage_summary: pd.DataFrame,
    storage_annual: pd.DataFrame,
    grid_annual: pd.DataFrame,
    mode_compare: pd.DataFrame,
    capacity_estimate: pd.DataFrame,
) -> None:
    max_curtail = no_storage_summary.loc[no_storage_summary["弃风弃光电量"].idxmax()]
    no_ann = no_storage_annual.iloc[0]
    st_ann = storage_annual.iloc[0]
    grid_ann = grid_annual.iloc[0]
    support_value = float(mode_compare["grid_support_value_yuan_per_t"].iloc[0])
    utilization_max_row = capacity_scan.loc[capacity_scan["日产氨量"].idxmax()]

    lines = [
        "# 问题四结果摘要",
        "",
        "## 离网无储能运行",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 年产氨量/t | {no_ann['annual_ammonia_t']:,.2f} |",
        f"| 年产能利用率 | {no_ann['annual_capacity_utilization']:.2%} |",
        f"| 年均吨氨成本/(元/t) | {no_ann['annual_average_ton_cost_yuan_per_t']:,.2f} |",
        f"| 年弃风弃光电量/MWh | {no_ann['annual_curtailment_mwh']:,.2f} |",
        f"| 年风光利用率 | {no_ann['annual_renewable_utilization']:.2%} |",
        f"| 年常规负荷缺供电量/MWh | {no_ann['annual_regular_shed_mwh']:,.2f} |",
        f"| 常规负荷完全自给代表天数/d | {int(no_ann['energy_autonomous_days'])} |",
        "",
        f"无储能时弃风弃光最大的场景为 **{max_curtail['scenario']}**，日弃电量为 **{max_curtail['弃风弃光电量']:,.2f} MWh**，日产氨量为 **{max_curtail['日产氨量']:,.2f} t**。",
        "",
        "## 储能容量配置",
        "",
        f"容量选择规则：{selection_note}",
        "",
        "| 储能容量/MWh | 储能功率/MW | 日产氨量/t | 弃风弃光/MWh | 风光利用率 | 吨氨成本/(元/t) |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {selected_capacity:.2f} | {selected_capacity * BATTERY_C_RATE:.2f} | {selected_row['日产氨量']:,.2f} | {selected_row['弃风弃光电量']:,.2f} | {selected_row['风光利用率']:.2%} | {selected_row['吨氨成本']:,.2f} |",
        "",
        "作为工程对照，若以最大化风光消纳和日产氨量为主目标，容量扫描中 **{cap:.2f} MWh** 可将该场景弃风弃光降至 **{cur:.2f} MWh**，日产氨量提高至 **{nh3:.2f} t**，但吨氨成本升至 **{cost:,.2f} 元/t**，因此未作为主经济方案。".format(
            cap=utilization_max_row["storage_capacity_mwh"],
            cur=utilization_max_row["弃风弃光电量"],
            nh3=utilization_max_row["日产氨量"],
            cost=utilization_max_row["吨氨成本"],
        ),
        "",
        "### 经济-消纳权衡方案",
        "",
        "| 方案 | 储能容量/MWh | 日产氨量/t | 弃风弃光率 | 风光利用率 | 吨氨成本/(元/t) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in pareto_summary.iterrows():
        lines.append(
            "| {scheme} | {cap:.2f} | {nh3:.2f} | {cur:.2%} | {use:.2%} | {cost:,.2f} |".format(
                scheme=row["scheme"],
                cap=row["storage_capacity_mwh"],
                nh3=row["daily_ammonia_t"],
                cur=row["curtailment_rate"],
                use=row["renewable_utilization"],
                cost=row["ton_cost_yuan_per_t"],
            )
        )
    lines.extend(
        [
        "",
        "## 24场景储能调度年度结果",
        "",
        "| 指标 | 无储能 | 有储能 |",
        "|---|---:|---:|",
        f"| 年产氨量/t | {no_ann['annual_ammonia_t']:,.2f} | {st_ann['annual_ammonia_t']:,.2f} |",
        f"| 年产能利用率 | {no_ann['annual_capacity_utilization']:.2%} | {st_ann['annual_capacity_utilization']:.2%} |",
        f"| 年均吨氨成本/(元/t) | {no_ann['annual_average_ton_cost_yuan_per_t']:,.2f} | {st_ann['annual_average_ton_cost_yuan_per_t']:,.2f} |",
        f"| 年弃风弃光/MWh | {no_ann['annual_curtailment_mwh']:,.2f} | {st_ann['annual_curtailment_mwh']:,.2f} |",
        f"| 年风光利用率 | {no_ann['annual_renewable_utilization']:.2%} | {st_ann['annual_renewable_utilization']:.2%} |",
        f"| 年常规负荷缺供/MWh | {no_ann['annual_regular_shed_mwh']:,.2f} | {st_ann['annual_regular_shed_mwh']:,.2f} |",
        "",
        "## 并网同产量对比",
        "",
        "| 模式 | 年产氨量/t | 年总成本/万元 | 年均吨氨成本/(元/t) |",
        "|---|---:|---:|---:|",
    ]
    )
    for _, row in mode_compare.iterrows():
        lines.append(
            f"| {row['display_mode']} | {row['annual_ammonia_t']:,.2f} | {row['annual_total_cost_yuan'] / 10000:,.2f} | {row['annual_average_ton_cost_yuan_per_t']:,.2f} |"
        )
    lines.extend(
        [
            "",
            f"以离网+储能方案的年产氨量为同产量约束，公共电网支撑价值为 **{support_value:,.2f} 元/t**（离网+储能成本减并网同产量成本）。",
            "",
            "## 无储能满产能源自治装机估算",
            "",
            "| 估算口径 | 风电容量/MW | 光伏容量/MW | 总容量/MW | 相对当前总容量倍数 | 最小时段裕度/MW |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, cap in capacity_estimate.iterrows():
        lines.append(
            f"| {cap['description']} | {cap['wind_capacity_mw']:,.2f} | {cap['pv_capacity_mw']:,.2f} | {cap['total_capacity_mw']:,.2f} | {cap['total_capacity_multiple_vs_current']:,.2f} | {cap['minimum_hourly_margin_mw']:.4f} |"
        )
    lines.extend(
        [
            "",
            "说明：储能容量扫描范围为0-300 MWh，步长5 MWh；储能功率按1C配置。无储能满产能源自治装机估算要求24个风光场景每个小时均可满足常规负荷和72 t/d满负荷制氨负荷，因此属于保守的瞬时自给容量下界估算。",
        ]
    )
    (OUT_DIR / "q4_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def validate_results(
    no_storage_hourly: pd.DataFrame,
    storage_hourly: pd.DataFrame,
    grid_hourly: pd.DataFrame,
    storage_capacity_mwh: float,
) -> pd.DataFrame:
    rows = [
        {
            "check": "no_storage_offgrid_power_balance_max_abs_mw",
            "value": float(no_storage_hourly["offgrid_balance_error_mw"].abs().max()),
        },
        {
            "check": "storage_offgrid_power_balance_max_abs_mw",
            "value": float(storage_hourly["offgrid_balance_error_mw"].abs().max()),
        },
        {
            "check": "storage_soc_min_mwh",
            "value": float(storage_hourly["storage_soc_mwh"].min()),
        },
        {
            "check": "storage_soc_max_mwh",
            "value": float(storage_hourly["storage_soc_mwh"].max()),
        },
        {
            "check": "storage_capacity_mwh",
            "value": float(storage_capacity_mwh),
        },
        {
            "check": "storage_charge_discharge_simultaneous_max_mw2",
            "value": float((storage_hourly["storage_charge_mw"] * storage_hourly["storage_discharge_mw"]).max()),
        },
        {
            "check": "grid_purchase_sale_simultaneous_max_mw2",
            "value": float((grid_hourly["grid_purchase_mw"] * grid_hourly["grid_sale_mw"]).max()),
        },
    ]
    cyclic_errors = []
    for _, group in storage_hourly.groupby("scenario", sort=False):
        group = group.sort_values("hour")
        soc = group["storage_soc_mwh"].to_numpy()
        ch = group["storage_charge_mw"].to_numpy()
        dis = group["storage_discharge_mw"].to_numpy()
        prev = np.roll(soc, 1)
        err = soc - (1 - SystemParams().storage_self_loss_ratio) * prev - SystemParams().storage_charge_eff * ch + dis / SystemParams().storage_discharge_eff
        cyclic_errors.append(np.max(np.abs(err)))
    rows.append({"check": "storage_soc_dynamic_max_abs_mwh", "value": float(np.max(cyclic_errors))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = SystemParams()

    no_storage_hourly, no_storage_summary = run_no_storage_cases(params)
    no_storage_annual = build_offgrid_annual_summary(no_storage_summary, "offgrid_no_storage")

    max_curtail = no_storage_summary.loc[no_storage_summary["弃风弃光电量"].idxmax()]
    scenarios = add_regular_load_to_scenarios(params)
    max_group = scenarios[scenarios["scenario"] == max_curtail["scenario"]].copy()
    capacity_scan = scan_storage_capacity(
        max_group,
        params,
        str(max_curtail["scenario"]),
        int(max_curtail["wind_scenario"]),
        int(max_curtail["pv_scenario"]),
    )
    selected_capacity, selected_row, selection_note = choose_storage_capacity(capacity_scan)
    pareto_summary = build_storage_pareto_summary(capacity_scan)

    storage_selected_hourly, _ = solve_storage_offgrid(
        max_group,
        params,
        selected_capacity,
        str(max_curtail["scenario"]),
        int(max_curtail["wind_scenario"]),
        int(max_curtail["pv_scenario"]),
    )
    storage_hourly, storage_summary = run_storage_cases(params, selected_capacity)
    storage_annual = build_offgrid_annual_summary(storage_summary, "offgrid_storage")

    grid_hourly, grid_summary = run_grid_same_production_cases(params, storage_summary)
    grid_annual = build_grid_annual_summary(grid_summary)
    mode_compare = build_mode_comparison(
        no_storage_annual,
        storage_annual,
        grid_annual,
        no_storage_summary,
        storage_summary,
        grid_summary,
    )
    capacity_estimate = estimate_min_capacity_for_full_autonomy(params)
    validation = validate_results(no_storage_hourly, storage_hourly, grid_hourly, selected_capacity)

    no_storage_hourly.to_csv(OUT_DIR / "q4_no_storage_hourly.csv", index=False, encoding="utf-8-sig")
    no_storage_summary.to_csv(OUT_DIR / "q4_no_storage_summary.csv", index=False, encoding="utf-8-sig")
    no_storage_annual.to_csv(OUT_DIR / "q4_no_storage_annual_summary.csv", index=False, encoding="utf-8-sig")
    capacity_scan.to_csv(OUT_DIR / "q4_storage_capacity_scan.csv", index=False, encoding="utf-8-sig")
    pareto_summary.to_csv(OUT_DIR / "q4_storage_pareto_summary.csv", index=False, encoding="utf-8-sig")
    storage_selected_hourly.to_csv(OUT_DIR / "q4_storage_selected_hourly.csv", index=False, encoding="utf-8-sig")
    storage_hourly.to_csv(OUT_DIR / "q4_storage_all_scenario_hourly.csv", index=False, encoding="utf-8-sig")
    storage_summary.to_csv(OUT_DIR / "q4_storage_all_scenario_summary.csv", index=False, encoding="utf-8-sig")
    storage_annual.to_csv(OUT_DIR / "q4_storage_annual_summary.csv", index=False, encoding="utf-8-sig")
    grid_hourly.to_csv(OUT_DIR / "q4_grid_comparison_hourly.csv", index=False, encoding="utf-8-sig")
    grid_summary.to_csv(OUT_DIR / "q4_grid_comparison_summary.csv", index=False, encoding="utf-8-sig")
    grid_annual.to_csv(OUT_DIR / "q4_grid_annual_summary.csv", index=False, encoding="utf-8-sig")
    mode_compare.to_csv(OUT_DIR / "q4_mode_comparison_summary.csv", index=False, encoding="utf-8-sig")
    capacity_estimate.to_csv(OUT_DIR / "q4_min_capacity_estimate.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUT_DIR / "q4_validation_summary.csv", index=False, encoding="utf-8-sig")

    write_markdown_summary(
        no_storage_summary,
        no_storage_annual,
        capacity_scan,
        pareto_summary,
        selected_capacity,
        selected_row,
        selection_note,
        storage_summary,
        storage_annual,
        grid_annual,
        mode_compare,
        capacity_estimate,
    )

    plot_no_storage_heatmaps(no_storage_summary, OUT_DIR)
    plot_storage_capacity_scan(capacity_scan, selected_capacity, OUT_DIR)
    plot_storage_pareto(capacity_scan, pareto_summary, selected_capacity, OUT_DIR)
    plot_selected_storage_dispatch(storage_selected_hourly, OUT_DIR)
    plot_no_storage_vs_storage(no_storage_summary, storage_summary, OUT_DIR)
    plot_mode_cost_breakdown(mode_compare, OUT_DIR)

    print("Q4 no-storage annual summary:")
    print(no_storage_annual.to_string(index=False))
    print("\nMax curtailment scenario:")
    print(max_curtail[["scenario", "日产氨量", "弃风弃光电量", "风光利用率", "吨氨成本"]].to_string())
    print("\nSelected storage capacity:")
    print(f"{selected_capacity:.2f} MWh / {selected_capacity * BATTERY_C_RATE:.2f} MW")
    print(selected_row[["日产氨量", "弃风弃光电量", "风光利用率", "吨氨成本"]].to_string())
    print(f"Selection note: {selection_note}")
    print("\nQ4 storage annual summary:")
    print(storage_annual.to_string(index=False))
    print("\nGrid-connected same-production annual summary:")
    print(grid_annual.to_string(index=False))
    print("\nMode comparison:")
    print(
        mode_compare[
            [
                "display_mode",
                "annual_ammonia_t",
                "annual_total_cost_yuan",
                "annual_average_ton_cost_yuan_per_t",
                "grid_support_value_yuan_per_t",
            ]
        ].to_string(index=False)
    )
    print("\nMinimum capacity estimate:")
    print(capacity_estimate.to_string(index=False))
    print("\nValidation:")
    print(validation.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
