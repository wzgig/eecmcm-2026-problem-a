from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import SystemParams


@dataclass(frozen=True)
class IndicatorThresholds:
    min_self_use_ratio: float = 0.60
    min_green_power_ratio: float = 0.30
    max_sale_ratio: float = 0.20


def add_process_load(
    df: pd.DataFrame,
    params: SystemParams,
    daily_ammonia_t: float = 36.0,
    load_factor_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if load_factor_col is None:
        factor = 1.0
        out["process_load_factor"] = factor
    else:
        out["process_load_factor"] = out[load_factor_col].astype(float)
    out["alk_load_mw"] = params.alk_power_mw(daily_ammonia_t) * out["process_load_factor"]
    out["pem_load_mw"] = params.pem_power_mw(daily_ammonia_t) * out["process_load_factor"]
    out["ammonia_load_mw"] = params.ammonia_power_mw(daily_ammonia_t) * out["process_load_factor"]
    out["process_load_mw"] = out["alk_load_mw"] + out["pem_load_mw"] + out["ammonia_load_mw"]
    out["ammonia_output_t"] = params.ammonia_output_t_per_h(daily_ammonia_t) * out["process_load_factor"]
    out["total_load_mw"] = out["regular_load_mw"] + out["process_load_mw"]
    return out


def add_grid_exchange(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    net_deficit = out["total_load_mw"] - out["renewable_mw"]
    out["grid_purchase_mw"] = net_deficit.clip(lower=0)
    out["grid_sale_mw"] = (-net_deficit).clip(lower=0)
    out["grid_exchange_mw"] = out["grid_purchase_mw"] - out["grid_sale_mw"]
    return out


def mwh(df: pd.DataFrame, column: str) -> float:
    return float(df[column].sum())


def q1_summary(df: pd.DataFrame, params: SystemParams, thresholds: IndicatorThresholds | None = None) -> pd.DataFrame:
    thresholds = thresholds or IndicatorThresholds()

    total_load_mwh = mwh(df, "total_load_mw")
    regular_load_mwh = mwh(df, "regular_load_mw")
    process_load_mwh = mwh(df, "process_load_mw")
    wind_mwh = mwh(df, "wind_mw")
    pv_mwh = mwh(df, "pv_mw")
    renewable_mwh = mwh(df, "renewable_mw")
    purchase_mwh = mwh(df, "grid_purchase_mw")
    sale_mwh = mwh(df, "grid_sale_mw")
    ammonia_t = mwh(df, "ammonia_output_t")

    renewable_self_use_mwh = renewable_mwh - sale_mwh
    load_side_self_use_mwh = total_load_mwh - purchase_mwh
    statement_formula_self_use_mwh = total_load_mwh - sale_mwh - purchase_mwh

    self_use_ratio = renewable_self_use_mwh / renewable_mwh
    green_power_ratio = renewable_self_use_mwh / total_load_mwh
    sale_ratio = sale_mwh / renewable_mwh
    statement_formula_ratio = statement_formula_self_use_mwh / renewable_mwh

    wind_cost = wind_mwh * 1000 * params.wind_lcoe_yuan_per_kwh
    pv_cost = pv_mwh * 1000 * params.pv_lcoe_yuan_per_kwh
    purchase_cost = float((df["grid_purchase_mw"] * 1000 * df["purchase_price_yuan_per_kwh"]).sum())
    alk_om = float((df["alk_load_mw"] * 1000 * params.alk_om_yuan_per_kwh).sum())
    pem_om = float((df["pem_load_mw"] * 1000 * params.pem_om_yuan_per_kwh).sum())
    ammonia_om = float((df["ammonia_load_mw"] * 1000 * params.ammonia_om_yuan_per_kwh).sum())
    sale_revenue = sale_mwh * 1000 * params.sell_price_yuan_per_kwh
    gross_cost = wind_cost + pv_cost + purchase_cost + alk_om + pem_om + ammonia_om
    net_cost = gross_cost - sale_revenue
    cost_per_ton = net_cost / ammonia_t

    rows = [
        ("典型日总用电量", total_load_mwh, "MWh"),
        ("其中：常规电负荷用电量", regular_load_mwh, "MWh"),
        ("其中：制氢制氨工艺用电量", process_load_mwh, "MWh"),
        ("风电发电量", wind_mwh, "MWh"),
        ("光伏发电量", pv_mwh, "MWh"),
        ("新能源发电量", renewable_mwh, "MWh"),
        ("网购电量", purchase_mwh, "MWh"),
        ("上网电量", sale_mwh, "MWh"),
        ("新能源自发自用电量（物理/政策口径）", renewable_self_use_mwh, "MWh"),
        ("新能源自发自用电量（由负荷侧校验）", load_side_self_use_mwh, "MWh"),
        ("题面括号公式复核量：总用电量-上网电量-网购电量", statement_formula_self_use_mwh, "MWh"),
        ("新能源自发自用比例（物理/政策口径）", self_use_ratio, "ratio"),
        ("总用电量绿电比例", green_power_ratio, "ratio"),
        ("新能源上网电量比例", sale_ratio, "ratio"),
        ("题面括号公式比例复核", statement_formula_ratio, "ratio"),
        ("新能源自发自用比例是否达标", float(self_use_ratio >= thresholds.min_self_use_ratio), "bool"),
        ("总用电量绿电比例是否达标", float(green_power_ratio >= thresholds.min_green_power_ratio), "bool"),
        ("新能源上网电量比例是否达标", float(sale_ratio <= thresholds.max_sale_ratio), "bool"),
        ("日产氨量", ammonia_t, "t"),
        ("风电成本", wind_cost, "yuan"),
        ("光伏成本", pv_cost, "yuan"),
        ("购电成本", purchase_cost, "yuan"),
        ("碱性电解槽运维成本", alk_om, "yuan"),
        ("PEM电解槽运维成本", pem_om, "yuan"),
        ("合成氨装置运维成本", ammonia_om, "yuan"),
        ("余电上网收益", sale_revenue, "yuan"),
        ("日总成本（扣除售电收益后）", net_cost, "yuan"),
        ("吨氨成本", cost_per_ton, "yuan/t"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def get_metric(summary: pd.DataFrame, metric: str) -> float:
    return float(summary.loc[summary["metric"] == metric, "value"].iloc[0])


def format_metric(value: float, unit: str) -> str:
    if unit == "ratio":
        return f"{value:.4%}"
    if unit == "bool":
        return "是" if value >= 0.5 else "否"
    if unit in {"yuan", "yuan/t"}:
        return f"{value:,.2f}"
    return f"{value:,.4f}"

