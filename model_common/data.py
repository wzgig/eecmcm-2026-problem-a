from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SystemParams:
    load_peak_mw: float = 6.0
    wind_capacity_mw: float = 40.0
    pv_capacity_mw: float = 64.0
    alk_power_mw_36t: float = 10.0
    pem_power_mw_36t: float = 10.0
    ammonia_power_mw_36t: float = 0.75
    ammonia_output_t_per_h_36t: float = 1.5
    alk_h2_kg_per_h_36t: float = 140.0
    pem_h2_kg_per_h_36t: float = 160.0
    wind_lcoe_yuan_per_kwh: float = 0.15
    pv_lcoe_yuan_per_kwh: float = 0.12
    alk_om_yuan_per_kwh: float = 0.10
    pem_om_yuan_per_kwh: float = 0.15
    ammonia_om_yuan_per_kwh: float = 0.002
    sell_price_yuan_per_kwh: float = 0.3779
    storage_capex_yuan_per_kwh: float = 1000.0
    storage_life_year: float = 15.0
    storage_charge_eff: float = 0.90
    storage_discharge_eff: float = 0.90
    storage_self_loss_ratio: float = 0.002

    def scale(self, daily_ammonia_t: float) -> float:
        return daily_ammonia_t / 36.0

    def process_power_mw(self, daily_ammonia_t: float = 36.0) -> float:
        k = self.scale(daily_ammonia_t)
        return (self.alk_power_mw_36t + self.pem_power_mw_36t + self.ammonia_power_mw_36t) * k

    def alk_power_mw(self, daily_ammonia_t: float = 36.0) -> float:
        return self.alk_power_mw_36t * self.scale(daily_ammonia_t)

    def pem_power_mw(self, daily_ammonia_t: float = 36.0) -> float:
        return self.pem_power_mw_36t * self.scale(daily_ammonia_t)

    def ammonia_power_mw(self, daily_ammonia_t: float = 36.0) -> float:
        return self.ammonia_power_mw_36t * self.scale(daily_ammonia_t)

    def ammonia_output_t_per_h(self, daily_ammonia_t: float = 36.0) -> float:
        return self.ammonia_output_t_per_h_36t * self.scale(daily_ammonia_t)


def find_input_file(keyword: str, root: Path = PROJECT_ROOT) -> Path:
    matches = sorted(root.glob(f"*{keyword}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"Cannot find Excel file containing keyword: {keyword}")
    return matches[0]


def normalize_hour_label(label: object) -> str:
    return str(label).strip().replace("：", ":")


def purchase_price(hour: int) -> float:
    if 10 <= hour < 15 or 18 <= hour < 21:
        return 0.8024
    if 7 <= hour < 10 or 15 <= hour < 18 or 21 <= hour < 23:
        return 0.6074
    return 0.3424


def load_typical_day(params: SystemParams | None = None, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    params = params or SystemParams()
    load_raw = pd.read_excel(find_input_file("附件1", root))
    renew_raw = pd.read_excel(find_input_file("附件2", root))

    load_raw.columns = ["time", "load_pu"]
    renew_raw.columns = ["time", "wind_pu", "pv_pu"]

    df = pd.merge(load_raw, renew_raw, on="time", how="inner")
    if len(df) != 24:
        raise ValueError(f"Expected 24 hourly rows for typical day, got {len(df)}")

    df["hour"] = np.arange(24)
    df["time"] = df["time"].map(normalize_hour_label)
    df["regular_load_mw"] = df["load_pu"].astype(float) * params.load_peak_mw
    df["wind_mw"] = df["wind_pu"].astype(float) * params.wind_capacity_mw
    df["pv_mw"] = df["pv_pu"].astype(float) * params.pv_capacity_mw
    df["renewable_mw"] = df["wind_mw"] + df["pv_mw"]
    df["purchase_price_yuan_per_kwh"] = df["hour"].map(purchase_price)
    df["sell_price_yuan_per_kwh"] = params.sell_price_yuan_per_kwh
    return df


def load_wind_solar_scenarios(params: SystemParams | None = None, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    params = params or SystemParams()
    wind_raw = pd.read_excel(find_input_file("附件3", root))
    pv_raw = pd.read_excel(find_input_file("附件4", root))

    wind_raw = wind_raw.rename(columns={wind_raw.columns[0]: "time"})
    pv_raw = pv_raw.rename(columns={pv_raw.columns[0]: "time"})
    wind_raw["hour"] = np.arange(24)
    pv_raw["hour"] = np.arange(24)

    rows: list[dict[str, float | int | str]] = []
    wind_cols = [c for c in wind_raw.columns if str(c).startswith("风电场景")]
    pv_cols = [c for c in pv_raw.columns if str(c).startswith("光伏场景")]
    for wi, w_col in enumerate(wind_cols, 1):
        for pi, pv_col in enumerate(pv_cols, 1):
            for h in range(24):
                rows.append(
                    {
                        "scenario": f"W{wi}_PV{pi}",
                        "wind_scenario": wi,
                        "pv_scenario": pi,
                        "hour": h,
                        "time": normalize_hour_label(wind_raw.loc[h, "time"]),
                        "wind_pu": float(wind_raw.loc[h, w_col]),
                        "pv_pu": float(pv_raw.loc[h, pv_col]),
                        "wind_mw": float(wind_raw.loc[h, w_col]) * params.wind_capacity_mw,
                        "pv_mw": float(pv_raw.loc[h, pv_col]) * params.pv_capacity_mw,
                    }
                )
    df = pd.DataFrame(rows)
    df["renewable_mw"] = df["wind_mw"] + df["pv_mw"]
    df["purchase_price_yuan_per_kwh"] = df["hour"].map(purchase_price)
    df["sell_price_yuan_per_kwh"] = params.sell_price_yuan_per_kwh
    return df

