from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_common.data import SystemParams, load_typical_day
from model_common.metrics import add_grid_exchange, add_process_load, format_metric, get_metric, q1_summary
from model_common.plotting import (
    plot_q1_components,
    plot_q1_cost_breakdown,
    plot_q1_indicator_bars,
    plot_q1_power_balance,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_q1_hourly_table(params: SystemParams) -> pd.DataFrame:
    hourly = load_typical_day(params=params, root=PROJECT_ROOT)
    hourly = add_process_load(hourly, params=params, daily_ammonia_t=36.0)
    hourly = add_grid_exchange(hourly)
    return hourly


def write_result_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# 问题一计算结果摘要",
        "",
        "| 指标 | 数值 | 单位 |",
        "|---|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row['metric']} | {format_metric(float(row['value']), row['unit'])} | {row['unit']} |")

    lines.extend(
        [
            "",
            "## 合格性判定（物理/政策口径）",
            "",
            f"- 新能源自发自用比例：{get_metric(summary, '新能源自发自用比例（物理/政策口径）'):.2%}，要求不低于60%。",
            f"- 总用电量绿电比例：{get_metric(summary, '总用电量绿电比例'):.2%}，要求不低于30%。",
            f"- 新能源上网电量比例：{get_metric(summary, '新能源上网电量比例'):.2%}，要求不高于20%。",
            "",
            "说明：本结果将新能源自发自用电量定义为新能源发电量中未上网、用于园区负荷的电量；同时输出题面括号公式复核量，便于论文中说明口径差异。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = SystemParams()
    hourly = build_q1_hourly_table(params)
    summary = q1_summary(hourly, params)

    hourly_columns = [
        "hour",
        "time",
        "load_pu",
        "wind_pu",
        "pv_pu",
        "regular_load_mw",
        "alk_load_mw",
        "pem_load_mw",
        "ammonia_load_mw",
        "process_load_mw",
        "total_load_mw",
        "wind_mw",
        "pv_mw",
        "renewable_mw",
        "grid_purchase_mw",
        "grid_sale_mw",
        "grid_exchange_mw",
        "purchase_price_yuan_per_kwh",
        "sell_price_yuan_per_kwh",
    ]
    hourly.loc[:, hourly_columns].to_csv(OUT_DIR / "q1_hourly_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "q1_summary.csv", index=False, encoding="utf-8-sig")
    write_result_summary(summary, OUT_DIR / "q1_result_summary.md")

    plot_q1_power_balance(hourly, OUT_DIR)
    plot_q1_components(hourly, OUT_DIR)
    plot_q1_indicator_bars(summary, OUT_DIR)
    plot_q1_cost_breakdown(summary, OUT_DIR)

    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
