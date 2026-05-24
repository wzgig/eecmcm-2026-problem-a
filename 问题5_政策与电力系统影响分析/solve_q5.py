from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_common.metrics import get_metric
from model_common.plotting import CSEE_COLORS, annotate_bars, legend_below, save_figure, set_csee_style


OUT_DIR = Path(__file__).resolve().parent / "outputs"
Q1_OUT = PROJECT_ROOT / "问题1_典型风光场景指标分析" / "outputs"
Q2_OUT = PROJECT_ROOT / "问题2_离散制氨调节优化" / "outputs"
Q3_OUT = PROJECT_ROOT / "问题3_连续制氨调节优化" / "outputs"
Q4_OUT = PROJECT_ROOT / "问题4_离网储能配置优化" / "outputs"


def pct(value: float) -> str:
    return f"{value:.2%}"


def load_prior_results() -> dict[str, pd.DataFrame]:
    return {
        "q1_summary": pd.read_csv(Q1_OUT / "q1_summary.csv"),
        "q2_annual": pd.read_csv(Q2_OUT / "q2_annual_summary.csv"),
        "q3_annual": pd.read_csv(Q3_OUT / "q3_annual_summary.csv"),
        "q3_compare": pd.read_csv(Q3_OUT / "q3_compare_q2_summary.csv"),
        "q4_no_storage_annual": pd.read_csv(Q4_OUT / "q4_no_storage_annual_summary.csv"),
        "q4_storage_annual": pd.read_csv(Q4_OUT / "q4_storage_annual_summary.csv"),
        "q4_grid_annual": pd.read_csv(Q4_OUT / "q4_grid_annual_summary.csv"),
        "q4_mode_compare": pd.read_csv(Q4_OUT / "q4_mode_comparison_summary.csv"),
        "q4_capacity": pd.read_csv(Q4_OUT / "q4_min_capacity_estimate.csv"),
    }


def build_model_evidence(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    q1 = results["q1_summary"]
    q2 = results["q2_annual"]
    q3 = results["q3_annual"]
    q3c = results["q3_compare"]
    q4_no = results["q4_no_storage_annual"].iloc[0]
    q4_st = results["q4_storage_annual"].iloc[0]
    q4_grid = results["q4_grid_annual"].iloc[0]
    q4_comp = results["q4_mode_compare"].iloc[0]
    q4_cap = results["q4_capacity"]

    q3_72 = q3.loc[q3["production_t_per_day"] == 72].iloc[0]
    q3_36 = q3.loc[q3["production_t_per_day"] == 36].iloc[0]
    q3c_36 = q3c.loc[q3c["production_t_per_day"] == 36].iloc[0]
    q2_36 = q2.loc[q2["production_t_per_day"] == 36].iloc[0]

    rows = [
        {
            "evidence_id": "E1",
            "source_problem": "Q1",
            "quantitative_result": "典型日上网比例35.92%，超过20%约束；同时购电172.04 MWh、上网216.77 MWh。",
            "system_impact": "源荷曲线不同步会同时形成购电缺口和反送电压力。",
            "policy_implication": "绿电直连项目不能只看年绿电量，应考核小时级源荷匹配和最大交换功率。",
        },
        {
            "evidence_id": "E2",
            "source_problem": "Q2",
            "quantitative_result": (
                f"离散开停36 t/d年均吨氨成本{q2_36['annual_average_ton_cost_yuan_per_t']:.2f}元/t最低，"
                f"但全年0天全满足；72 t/d全满足270天但成本{q3_72['annual_average_ton_cost_yuan_per_t']:.2f}元/t。"
            ),
            "system_impact": "低产量降低购电成本但推高余电上网比例，高产量提高消纳但增加用电和成本。",
            "policy_implication": "运行考核应同时约束成本、产能利用率和绿电指标，避免单纯追求最低吨成本。",
        },
        {
            "evidence_id": "E3",
            "source_problem": "Q3",
            "quantitative_result": (
                f"连续调节36 t/d较离散开停降本{abs(q3c_36['ton_cost_delta_yuan_per_t']):.2f}元/t，"
                f"购电减少{abs(q3c_36['purchase_delta_mwh']):.2f}MWh，上网减少{abs(q3c_36['sale_delta_mwh']):.2f}MWh，"
                f"全满足天数由0增至165天。"
            ),
            "system_impact": "电解槽和合成氨负荷连续调节可显著降低公共电网交换电量。",
            "policy_implication": "应把可调负荷能力作为绿电直连项目接入评审和运行考核指标。",
        },
        {
            "evidence_id": "E4",
            "source_problem": "Q3",
            "quantitative_result": (
                f"连续调节72 t/d年上网比例{pct(q3_72['annual_sale_ratio'])}，36 t/d上网比例{pct(q3_36['annual_sale_ratio'])}。"
            ),
            "system_impact": "负荷利用率下降会降低本地消纳能力并增加新能源外送压力。",
            "policy_implication": "新能源装机应按以荷定源原则滚动校核，并设置上网比例和消纳困难时段反送约束。",
        },
        {
            "evidence_id": "E5",
            "source_problem": "Q4",
            "quantitative_result": (
                f"离网无储能年产氨{q4_no['annual_ammonia_t']:.2f}t，产能利用率{pct(q4_no['annual_capacity_utilization'])}，"
                f"年弃电{q4_no['annual_curtailment_mwh']:.2f}MWh；5MWh储能后年产氨{q4_st['annual_ammonia_t']:.2f}t，"
                f"弃电降至{q4_st['annual_curtailment_mwh']:.2f}MWh且缺供为0。"
            ),
            "system_impact": "离网模式提高物理绿电属性，但对储能、冗余装机和运行安全要求更高。",
            "policy_implication": "离网项目应建立自给能力、缺供风险、弃电率和储能配置的专项评估。",
        },
        {
            "evidence_id": "E6",
            "source_problem": "Q4",
            "quantitative_result": (
                f"同产量下离网+储能吨氨成本{q4_st['annual_average_ton_cost_yuan_per_t']:.2f}元/t，"
                f"并网同产量{q4_grid['annual_average_ton_cost_yuan_per_t']:.2f}元/t，"
                f"系统支撑价值{q4_comp['grid_support_value_yuan_per_t']:.2f}元/t。"
            ),
            "system_impact": "公共电网提供备用、互济和消纳服务，降低园区自建储能和冗余容量成本。",
            "policy_implication": "并网型项目应公平承担输配电费、系统运行费、备用和辅助服务成本。",
        },
        {
            "evidence_id": "E7",
            "source_problem": "Q4",
            "quantitative_result": (
                "无储能满产逐小时自治若不固定风光比例需总装机"
                f"{q4_cap.iloc[0]['total_capacity_mw']:.2f}MW；保持当前风光比例需{q4_cap.iloc[1]['total_capacity_mw']:.2f}MW。"
            ),
            "system_impact": "完全离网满产自治需要极大冗余装机，经济性和土地资源约束突出。",
            "policy_implication": "应鼓励弱联网、源网荷储协同和市场化备用，而不是简单追求完全离网。",
        },
    ]
    return pd.DataFrame(rows)


def build_policy_matrix() -> pd.DataFrame:
    rows = [
        {
            "source_id": "P01",
            "source_name": "发改能源〔2026〕688号：多用户绿电直连",
            "local_file": "参考资料/01_政策与官方报告/P01_2026_多用户绿电直连发展通知_发改能源688号.html",
            "url": "https://www.ndrc.gov.cn/xxgk/zcfb/tz/202605/t20260520_1405313.html",
            "use_in_q5": "支持绿色氢氨醇开展绿电直连；提出以荷定源、自发自用比例、上网比例、小时级溯源、接网容量和系统费用分担。",
        },
        {
            "source_id": "P02",
            "source_name": "发改能源〔2025〕650号：绿电直连",
            "local_file": "参考资料/01_政策与官方报告/P02_2025_绿电直连发展通知_发改能源650号.html",
            "url": "https://www.nea.gov.cn/20250530/2d67a6e49c044f2eacabe1fddf48d20f/c.html",
            "use_in_q5": "三项绿电直连指标、源荷匹配、系统友好性、储能和负荷灵活性、分表计量与费用承担依据。",
        },
        {
            "source_id": "P03",
            "source_name": "发改价格〔2025〕136号：新能源上网电价市场化改革",
            "local_file": "参考资料/01_政策与官方报告/P03_2025_新能源上网电价市场化改革_发改价格136号.html",
            "url": "https://www.ndrc.gov.cn/xxgk/zcfb/tz/202502/t20250209_1396066.html",
            "use_in_q5": "余电上网价格、市场化交易和新能源消纳责任的价格背景。",
        },
        {
            "source_id": "P04",
            "source_name": "发改能源〔2024〕1537号：可再生能源替代行动",
            "local_file": "参考资料/01_政策与官方报告/P04_2024_可再生能源替代行动指导意见.html",
            "url": "https://www.ndrc.gov.cn/xxgk/zcfb/tz/202410/t20241030_1394119.html",
            "use_in_q5": "工业绿电直供、源网荷储、低碳氢替代、风光氢氨醇一体化基地和需求侧调节依据。",
        },
        {
            "source_id": "P05",
            "source_name": "国能发法改〔2024〕93号：电力领域新型经营主体",
            "local_file": "参考资料/01_政策与官方报告/P05_2024_电力领域新型经营主体创新发展指导意见.html",
            "url": "https://www.gov.cn/zhengce/zhengceku/202412/content_6991420.htm",
            "use_in_q5": "虚拟电厂、负荷聚合、智能微电网等新型主体参与调节和市场机制。",
        },
        {
            "source_id": "P06",
            "source_name": "国能综通科技〔2025〕91号：能源领域氢能试点",
            "local_file": "参考资料/01_政策与官方报告/P06_2025_能源领域氢能试点通知_国能综通科技91号.pdf",
            "url": "https://www.nea.gov.cn/20250610/472b12c43f534aab9a95de81034dcd92/20250610472b12c43f534aab9a95de81034dcd92_2288580504e06346d39eb8d268c8e70947.pdf",
            "use_in_q5": "柔性离网制氢、氢氨燃料供能、氢储能和示范试点依据。",
        },
        {
            "source_id": "P08",
            "source_name": "发改能源规〔2021〕280号：源网荷储一体化和多能互补",
            "local_file": "参考资料/01_政策与官方报告/P08_2021_源网荷储一体化和多能互补指导意见.html",
            "url": "https://www.gov.cn/zhengce/zhengceku/2021-03/06/content_5590895.htm",
            "use_in_q5": "园区级源网荷储、负荷侧调节、储能参与调峰和多能互补依据。",
        },
        {
            "source_id": "R01",
            "source_name": "IEA Global Hydrogen Review 2025",
            "local_file": "参考资料/01_政策与官方报告/R01_IEA_Global_Hydrogen_Review_2025.pdf",
            "url": "https://iea.blob.core.windows.net/assets/12d92ecc-e960-40f3-aff5-b2de6690ab6b/GlobalHydrogenReview2025.pdf",
            "use_in_q5": "低排放氢产业、项目落地、政策支持和需求创造的国际背景。",
        },
        {
            "source_id": "R02",
            "source_name": "IRENA Innovation Outlook: Renewable Ammonia",
            "local_file": "参考资料/01_政策与官方报告/R02_IRENA_Innovation_Outlook_Renewable_Ammonia_2022.pdf",
            "url": "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2022/May/IRENA_Innovation_Outlook_Ammonia_2022.pdf",
            "use_in_q5": "绿氨技术路线、可再生电力制氢制氨和成本构成依据。",
        },
        {
            "source_id": "R05",
            "source_name": "EU 2023/1184 RFNBO Delegated Regulation",
            "local_file": "参考资料/01_政策与官方报告/R05_EU_RFNBO_Delegated_Regulation_2023_1184.pdf",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32023R1184",
            "use_in_q5": "绿氢绿氨电力来源的额外性、时间匹配和地域匹配国际经验。",
        },
    ]
    return pd.DataFrame(rows)


def build_recommendation_matrix() -> pd.DataFrame:
    rows = [
        {
            "recommendation": "建立小时级绿电溯源与分表计量",
            "model_basis": "Q1显示同一典型日既购电172.04MWh又上网216.77MWh；Q3低产量方案上网比例可达35.76%。",
            "policy_basis": "P01要求小时级新能源发用电量匹配，P02要求分表计量和源荷匹配。",
            "expected_effect": "避免年累计绿电量掩盖小时错配，提高绿电认证可信度。",
        },
        {
            "recommendation": "将上网比例、最大交换功率和反送时段纳入运行考核",
            "model_basis": "Q1上网比例35.92%不达标，Q4并网同产量上网比例49.82%。",
            "policy_basis": "P01/P02均强调上网比例一般不超过20%，消纳困难时段不应反送。",
            "expected_effect": "降低局部潮流反转和新能源消纳压力。",
        },
        {
            "recommendation": "把柔性负荷能力作为准入和补偿对象",
            "model_basis": "Q3连续调节使36t/d方案较离散开停减少购电16216.93MWh、上网16216.93MWh，并增加165个全满足代表日。",
            "policy_basis": "P04、P08均强调需求侧资源和源网荷储互动。",
            "expected_effect": "提升园区对电网的友好性，减少外部调峰需求。",
        },
        {
            "recommendation": "建立储能配置的经济性和可靠性双目标评估",
            "model_basis": "Q4中5MWh储能为经济方案，155MWh可零弃电但吨氨成本更高。",
            "policy_basis": "P01/P02要求合理配置储能，P08强调优化储能规模和系统平衡能力。",
            "expected_effect": "避免过度储能或储能不足，兼顾消纳、可靠性和成本。",
        },
        {
            "recommendation": "完善系统运行费、备用和辅助服务费用分担",
            "model_basis": "Q4测算并网支撑价值为735.81元/t。",
            "policy_basis": "P01/P02要求并网型项目公平承担输配电费、系统运行费等费用。",
            "expected_effect": "防止成本向普通用户转移，反映公共电网支撑价值。",
        },
        {
            "recommendation": "推进风光氢氨醇一体化与弱联网示范",
            "model_basis": "Q4完全离网满产自治需极高冗余装机，弱联网更具经济性。",
            "policy_basis": "P01优先支持绿色氢氨醇，P04支持风光氢氨醇一体化基地，P06支持氢能试点。",
            "expected_effect": "在产业脱碳和电力系统安全之间取得更优平衡。",
        },
    ]
    return pd.DataFrame(rows)


def plot_q3_grid_exchange(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    set_csee_style()
    q3 = results["q3_annual"].sort_values("production_t_per_day")
    x = np.arange(len(q3))
    width = 0.34
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.1), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]})
    bars1 = ax1.bar(
        x - width / 2,
        q3["annual_purchase_mwh"],
        width,
        color=CSEE_COLORS["light_red"],
        edgecolor=CSEE_COLORS["red"],
        linewidth=0.5,
        label="年购电量",
    )
    bars2 = ax1.bar(
        x + width / 2,
        q3["annual_sale_mwh"],
        width,
        color=CSEE_COLORS["light_blue"],
        edgecolor=CSEE_COLORS["blue"],
        linewidth=0.5,
        label="年上网电量",
    )
    ax1.set_ylabel("年度电量/MWh")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))

    ax2.plot(x, q3["annual_sale_ratio"] * 100, color=CSEE_COLORS["red"], marker="o", label="上网比例")
    ax2.axhline(20, color=CSEE_COLORS["red"], linestyle=":", linewidth=0.9, label="20%约束")
    ax2.set_ylabel("上网比例/%")
    ax2.set_xlabel("日产量/t")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(v)}" for v in q3["production_t_per_day"]])
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax3 = ax2.twinx()
    bars3 = ax3.bar(
        x,
        q3["full_satisfied_days"],
        width=0.26,
        color="#d9ead3",
        edgecolor=CSEE_COLORS["green"],
        linewidth=0.5,
        alpha=0.75,
        label="全满足天数",
    )
    ax3.set_ylabel("全满足天数/d")
    ax3.set_ylim(0, 360)
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(handles2 + handles3, labels2 + labels3, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.suptitle("连续调节下日产量对电网交换和指标合格性的影响", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.91, bottom=0.20, hspace=0.35)
    save_figure(fig, output_dir / "q5_fig1_grid_exchange_policy_evidence_csee")
    plt.close(fig)


def plot_q4_grid_support(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    set_csee_style()
    comp = results["q4_mode_compare"].copy()
    x = np.arange(len(comp))
    labels = comp["display_mode"].tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(
        x,
        comp["annual_average_ton_cost_yuan_per_t"],
        color=[CSEE_COLORS["light_blue"], "#f7c97f", CSEE_COLORS["light_red"]],
        edgecolor=[CSEE_COLORS["blue"], CSEE_COLORS["orange"], CSEE_COLORS["red"]],
        linewidth=0.6,
        width=0.58,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("年均吨氨成本/(元/t)")
    ax.set_title("离网与并网同产量的系统支撑价值")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, comp["annual_average_ton_cost_yuan_per_t"].max() * 1.18)
    annotate_bars(ax, bars, fmt="{:.0f}", dy_frac=0.012)
    off_cost = float(comp.loc[comp["display_mode"] == "离网+储能", "annual_average_ton_cost_yuan_per_t"].iloc[0])
    grid_cost = float(comp.loc[comp["display_mode"] == "并网同产量", "annual_average_ton_cost_yuan_per_t"].iloc[0])
    support = off_cost - grid_cost
    ax.annotate(
        f"系统支撑价值：{support:.0f} 元/t",
        xy=(2, grid_cost),
        xytext=(1.25, off_cost * 0.72),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": CSEE_COLORS["gray"]},
        fontsize=8,
        color="#222222",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.16)
    save_figure(fig, output_dir / "q5_fig2_grid_support_value_csee")
    plt.close(fig)


def write_summary(
    evidence: pd.DataFrame,
    policy_matrix: pd.DataFrame,
    recommendations: pd.DataFrame,
    results: dict[str, pd.DataFrame],
) -> None:
    q4_comp = results["q4_mode_compare"]
    support_value = float(q4_comp["grid_support_value_yuan_per_t"].iloc[0])
    lines = [
        "# 问题五结果摘要",
        "",
        "## 前四问支撑结论",
        "",
        "| 编号 | 来源 | 数值证据 | 系统影响 | 政策含义 |",
        "|---|---|---|---|---|",
    ]
    for _, row in evidence.iterrows():
        lines.append(
            f"| {row['evidence_id']} | {row['source_problem']} | {row['quantitative_result']} | {row['system_impact']} | {row['policy_implication']} |"
        )
    lines.extend(
        [
            "",
            "## 绿电直连园区高渗透的主要影响",
            "",
            "有利影响：促进新能源就近消纳；推动化工行业绿氢绿氨替代；形成可调负荷和储能资源；带动源网荷储一体化投资。",
            "",
            "不利影响或风险：增加公共电网净负荷波动和反送压力；可能转移备用、调峰和系统运行成本；弱联网或离网项目对安全稳定、储能和冗余容量要求高；绿电溯源和多主体结算复杂度提高；电氢氨安全与电力安全耦合增强。",
            "",
            "## 政策建议",
            "",
            "| 建议 | 模型依据 | 政策依据 | 预期效果 |",
            "|---|---|---|---|",
        ]
    )
    for _, row in recommendations.iterrows():
        lines.append(
            f"| {row['recommendation']} | {row['model_basis']} | {row['policy_basis']} | {row['expected_effect']} |"
        )
    lines.extend(
        [
            "",
            f"本文将并网同产量相对离网+储能的成本差定义为公共电网支撑价值，本算例为 **{support_value:.2f} 元/t**。",
            "",
            "## 主要参考资料",
            "",
            "| 编号 | 资料 | 用途 | 链接 |",
            "|---|---|---|---|",
        ]
    )
    for _, row in policy_matrix.iterrows():
        lines.append(f"| {row['source_id']} | {row['source_name']} | {row['use_in_q5']} | {row['url']} |")
    (OUT_DIR / "q5_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_prior_results()
    evidence = build_model_evidence(results)
    policy_matrix = build_policy_matrix()
    recommendations = build_recommendation_matrix()

    evidence.to_csv(OUT_DIR / "q5_model_evidence_summary.csv", index=False, encoding="utf-8-sig")
    policy_matrix.to_csv(OUT_DIR / "q5_policy_reference_matrix.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(OUT_DIR / "q5_policy_recommendation_matrix.csv", index=False, encoding="utf-8-sig")
    write_summary(evidence, policy_matrix, recommendations, results)
    plot_q3_grid_exchange(results, OUT_DIR)
    plot_q4_grid_support(results, OUT_DIR)

    print("Q5 model evidence:")
    print(evidence.to_string(index=False))
    print("\nQ5 recommendations:")
    print(recommendations.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
