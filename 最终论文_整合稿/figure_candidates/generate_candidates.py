# -*- coding: utf-8 -*-
"""
Generate editable SVG diagrams and matching non-editable PNG previews for the
green-electricity direct-connection hydrogen-ammonia paper.

The SVG files are intended for editing in Inkscape/Illustrator/PowerPoint.
The PNG files are raster previews suitable for direct LaTeX insertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import atan2, cos, sin, pi
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SVG_DIR = ROOT / "editable_svg"
PNG_DIR = ROOT / "bitmap_png"
W, H = 2400, 1350


PALETTE = {
    "ink": "#17324D",
    "muted": "#5D6974",
    "line": "#6F7B83",
    "bg": "#F7FAFC",
    "panel": "#FFFFFF",
    "green": "#2F7D5A",
    "teal": "#138A94",
    "blue": "#2F5F9F",
    "orange": "#C46A1A",
    "yellow": "#D99A1B",
    "purple": "#6A57A5",
    "red": "#A64245",
    "gray": "#E6ECF0",
    "pale_green": "#EAF5EE",
    "pale_blue": "#EAF1FA",
    "pale_teal": "#E7F5F6",
    "pale_orange": "#FBEFE4",
    "pale_purple": "#F0EDF8",
    "pale_red": "#F8ECEE",
    "pale_yellow": "#FFF6DA",
}


def ensure_dirs() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def font_path(bold: bool = False) -> str:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return ""


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = font_path(bold)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def svg_text(
    x: float,
    y: float,
    lines: list[str] | tuple[str, ...] | str,
    *,
    size: int = 34,
    fill: str = PALETTE["ink"],
    bold: bool = False,
    anchor: str = "middle",
    line_height: float = 1.25,
) -> str:
    if isinstance(lines, str):
        lines = [lines]
    weight = "700" if bold else "400"
    tspans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else size * line_height
        tspans.append(
            f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Microsoft YaHei, SimHei, '
        f'Noto Sans CJK SC, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def svg_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "#FFFFFF",
    stroke: str = PALETTE["line"],
    sw: float = 3,
    r: float = 22,
    opacity: float | None = None,
) -> str:
    opacity_attr = "" if opacity is None else f' opacity="{opacity}"'
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
        f"{opacity_attr}/>"
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = PALETTE["line"],
    width: float = 4,
    arrow: bool = True,
    dash: str | None = None,
) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'fill="none"{marker}{dash_attr}/>'
    )


def svg_polyline(
    points: list[tuple[float, float]],
    *,
    color: str = PALETTE["line"],
    width: float = 4,
    arrow: bool = True,
    dash: str | None = None,
) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{pts}" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" fill="none"'
        f'{marker}{dash_attr}/>'
    )


def svg_path(
    d: str,
    *,
    color: str = PALETTE["line"],
    width: float = 4,
    arrow: bool = False,
    fill: str = "none",
    dash: str | None = None,
) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<path d="{d}" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" fill="{fill}"'
        f'{marker}{dash_attr}/>'
    )


def svg_circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "none", sw: float = 0) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def svg_header(title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["line"]}"/>',
        "</marker>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">',
        '<feDropShadow dx="0" dy="8" stdDeviation="7" flood-color="#263645" flood-opacity="0.14"/>',
        "</filter>",
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{PALETTE["bg"]}"/>',
        svg_text(W / 2, 86, title, size=52, fill=PALETTE["ink"], bold=True),
    ]


def svg_footer() -> str:
    return "</svg>\n"


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str] | tuple[str, ...] | str,
    *,
    size: int = 36,
    color: str = PALETTE["ink"],
    bold: bool = False,
    anchor: str = "mm",
    line_gap: int = 12,
) -> None:
    if isinstance(lines, str):
        lines = [lines]
    font = pil_font(size, bold)
    x, y = xy
    line_heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)
    if anchor.endswith("m"):
        start_y = y - total_h / 2
    elif anchor.endswith("a"):
        start_y = y
    else:
        start_y = y - total_h
    cur_y = start_y
    for line, width, height in zip(lines, widths, line_heights):
        if anchor.startswith("m"):
            tx = x - width / 2
        elif anchor.startswith("r"):
            tx = x - width
        else:
            tx = x
        draw.text((int(tx), int(cur_y)), line, font=font, fill=color)
        cur_y += height + line_gap


def rr(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str = PALETTE["line"],
    width: int = 4,
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    *,
    color: str = PALETTE["line"],
    width: int = 5,
    dash: bool = False,
) -> None:
    if dash:
        dashed_line(draw, p1, p2, fill=color, width=width, dash_len=22, gap=14)
    else:
        draw.line([p1, p2], fill=color, width=width)
    ang = atan2(p2[1] - p1[1], p2[0] - p1[0])
    length = 22
    spread = pi / 7
    pts = [
        p2,
        (int(p2[0] - length * cos(ang - spread)), int(p2[1] - length * sin(ang - spread))),
        (int(p2[0] - length * cos(ang + spread)), int(p2[1] - length * sin(ang + spread))),
    ]
    draw.polygon(pts, fill=color)


def polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str = PALETTE["line"],
    width: int = 5,
    dash: bool = False,
    head: bool = True,
) -> None:
    for start, end in zip(points, points[1:]):
        if dash:
            dashed_line(draw, start, end, fill=color, width=width, dash_len=22, gap=14)
        else:
            draw.line([start, end], fill=color, width=width)
    if head and len(points) >= 2:
        p1, p2 = points[-2], points[-1]
        ang = atan2(p2[1] - p1[1], p2[0] - p1[0])
        length = 22
        spread = pi / 7
        pts = [
            p2,
            (int(p2[0] - length * cos(ang - spread)), int(p2[1] - length * sin(ang - spread))),
            (int(p2[0] - length * cos(ang + spread)), int(p2[1] - length * sin(ang + spread))),
        ]
        draw.polygon(pts, fill=color)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    *,
    fill: str,
    width: int = 4,
    dash_len: int = 18,
    gap: int = 10,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    cur = 0.0
    while cur < dist:
        end = min(cur + dash_len, dist)
        draw.line(
            [
                (int(x1 + ux * cur), int(y1 + uy * cur)),
                (int(x1 + ux * end), int(y1 + uy * end)),
            ],
            fill=fill,
            width=width,
        )
        cur += dash_len + gap


def title_png(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw_text(draw, (W // 2, 82), text, size=54, color=PALETTE["ink"], bold=True)


def save_png(name: str, draw_fn) -> None:
    img = Image.new("RGB", (W, H), PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    img.save(PNG_DIR / f"{name}.png", quality=95)


def save_svg(name: str, title: str, body: list[str]) -> None:
    with (SVG_DIR / f"{name}.svg").open("w", encoding="utf-8") as f:
        f.write("\n".join(svg_header(title) + body + [svg_footer()]))


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: tuple[str, ...]
    fill: str
    stroke: str


def draw_box_png(draw: ImageDraw.ImageDraw, b: Box, *, title_size: int = 34, body_size: int = 28) -> None:
    rr(draw, (b.x, b.y, b.x + b.w, b.y + b.h), fill=b.fill, outline=b.stroke, width=5, radius=22)
    draw_text(draw, (b.x + b.w // 2, b.y + 40), b.title, size=title_size, color=b.stroke, bold=True)
    if b.lines:
        draw_text(draw, (b.x + b.w // 2, b.y + b.h // 2 + 26), list(b.lines), size=body_size, color=PALETTE["ink"])


def box_svg(b: Box, *, title_size: int = 34, body_size: int = 28) -> list[str]:
    return [
        svg_rect(b.x, b.y, b.w, b.h, fill=b.fill, stroke=b.stroke, sw=4.2, r=22),
        svg_text(b.x + b.w / 2, b.y + 44, b.title, size=title_size, fill=b.stroke, bold=True),
        svg_text(
            b.x + b.w / 2,
            b.y + b.h / 2 + 16,
            list(b.lines),
            size=body_size,
            fill=PALETTE["ink"],
            line_height=1.22,
        ),
    ]


def make_01_system() -> None:
    name = "candidate_01_system_coupling"
    title = "绿电直连电氢氨园区系统耦合示意"
    boxes = [
        Box(95, 230, 270, 145, "风电", ("6类场景", "P^w_s,t"), PALETTE["pale_green"], PALETTE["green"]),
        Box(95, 415, 270, 145, "光伏", ("4类场景", "P^pv_s,t"), PALETTE["pale_yellow"], PALETTE["yellow"]),
        Box(95, 600, 270, 145, "场景权重", ("24组合场景", "每场景15天"), "#FFFFFF", PALETTE["muted"]),
        Box(520, 385, 340, 185, "园区电母线", ("小时级功率平衡", "购电/上网互斥"), PALETTE["pale_blue"], PALETTE["blue"]),
        Box(1000, 215, 280, 125, "常规负荷", ("6 MW标幺曲线", "刚性用电"), "#FFFFFF", PALETTE["muted"]),
        Box(1000, 390, 280, 140, "电解制氢", ("碱性+PEM", "10%--100%负荷率"), PALETTE["pale_orange"], PALETTE["orange"]),
        Box(1385, 390, 280, 140, "合成氨", ("36/72 t/d", "NH3产量"), PALETTE["pale_orange"], PALETTE["orange"]),
        Box(1000, 610, 280, 140, "储能系统", ("SOC闭环", "5/80/155 MWh"), PALETTE["pale_purple"], PALETTE["purple"]),
        Box(520, 650, 340, 145, "公共电网", ("购电/上网", "735.81元/t支撑价值"), PALETTE["pale_blue"], PALETTE["blue"]),
        Box(1810, 210, 405, 145, "绿电直连指标", ("Rself ≥ 60%", "Rgreen ≥ 30%", "Rsell ≤ 20%"), PALETTE["pale_red"], PALETTE["red"]),
        Box(1810, 420, 405, 145, "经济评价", ("吨氨成本", "产能利用率、弃电率"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(1810, 630, 405, 145, "政策证据链", ("E1--E8模型证据", "准入、补偿、分担"), "#FFFFFF", PALETTE["ink"]),
    ]
    qboxes = [
        Box(105, 990, 335, 155, "Q1 刚性基准", ("上网35.92%", "识别源荷错配"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(505, 990, 335, 155, "Q2 离散开停", ("边际成本排序", "成本--指标权衡"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(905, 990, 335, 155, "Q3 连续调节", ("MILP负荷率", "柔性负荷价值"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(1305, 990, 335, 155, "Q4 离网储能", ("储能容量扫描", "经济--消纳权衡"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(1705, 990, 335, 155, "Q5 系统影响", ("证据矩阵", "政策机制设计"), PALETTE["pale_teal"], PALETTE["teal"]),
    ]

    def png(draw: ImageDraw.ImageDraw) -> None:
        title_png(draw, title)
        rr(draw, (50, 160, 2250, 855), fill="#FFFFFF", outline=PALETTE["gray"], width=3, radius=34)
        for b in boxes:
            draw_box_png(draw, b)
        for b in qboxes:
            draw_box_png(draw, b, title_size=32, body_size=26)
        draw_text(draw, (235, 185), "资源与场景", size=30, color=PALETTE["muted"], bold=True)
        draw_text(draw, (1110, 185), "园区能量耦合", size=30, color=PALETTE["muted"], bold=True)
        draw_text(draw, (2010, 185), "指标与证据", size=30, color=PALETTE["muted"], bold=True)
        arrow(draw, (365, 302), (520, 448), color=PALETTE["green"])
        arrow(draw, (365, 488), (520, 475), color=PALETTE["yellow"])
        arrow(draw, (365, 672), (520, 520), color=PALETTE["line"], dash=True)
        arrow(draw, (860, 470), (1000, 455), color=PALETTE["orange"])
        arrow(draw, (1280, 460), (1385, 460), color=PALETTE["orange"])
        arrow(draw, (860, 500), (1000, 655), color=PALETTE["purple"])
        arrow(draw, (690, 570), (690, 650), color=PALETTE["blue"])
        arrow(draw, (1665, 460), (1810, 282), color=PALETTE["red"], dash=True)
        arrow(draw, (1665, 460), (1810, 493), color=PALETTE["teal"])
        arrow(draw, (1280, 675), (1810, 704), color=PALETTE["ink"], dash=True)
        for i in range(len(qboxes) - 1):
            arrow(draw, (qboxes[i].x + qboxes[i].w, 1068), (qboxes[i + 1].x, 1068), color=PALETTE["line"])
        dashed_line(draw, (690, 795), (690, 990), fill=PALETTE["line"], width=4)
        dashed_line(draw, (1140, 750), (1472, 990), fill=PALETTE["purple"], width=4)
        dashed_line(draw, (2012, 775), (1870, 990), fill=PALETTE["red"], width=4)
        draw_text(draw, (1080, 932), "五问递进：从核算到优化，再到系统影响", size=30, color=PALETTE["muted"], bold=True)

    body: list[str] = [svg_rect(50, 160, 2200, 695, fill="#FFFFFF", stroke=PALETTE["gray"], sw=3, r=34)]
    body.extend([svg_text(235, 185, "资源与场景", size=30, fill=PALETTE["muted"], bold=True)])
    body.extend([svg_text(1110, 185, "园区能量耦合", size=30, fill=PALETTE["muted"], bold=True)])
    body.extend([svg_text(2010, 185, "指标与证据", size=30, fill=PALETTE["muted"], bold=True)])
    for b in boxes:
        body.extend(box_svg(b))
    for b in qboxes:
        body.extend(box_svg(b, title_size=32, body_size=26))
    body.extend(
        [
            svg_line(365, 302, 520, 448, color=PALETTE["green"]),
            svg_line(365, 488, 520, 475, color=PALETTE["yellow"]),
            svg_line(365, 672, 520, 520, color=PALETTE["line"], dash="18 14"),
            svg_line(860, 470, 1000, 455, color=PALETTE["orange"]),
            svg_line(1280, 460, 1385, 460, color=PALETTE["orange"]),
            svg_line(860, 500, 1000, 655, color=PALETTE["purple"]),
            svg_line(690, 570, 690, 650, color=PALETTE["blue"]),
            svg_line(1665, 460, 1810, 282, color=PALETTE["red"], dash="18 14"),
            svg_line(1665, 460, 1810, 493, color=PALETTE["teal"]),
            svg_line(1280, 675, 1810, 704, color=PALETTE["ink"], dash="18 14"),
            svg_text(1080, 932, "五问递进：从核算到优化，再到系统影响", size=30, fill=PALETTE["muted"], bold=True),
        ]
    )
    for i in range(len(qboxes) - 1):
        body.append(svg_line(qboxes[i].x + qboxes[i].w, 1068, qboxes[i + 1].x, 1068))
    body.extend(
        [
            svg_line(690, 795, 690, 990, arrow=False, dash="18 14"),
            svg_line(1140, 750, 1472, 990, arrow=False, color=PALETTE["purple"], dash="18 14"),
            svg_line(2012, 775, 1870, 990, arrow=False, color=PALETTE["red"], dash="18 14"),
        ]
    )
    save_png(name, png)
    save_svg(name, title, body)


def make_02_algorithm() -> None:
    name = "candidate_02_algorithm_pipeline"
    title = "五问模型求解与结果校验闭环"
    top = [
        Box(110, 230, 260, 130, "附件1--8", ("负荷/风光", "设备/电价"), "#FFFFFF", PALETTE["muted"]),
        Box(430, 230, 300, 130, "场景构造", ("典型日+24场景", "360天加权"), PALETTE["pale_blue"], PALETTE["blue"]),
        Box(790, 230, 300, 130, "公共模块", ("data.py", "metrics.py"), PALETTE["pale_blue"], PALETTE["blue"]),
        Box(1150, 230, 300, 130, "统一口径", ("功率平衡", "绿电三指标"), PALETTE["pale_red"], PALETTE["red"]),
        Box(1510, 230, 300, 130, "求解器/规则", ("排序法", "SciPy HiGHS"), PALETTE["pale_yellow"], PALETTE["yellow"]),
        Box(1870, 230, 300, 130, "绘图输出", ("统一风格", "PDF/SVG/PNG"), PALETTE["pale_purple"], PALETTE["purple"]),
    ]
    mid = [
        Box(110, 545, 300, 145, "Q1", ("逐小时核算", "O(24)"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(485, 545, 300, 145, "Q2", ("边际成本排序", "离散开停"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(860, 545, 300, 145, "Q3", ("连续负荷率MILP", "购售电互斥"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(1235, 545, 300, 145, "Q4", ("离网储能MILP", "容量扫描"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(1610, 545, 300, 145, "Q5", ("读取前四问CSV", "证据矩阵"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(1985, 545, 270, 145, "模型结果", ("成本/指标", "图表/摘要"), "#FFFFFF", PALETTE["ink"]),
    ]
    bot = [
        Box(240, 920, 360, 150, "自动校验", ("功率平衡、年度权重", "互斥约束、SOC闭环"), PALETTE["pale_yellow"], PALETTE["yellow"]),
        Box(720, 920, 360, 150, "结果沉淀", ("CSV摘要", "Markdown说明、PDF图件"), PALETTE["pale_purple"], PALETTE["purple"]),
        Box(1200, 920, 360, 150, "论文整合", ("正文数字交叉核对", "main.tex"), PALETTE["pale_blue"], PALETTE["blue"]),
        Box(1680, 920, 360, 150, "最终交付", ("XeLaTeX编译", "PDF+支撑材料"), PALETTE["pale_red"], PALETTE["red"]),
    ]

    def png(draw: ImageDraw.ImageDraw) -> None:
        title_png(draw, title)
        rr(draw, (70, 175, 2250, 410), fill="#EEF5FB", outline="#D4E2EE", width=3, radius=28)
        rr(draw, (70, 480, 2250, 745), fill="#EAF7F7", outline="#CFE7E9", width=3, radius=28)
        rr(draw, (70, 850, 2250, 1125), fill="#FFF8E6", outline="#E8DCA8", width=3, radius=28)
        draw_text(draw, (150, 200), "数据与口径", size=30, color=PALETTE["muted"], bold=True, anchor="la")
        draw_text(draw, (150, 505), "分问题求解", size=30, color=PALETTE["muted"], bold=True, anchor="la")
        draw_text(draw, (150, 875), "校验与交付", size=30, color=PALETTE["muted"], bold=True, anchor="la")
        for group in (top, mid, bot):
            for b in group:
                draw_box_png(draw, b, title_size=32, body_size=27)
        for group in (top, mid, bot):
            for i in range(len(group) - 1):
                arrow(draw, (group[i].x + group[i].w, group[i].y + group[i].h // 2), (group[i + 1].x, group[i + 1].y + group[i + 1].h // 2))
        for b in mid[:5]:
            arrow(draw, (940, 360), (b.x + b.w // 2, b.y), color=PALETTE["teal"])
        arrow(draw, (2120, 690), (2040, 920), color=PALETTE["line"])
        arrow(draw, (600, 995), (720, 995), color=PALETTE["line"])
        polyline(draw, [(420, 920), (420, 800), (1290, 800), (1290, 360)], color=PALETTE["red"], width=4, dash=True)
        polyline(draw, [(900, 920), (900, 780), (2020, 780), (2020, 360)], color=PALETTE["purple"], width=4, dash=True)

    body = [
        svg_rect(70, 175, 2250, 235, fill="#EEF5FB", stroke="#D4E2EE", sw=3, r=28),
        svg_rect(70, 480, 2250, 265, fill="#EAF7F7", stroke="#CFE7E9", sw=3, r=28),
        svg_rect(70, 850, 2250, 275, fill="#FFF8E6", stroke="#E8DCA8", sw=3, r=28),
        svg_text(150, 205, "数据与口径", size=30, fill=PALETTE["muted"], bold=True, anchor="start"),
        svg_text(150, 535, "分问题求解", size=30, fill=PALETTE["muted"], bold=True, anchor="start"),
        svg_text(150, 905, "校验与交付", size=30, fill=PALETTE["muted"], bold=True, anchor="start"),
    ]
    for group in (top, mid, bot):
        for b in group:
            body.extend(box_svg(b, title_size=32, body_size=27))
        for i in range(len(group) - 1):
            body.append(svg_line(group[i].x + group[i].w, group[i].y + group[i].h / 2, group[i + 1].x, group[i + 1].y + group[i + 1].h / 2))
    for b in mid[:5]:
        body.append(svg_line(940, 360, b.x + b.w / 2, b.y, color=PALETTE["teal"]))
    body.extend(
        [
            svg_line(2120, 690, 2040, 920),
            svg_line(600, 995, 720, 995),
            svg_polyline([(420, 920), (420, 800), (1290, 800), (1290, 360)], color=PALETTE["red"], width=4, dash="18 14"),
            svg_polyline([(900, 920), (900, 780), (2020, 780), (2020, 360)], color=PALETTE["purple"], width=4, dash="18 14"),
        ]
    )
    save_png(name, png)
    save_svg(name, title, body)


def mini_curve_png(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, color: str, mode: str) -> None:
    rr(draw, (x, y, x + w, y + h), fill="#FFFFFF", outline="#D7E0E6", width=2, radius=16)
    baseline = y + h - 32
    draw.line((x + 24, baseline, x + w - 18, baseline), fill="#BAC5CC", width=3)
    if mode == "flat":
        pts = [(x + 28, y + 70), (x + w - 22, y + 70)]
        draw.line(pts, fill=color, width=8)
    elif mode == "step":
        pts = [(x + 28, baseline), (x + 90, baseline), (x + 90, y + 60), (x + 170, y + 60), (x + 170, baseline), (x + 250, baseline), (x + 250, y + 60), (x + w - 25, y + 60)]
        draw.line(pts, fill=color, width=7, joint="curve")
    elif mode == "smooth":
        pts = []
        for i in range(0, 250):
            xx = x + 28 + i * (w - 55) / 249
            yy = y + 70 + 28 * sin(i / 28)
            pts.append((int(xx), int(yy)))
        draw.line(pts, fill=color, width=7)
    else:
        pts = [(x + 28, y + 90), (x + 95, y + 55), (x + 160, y + 98), (x + 230, y + 50), (x + w - 28, y + 80)]
        draw.line(pts, fill=color, width=7)


def mini_curve_svg(x: int, y: int, w: int, h: int, color: str, mode: str) -> list[str]:
    body = [svg_rect(x, y, w, h, fill="#FFFFFF", stroke="#D7E0E6", sw=2.2, r=16), svg_line(x + 24, y + h - 32, x + w - 18, y + h - 32, color="#BAC5CC", width=3, arrow=False)]
    if mode == "flat":
        body.append(svg_line(x + 28, y + 70, x + w - 22, y + 70, color=color, width=8, arrow=False))
    elif mode == "step":
        pts = [(x + 28, y + h - 32), (x + 90, y + h - 32), (x + 90, y + 60), (x + 170, y + 60), (x + 170, y + h - 32), (x + 250, y + h - 32), (x + 250, y + 60), (x + w - 25, y + 60)]
        body.append(svg_polyline(pts, color=color, width=7, arrow=False))
    elif mode == "smooth":
        pts = []
        for i in range(0, 90):
            xx = x + 28 + i * (w - 55) / 89
            yy = y + 70 + 28 * sin(i / 10)
            pts.append((xx, yy))
        body.append(svg_polyline(pts, color=color, width=7, arrow=False))
    else:
        body.append(svg_polyline([(x + 28, y + 90), (x + 95, y + 55), (x + 160, y + 98), (x + 230, y + 50), (x + w - 28, y + 80)], color=color, width=7, arrow=False))
    return body


def make_03_flexibility() -> None:
    name = "candidate_03_flexibility_logic"
    title = "制氨负荷柔性递进与储能配置逻辑"
    cards = [
        Box(110, 240, 470, 570, "Q1 刚性满负荷", ("生产负荷全天固定", "上网比例35.92%", "作为后续基准"), "#FFFFFF", PALETTE["red"]),
        Box(675, 240, 470, 570, "Q2 离散开停", ("选择生产小时", "边际成本排序", "54 t/d典型合规"), "#FFFFFF", PALETTE["orange"]),
        Box(1240, 240, 470, 570, "Q3 连续调节", ("10%--100%负荷率", "MILP+购售电互斥", "提高全满足天数"), "#FFFFFF", PALETTE["teal"]),
        Box(1805, 240, 470, 570, "Q4 离网储能", ("删除购售电变量", "SOC闭环+容量扫描", "5/80/155 MWh权衡"), "#FFFFFF", PALETTE["purple"]),
    ]
    colors = [PALETTE["red"], PALETTE["orange"], PALETTE["teal"], PALETTE["purple"]]
    modes = ["flat", "step", "smooth", "storage"]

    def png(draw: ImageDraw.ImageDraw) -> None:
        title_png(draw, title)
        draw_text(draw, (W // 2, 148), "从“固定生产”到“小时级柔性”，再到“完全离网可靠性”的建模升级", size=34, color=PALETTE["muted"], bold=True)
        for idx, b in enumerate(cards):
            rr(draw, (b.x, b.y, b.x + b.w, b.y + b.h), fill=b.fill, outline=colors[idx], width=6, radius=28)
            draw_text(draw, (b.x + b.w // 2, b.y + 56), b.title, size=38, color=colors[idx], bold=True)
            mini_curve_png(draw, b.x + 55, b.y + 135, b.w - 110, 165, colors[idx], modes[idx])
            draw_text(draw, (b.x + b.w // 2, b.y + 405), list(b.lines), size=31, color=PALETTE["ink"], line_gap=15)
        for idx in range(3):
            arrow(draw, (cards[idx].x + cards[idx].w + 10, 525), (cards[idx + 1].x - 20, 525), color=PALETTE["line"])
        rr(draw, (230, 930, 2150, 210 + 930), fill=PALETTE["pale_blue"], outline=PALETTE["blue"], width=5, radius=30)
        draw_text(draw, (W // 2, 982), "Q5 系统影响分析", size=40, color=PALETTE["blue"], bold=True)
        draw_text(draw, (W // 2, 1066), ("将前四问输出转化为新能源消纳、电网调峰、备用配置、费用分担和政策准入证据", "形成“模型依据--系统影响--政策含义”的闭环"), size=34, color=PALETTE["ink"], line_gap=18)

    body: list[str] = [
        svg_text(W / 2, 148, "从“固定生产”到“小时级柔性”，再到“完全离网可靠性”的建模升级", size=34, fill=PALETTE["muted"], bold=True)
    ]
    for idx, b in enumerate(cards):
        body.append(svg_rect(b.x, b.y, b.w, b.h, fill=b.fill, stroke=colors[idx], sw=6, r=28))
        body.append(svg_text(b.x + b.w / 2, b.y + 66, b.title, size=38, fill=colors[idx], bold=True))
        body.extend(mini_curve_svg(b.x + 55, b.y + 135, b.w - 110, 165, colors[idx], modes[idx]))
        body.append(svg_text(b.x + b.w / 2, b.y + 405, list(b.lines), size=31, fill=PALETTE["ink"], line_height=1.25))
    for idx in range(3):
        body.append(svg_line(cards[idx].x + cards[idx].w + 10, 525, cards[idx + 1].x - 20, 525))
    body.extend(
        [
            svg_rect(230, 930, 2150, 210, fill=PALETTE["pale_blue"], stroke=PALETTE["blue"], sw=5, r=30),
            svg_text(W / 2, 992, "Q5 系统影响分析", size=40, fill=PALETTE["blue"], bold=True),
            svg_text(W / 2, 1066, ["将前四问输出转化为新能源消纳、电网调峰、备用配置、费用分担和政策准入证据", "形成“模型依据--系统影响--政策含义”的闭环"], size=34, fill=PALETTE["ink"], line_height=1.32),
        ]
    )
    save_png(name, png)
    save_svg(name, title, body)


def make_04_policy() -> None:
    name = "candidate_04_policy_evidence"
    title = "模型结果到政策建议的证据链"
    left = [
        Box(125, 230, 430, 145, "E1 源荷错配", ("刚性运行下购电与上网并存", "上网比例35.92%"), PALETTE["pale_red"], PALETTE["red"]),
        Box(125, 415, 430, 145, "E2 柔性负荷价值", ("连续调节降低吨氨成本", "提高指标合格天数"), PALETTE["pale_teal"], PALETTE["teal"]),
        Box(125, 600, 430, 145, "E3 储能边际收益", ("5 MWh经济", "80/155 MWh消纳目标"), PALETTE["pale_purple"], PALETTE["purple"]),
        Box(125, 785, 430, 145, "E4 电网支撑价值", ("并网同产量对比", "735.81元/t"), PALETTE["pale_blue"], PALETTE["blue"]),
    ]
    middle = [
        Box(815, 250, 390, 160, "新能源消纳", ("上网比例约束", "弃风弃光评价"), "#FFFFFF", PALETTE["green"]),
        Box(815, 475, 390, 160, "调峰与备用", ("净交换功率", "柔性负荷响应"), "#FFFFFF", PALETTE["orange"]),
        Box(815, 700, 390, 160, "费用分担", ("公共电网支撑", "离网成本对照"), "#FFFFFF", PALETTE["blue"]),
    ]
    right = [
        Box(1510, 205, 560, 120, "小时级绿电溯源", ("避免年电量平衡掩盖时序错配",), "#FFFFFF", PALETTE["red"]),
        Box(1510, 365, 560, 120, "上网比例与交换功率考核", ("同时约束能量占比和峰值冲击",), "#FFFFFF", PALETTE["green"]),
        Box(1510, 525, 560, 120, "柔性负荷准入与补偿", ("按削峰填谷和消纳贡献定价",), "#FFFFFF", PALETTE["teal"]),
        Box(1510, 685, 560, 120, "储能多目标评价", ("区分经济容量、低弃电和零弃电方案",), "#FFFFFF", PALETTE["purple"]),
        Box(1510, 845, 560, 120, "系统费用分担机制", ("将电网平衡服务显性化",), "#FFFFFF", PALETTE["blue"]),
    ]

    def png(draw: ImageDraw.ImageDraw) -> None:
        title_png(draw, title)
        draw_text(draw, (330, 175), "前四问模型证据", size=34, color=PALETTE["muted"], bold=True)
        draw_text(draw, (1010, 175), "电力系统影响", size=34, color=PALETTE["muted"], bold=True)
        draw_text(draw, (1790, 175), "政策建议", size=34, color=PALETTE["muted"], bold=True)
        for group in (left, middle, right):
            for b in group:
                draw_box_png(draw, b, title_size=31, body_size=27)
        for b in left:
            arrow(draw, (b.x + b.w, b.y + b.h // 2), (815, b.y + b.h // 2 if b.y < 650 else 780), color=PALETTE["line"])
        for b in middle:
            arrow(draw, (b.x + b.w, b.y + b.h // 2), (1510, b.y + b.h // 2), color=PALETTE["line"])
        rr(draw, (655, 980, 2125, 1150), fill=PALETTE["pale_yellow"], outline=PALETTE["yellow"], width=5, radius=28)
        draw_text(draw, (1390, 1037), "核心逻辑", size=36, color=PALETTE["yellow"], bold=True)
        draw_text(draw, (1390, 1100), "运行模型不是政策论述的附属，而是政策机制设计的证据来源", size=34, color=PALETTE["ink"])

    body = [
        svg_text(330, 175, "前四问模型证据", size=34, fill=PALETTE["muted"], bold=True),
        svg_text(1010, 175, "电力系统影响", size=34, fill=PALETTE["muted"], bold=True),
        svg_text(1790, 175, "政策建议", size=34, fill=PALETTE["muted"], bold=True),
    ]
    for group in (left, middle, right):
        for b in group:
            body.extend(box_svg(b, title_size=31, body_size=27))
    for b in left:
        body.append(svg_line(b.x + b.w, b.y + b.h / 2, 815, b.y + b.h / 2 if b.y < 650 else 780))
    for b in middle:
        body.append(svg_line(b.x + b.w, b.y + b.h / 2, 1510, b.y + b.h / 2))
    body.extend(
        [
            svg_rect(655, 980, 1470, 170, fill=PALETTE["pale_yellow"], stroke=PALETTE["yellow"], sw=5, r=28),
            svg_text(1390, 1047, "核心逻辑", size=36, fill=PALETTE["yellow"], bold=True),
            svg_text(1390, 1100, "运行模型不是政策论述的附属，而是政策机制设计的证据来源", size=34, fill=PALETTE["ink"]),
        ]
    )
    save_png(name, png)
    save_svg(name, title, body)


def write_readme() -> None:
    text = """# 图件候选说明

本文件夹用于单独比较论文插图候选，不会自动改动 `main.tex`。

## 文件夹

- `bitmap_png/`：不可编辑位图，可直接用 `\\includegraphics` 插入论文。
- `editable_svg/`：可编辑矢量图，可用 Inkscape、Illustrator、PowerPoint 等打开后改字、改颜色、移动元素。
- `generate_candidates.py`：图件生成脚本，可重新生成同名 PNG/SVG。

## 候选图

1. `candidate_01_system_coupling`：系统背景与源-荷-储-网-氨耦合示意，适合放在“问题分析”后，替换当前图1。
2. `candidate_02_algorithm_pipeline`：数据读取、五问求解、校验和交付闭环，适合放在“求解算法与可复现流程”后，替换当前图2。
3. `candidate_03_flexibility_logic`：从刚性满负荷、离散开停、连续调节到离网储能的算法逻辑递进图，适合放在“问题分析”或“统一模型框架”附近。
4. `candidate_04_policy_evidence`：前四问结果支撑问题五政策建议的证据链，适合放在“问题五：绿电直连高渗透影响分析”中。

## 论文插入示例

如果选择 PNG：

```tex
\\begin{figure}[H]
  \\centering
  \\includegraphics[width=0.95\\textwidth]{figure_candidates/bitmap_png/candidate_01_system_coupling.png}
  \\caption{绿电直连型电氢氨园区系统耦合示意}
\\end{figure}
```

如果选择 SVG 矢量图，建议先用 Inkscape/Illustrator 修改并导出为 PDF，再在论文中插入 PDF。
"""
    (ROOT / "README_图件候选说明.md").write_text(text, encoding="utf-8")


def write_index() -> None:
    cards = [
        ("candidate_01_system_coupling", "系统耦合示意", "适合替换当前图1，放在“问题分析”后。"),
        ("candidate_02_algorithm_pipeline", "算法与校验闭环", "适合替换当前图2，放在“求解算法与可复现流程”后。"),
        ("candidate_03_flexibility_logic", "柔性递进逻辑", "适合放在“问题分析”或“统一模型框架”附近。"),
        ("candidate_04_policy_evidence", "政策证据链", "适合放在问题五系统影响分析中。"),
    ]
    body = "\n".join(
        f"""
        <section class="card">
          <h2>{idx}. {title}</h2>
          <p>{desc}</p>
          <img src="bitmap_png/{name}.png" alt="{title}">
          <p class="links">
            <a href="bitmap_png/{name}.png">PNG位图</a>
            <a href="editable_svg/{name}.svg">SVG矢量图</a>
          </p>
        </section>
        """
        for idx, (name, title, desc) in enumerate(cards, start=1)
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>论文图件候选总览</title>
  <style>
    body {{ margin: 0; padding: 28px; background: #f4f7fa; color: #17324d; font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; }}
    h1 {{ text-align: center; margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; max-width: 1280px; margin: 0 auto; }}
    .card {{ background: white; border: 1px solid #dbe5ec; border-radius: 14px; padding: 18px 20px 22px; box-shadow: 0 8px 28px rgba(23,50,77,.08); }}
    h2 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 0 0 12px; color: #5d6974; }}
    img {{ display: block; width: 100%; border: 1px solid #e6ecf0; border-radius: 10px; }}
    .links {{ margin-top: 12px; }}
    a {{ display: inline-block; margin-right: 12px; color: #2f5f9f; text-decoration: none; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>论文图件候选总览</h1>
  <main class="grid">
    {body}
  </main>
</body>
</html>
"""
    (ROOT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    make_01_system()
    make_02_algorithm()
    make_03_flexibility()
    make_04_policy()
    write_readme()
    write_index()
    print(f"Generated candidates under: {ROOT}")


if __name__ == "__main__":
    main()
