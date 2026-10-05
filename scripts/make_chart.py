#!/usr/bin/env python3
"""Render the DemoDoctor result chart (card light/dark 1200x750, cover 2000x840) with headless Chrome.

Numbers come from outputs/final_table.txt (condition means and bootstrap CIs) and
outputs/eval_labelled/*/eval_info.json (per-policy dots), so the picture cannot drift from the table.
Usage: python scripts/make_chart.py   (run scripts/final_table.py first)
"""
import glob
import json
import re
import subprocess
from pathlib import Path
from statistics import mean

OUT = Path("outputs/chart")
OUT.mkdir(parents=True, exist_ok=True)
ROWS = [("clean", "Clean"), ("corrupted", "Corrupted"), ("cleaned", "Auto-cleaned"), ("random175", "Random 175 (size control)")]
LO, HI = 0.28, 0.52

dots = {c: [] for c, _ in ROWS}
for f in sorted(glob.glob("outputs/eval_labelled/*/eval_info.json")):
    m = re.search(r"/(\w+)_s(\d)/", f)
    dots[m.group(1)].append(mean(json.load(open(f))["per_task"][0]["metrics"]["max_rewards"]))
ci = {}
for line in open("outputs/final_table.txt"):
    m = re.match(r"(\w+)\s+n_policies=\d+.*max reward ([\d.]+) \[([\d.]+), ([\d.]+)\]", line)
    if m:
        ci[m.group(1)] = tuple(float(x) for x in m.groups()[1:])

THEMES = {
    "light": dict(bg="#fbfbfa", fg="#111111", muted="#6b6b6b", grid="#d8d6d0", dot="#b8b5ad", accent="#2b6fdc", card="#fbfbfa", border="#e4e2dc"),
    "dark": dict(bg="#0e0f11", fg="#ffffff", muted="#9a9a9a", grid="#3a3a3a", dot="#6a6a6a", accent="#3b82f0", card="#0e0f11", border="#2a2a2a"),
}


def card_svg(t, w=1200, h=750):
    x0, x1 = 400, w - 70
    sx = lambda v: x0 + (v - LO) / (HI - LO) * (x1 - x0)
    parts = [f'<text x="48" y="78" font-size="42" font-weight="700" fill="{t["fg"]}">Data quality did not separate the policies</text>',
             f'<text x="48" y="124" font-size="24" fill="{t["muted"]}">Mean max reward per policy (dots), condition mean with 95% CI (bar)</text>']
    for v in (0.30, 0.35, 0.40, 0.45, 0.50):
        parts.append(f'<line x1="{sx(v)}" y1="170" x2="{sx(v)}" y2="548" stroke="{t["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{sx(v)}" y="580" font-size="20" text-anchor="middle" fill="{t["muted"]}">{v:.2f}</text>')
    for i, (c, label) in enumerate(ROWS):
        y = 215 + i * 92
        mu, lo, hi = ci[c]
        parts.append(f'<text x="48" y="{y + 8}" font-size="26" fill="{t["fg"]}">{label}</text>')
        parts.append(f'<rect x="{sx(lo)}" y="{y - 5}" width="{sx(hi) - sx(lo)}" height="10" rx="5" fill="{t["accent"]}" opacity="0.28"/>')
        for d in dots[c]:
            parts.append(f'<circle cx="{sx(d)}" cy="{y + 24}" r="8" fill="{t["dot"]}"/>')
        parts.append(f'<rect x="{sx(mu) - 3}" y="{y - 14}" width="6" height="28" rx="3" fill="{t["accent"]}"/>')
    parts.append(f'<text x="48" y="640" font-size="22" fill="{t["muted"]}">PushT, ACT, 60k steps. 3 seeds x 50 rollouts per condition. Success is 0-4% everywhere,</text>')
    parts.append(f'<text x="48" y="672" font-size="22" fill="{t["muted"]}">so max reward is the signal. No pairwise CI excludes zero.</text>')
    parts.append(f'<g transform="translate(48 706)"><circle cx="8" cy="-6" r="8" fill="{t["dot"]}"/><text x="26" y="0" font-size="20" fill="{t["muted"]}">one policy</text>'
                 f'<rect x="170" y="-20" width="6" height="28" rx="3" fill="{t["accent"]}"/><text x="188" y="0" font-size="20" fill="{t["muted"]}">condition mean, 95% CI</text></g>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Inter, Helvetica, Arial, sans-serif"><rect width="{w}" height="{h}" fill="{t["bg"]}"/>{"".join(parts)}</svg>'


def render(html, png, w, h):
    f = OUT / (png + ".html")
    f.write_text(f'<!doctype html><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style>{html}')
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1",
                    f"--window-size={w},{h}", f"--screenshot={OUT / (png + '.png')}", f"file://{f.resolve()}"],
                   check=True, capture_output=True, timeout=120)


for name, t in THEMES.items():
    render(card_svg(t), f"card-{name}", 1200, 750)

t = THEMES["light"]
scale = 0.72
cover = f'''<div style="width:2000px;height:840px;background:{t["bg"]};position:relative;font-family:Inter,Helvetica,Arial,sans-serif;color:{t["fg"]}">
<div style="position:absolute;left:96px;top:290px;font-size:34px;color:{t["muted"]}">raihan-js / demodoctor</div>
<div style="position:absolute;left:92px;top:350px;font-size:116px;font-weight:700;letter-spacing:-2px">DemoDoctor</div>
<div style="position:absolute;left:96px;top:500px;font-size:40px;color:{t["muted"]};width:820px;line-height:1.3">Bad robot demonstrations: detectable, but did they matter?</div>
<div style="position:absolute;left:1000px;top:150px;width:{int(1200*scale)}px;height:{int(750*scale)}px;border:2px solid {t["border"]};border-radius:36px;overflow:hidden">
<div style="transform:scale({scale});transform-origin:0 0;width:1200px;height:750px">{card_svg(t)}</div></div></div>'''
render(cover, "cover", 2000, 840)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
