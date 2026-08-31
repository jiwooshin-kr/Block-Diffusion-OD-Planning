"""
n_is sweep report: how many importance-sampling candidates does IW guidance need?

  python report_src/build_nis_sweep.py -variant LE
  python report_src/build_nis_sweep.py -variant LP -nis 10,30,50,100,150

Inputs
  sets_v6/{V}/res/va_table_v6{V}_nis{n}{,_rawL,_P1P3,_P1P3L}.json   scripts/run_v6_nis.sh
  sets_v6/{V}/res/nis_diversity_v6{V}.json                          tools/diag/diag_proposal_diversity.py
  sets_v6/{V}/res/shortest_v6{V}.json                               tools/eval/shortest_baseline.py
Output
  report_src/nis_sweep_{V}_ko.html  ->  pdfs/Porto_{V}/nis_sweep/nis_sweep_{V}_ko.pdf
  sets_v6/{V}/res/nis_plot_data.csv (tidy, one row per post/regime/blk/arm/n_is)

  PDF:
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
      --no-pdf-header-footer --print-to-pdf=pdfs/Porto_LE/nis_sweep/nis_sweep_LE_ko.pdf \
      "file://$PWD/report_src/nis_sweep_LE_ko.html"

Layout choice: one table per metric with n_is across the columns, so a flat row is
visible at a glance. base and Adj-only do not consume candidates at all (base does
not call plan_guided; Adj-only passes disc=None and reveals straight from the
masked marginal), so their rows must be bit-identical across n_is -- they are
printed as a determinism check, not as a result.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import csv
import io
import json
import os
import re
from os.path import dirname, join

ROOT = dirname(dirname(os.path.abspath(__file__)))

ARMS = [("base", "base", "none"), ("adjonly", "Adj-only", "iw"),
        ("modelD", "IW-only", "iw"), ("adj+modelD", "Adj+IW", "iw")]
NIS_FREE = {"base", "adjonly"}          # arms that never draw candidates
POSTS = [("raw", ""), ("rawL", "_rawL"), ("P1P3", "_P1P3"), ("P1P3L", "_P1P3L")]

# (json key, label, direction: +1 larger better, -1 smaller better, 0 context)
MET = [("valid_and_arrival", "valid &amp; arrival", +1),
       ("dtw_sum", "DTW&darr; (m)", -1),
       ("lcs", "LCS", +1),
       ("gen_len", "gen_len", 0)]
FMT = {"valid_and_arrival": "{:.3f}", "dtw_sum": "{:.0f}", "lcs": "{:.2f}",
       "gen_len": "{:.1f}"}

_H2N = [0]


def H2(title):
    _H2N[0] += 1
    return f"<h2>{_H2N[0]}. {title}</h2>"


def jload(p, required=True):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        if required:
            raise SystemExit(f"MISSING required input: {p}\n"
                             f"(scripts/run_v6_nis.sh 단계 B 가 끝나야 생성된다)")
        return None


def get(J, regime, blk, cfg):
    """unseen 에 없는 arm(base/Adj-only)은 seen 실행분을 공유한다 — 원래 프로토콜."""
    return J.get(f"{regime}_blk{blk}_{cfg}") or J.get(f"seen_blk{blk}_{cfg}")


def spread(vals):
    """Range across n_is, the quantity this report is really about."""
    v = [x for x in vals if x is not None]
    return (max(v) - min(v)) if len(v) >= 2 else float("nan")


def trend(vals):
    """Direction marker for a row: a range alone cannot distinguish a real trend
    from noise, so report whether the steps go one way.

    Returns (marker, n_up, n_down) where the marker is
      ''  flat (range 0)
      &uarr;/&darr;  every step in one direction -- a real trend
      &nearr;/&searr;  a clear majority in one direction
      &sim;  mixed, i.e. noise
    """
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return "", 0, 0
    d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    up = sum(1 for x in d if x > 0)
    dn = sum(1 for x in d if x < 0)
    if up == 0 and dn == 0:
        return "", 0, 0
    if dn == 0:
        return "&uarr;", up, dn
    if up == 0:
        return "&darr;", up, dn
    if up >= 3 * max(dn, 1):
        return "&nearr;", up, dn
    if dn >= 3 * max(up, 1):
        return "&searr;", up, dn
    return "&sim;", up, dn


def metric_table(TAB, nis, blks, regime, met, direction, S=None):
    """One metric, n_is across the columns, (blk, arm) down the rows.

    The last column is the range across n_is: that is the number the reader should
    look at, because the claim under test is that the row is flat.
    """
    head = ("<tr><th>blk</th><th>구성</th>"
            + "".join(f'<th class="num">n={n}</th>' for n in nis)
            + '<th class="num">범위</th>'
            + '<th class="num" title="n 증가 방향의 추세: ↑/↓ 전 구간 일방, ↗/↘ 다수, ~ 혼재">추세</th></tr>')
    rows = [head]
    for b in blks:
        present = [(c, lab, fam) for c, lab, fam in ARMS
                   if any(get(TAB[n], regime, b, c) for n in nis)]
        if not present:
            continue
        span = len(present) + (1 if S else 0)
        first = True
        if S:
            # non-learned reference; constant by construction, spread is exactly 0
            rows.append(f'<tr class="ref"><td rowspan={span}>{b}</td>'
                        f'<td>shortest path (BFS)</td>'
                        + "".join(f'<td class="num">{FMT[met].format(S[met])}</td>'
                                  for _ in nis)
                        + '<td class="num">0</td><td class="num"></td></tr>')
            first = False
        for c, lab, fam in present:
            vals = [(m.get(met) if (m := get(TAB[n], regime, b, c)) else None)
                    for n in nis]
            good = [v for v in vals if v is not None]
            bestv = ((max if direction > 0 else min)(good)
                     if good and direction != 0 else None)
            cells = ""
            for v in vals:
                if v is None:
                    cells += '<td class="num muted">&ndash;</td>'
                else:
                    s = FMT[met].format(v)
                    cells += (f'<td class="num"><b>{s}</b></td>' if v == bestv
                              else f'<td class="num">{s}</td>')
            sp = spread(vals)
            mk, _, _ = trend(vals)
            note = ' title="n_is 를 쓰지 않는 arm"' if c in NIS_FREE else ""
            cls = "invar" if c in NIS_FREE else fam
            rows.append(f'<tr class="{cls}">'
                        + (f"<td rowspan={span}>{b}</td>" if first else "")
                        + f'<td{note}>{lab}</td>{cells}'
                        f'<td class="num">{FMT[met].format(sp)}</td>'
                        f'<td class="num">{mk}</td></tr>')
            first = False
    return "\n".join(rows)


def decomp_table(T, nis, blks, regime, arm):
    """valid / arrival / v&a for one arm, to locate WHERE an n_is gain comes from.

    IW-only is the arm that still moves with n_is, and the whole movement sits in
    `valid`: with no adjacency mask, drawing more candidates raises the chance that
    at least one is a legal transition, so extra samples buy legality by luck.
    Arrival is flat, so it is not a "better planning" effect.
    """
    rows = ['<tr><th>blk</th><th>지표</th>'
            + "".join(f'<th class="num">n={n}</th>' for n in nis if n in T)
            + '<th class="num">범위</th></tr>']
    use = [n for n in nis if n in T]
    for b in blks:
        first = True
        for met, lab in (("valid", "valid"), ("arrival", "arrival"),
                         ("valid_and_arrival", "v&amp;a")):
            vs = [(m.get(met) if (m := get(T[n], regime, b, arm)) else None)
                  for n in use]
            if not any(v is not None for v in vs):
                continue
            cells = "".join(f'<td class="num">{v:.3f}</td>' if v is not None
                            else '<td class="num muted">&ndash;</td>' for v in vs)
            rows.append(f'<tr class="iw">'
                        + (f"<td rowspan=3>{b}</td>" if first else "")
                        + f'<td>{lab}</td>{cells}'
                        f'<td class="num">{spread(vs):.3f}</td></tr>')
            first = False
    return "\n".join(rows)


def diversity_table(D, nis):
    """uniq / ESS vs n_is: the structural half of the sweep."""
    if not D:
        return ""
    blks = sorted({r["blk"] for r in D})
    idx = {(r["blk"], r["n_is"], r["adj_prop"]): r for r in D}
    rows = ['<tr><th rowspan=2>blk</th><th rowspan=2>adjacency<br>마스킹</th>'
            f'<th class="num" colspan={len(nis)}>고유 후보 수</th>'
            f'<th class="num" colspan={len(nis)}>ESS / n</th></tr>'
            + "<tr>" + "".join(f'<th class="num">n={n}</th>' for n in nis) * 2
            + "</tr>"]
    for b in blks:
        for adjp in (True, False):
            if not any((b, n, adjp) in idx for n in nis):
                continue
            u = "".join(
                f'<td class="num">{idx[(b, n, adjp)]["uniq"]:.1f}</td>'
                if (b, n, adjp) in idx else '<td class="num muted">&ndash;</td>'
                for n in nis)
            e = "".join(
                f'<td class="num">{idx[(b, n, adjp)]["ess_frac"]:.3f}</td>'
                if (b, n, adjp) in idx else '<td class="num muted">&ndash;</td>'
                for n in nis)
            rows.append(f'<tr class="{"iw" if adjp else ""}">'
                        f'<td>{b}</td><td>{"ON" if adjp else "OFF"}</td>{u}{e}</tr>')
    return "\n".join(rows)


def write_csv(TAB, nis, blks, out):
    fields = ["post", "regime", "blk", "arm", "family", "n_is", "va", "valid",
              "arrival", "dtw_sum", "lcs", "dtw_per_pair", "lcs_norm",
              "gen_len", "ref_len"]
    rows = []
    for post, _ in POSTS:
        if post not in TAB:
            continue
        for regime in ("seen", "unseen"):
            for b in blks:
                for c, _, fam in ARMS:
                    for n in nis:
                        m = get(TAB[post][n], regime, b, c) if n in TAB[post] else None
                        if not m:
                            continue
                        rows.append({
                            "post": post, "regime": regime, "blk": b, "arm": c,
                            "family": fam, "n_is": n,
                            "va": round(m.get("valid_and_arrival", float("nan")), 6),
                            "valid": round(m.get("valid", float("nan")), 6),
                            "arrival": round(m.get("arrival", float("nan")), 6),
                            "dtw_sum": round(m["dtw_sum"], 3) if "dtw_sum" in m else "",
                            "lcs": round(m["lcs"], 4) if "lcs" in m else "",
                            "dtw_per_pair": round(m["dtw"], 4) if "dtw" in m else "",
                            "lcs_norm": round(m["lcs_norm"], 6) if "lcs_norm" in m else "",
                            "gen_len": round(m["gen_len"], 2) if "gen_len" in m else "",
                            "ref_len": round(m["ref_len"], 2) if "ref_len" in m else "",
                        })
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"written {out}  ({len(rows)} rows)")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    ap.add_argument("-nis", type=str, default="10,30,50,100,150")
    ap.add_argument("-blocks", type=str, default="")
    ap.add_argument("-regime", type=str, default="both",
                    choices=["both", "seen", "unseen"],
                    help="both = seen/unseen 둘 다 (기본). seen|unseen = 그 체제만 담은 "
                         "문서를 nis_sweep_{V}_{regime}_ko.html 로 쓴다. base 와 "
                         "Adj-only 는 판별자와 무관해 unseen 실행분이 없으므로 seen "
                         "실행분을 공유한다(원래 프로토콜).")
    ap.add_argument("-post", type=str, default="rawL",
                    help="주 표에 쓰는 후처리 변형. 기본 rawL = simplification 만 "
                         "(샘플러가 지금 기본으로 하는 것). 부록에 P1P3L 도 싣는다.")
    args = ap.parse_args()

    V = args.variant
    RES = join(ROOT, f"sets_v6/{V}/res")
    nis = [int(x) for x in args.nis.split(",")]

    # TAB[post][n] -> table json
    TAB, missing = {}, []
    for post, sfx in POSTS:
        d = {}
        for n in nis:
            p = join(RES, f"va_table_v6{V}_nis{n}{sfx}.json")
            j = jload(p, required=False)
            if j is None:
                missing.append(os.path.basename(p))
            else:
                d[n] = j
        if d:
            TAB[post] = d
    if args.post not in TAB:
        raise SystemExit(f"주 표 변형 '{args.post}' 이 없다. 있는 것: {sorted(TAB)}")
    if missing:
        print(f"[build] missing tables ({len(missing)}): {missing}")
    MAIN = TAB[args.post]
    have_nis = [n for n in nis if n in MAIN]

    blks = ([int(x) for x in args.blocks.split(",")] if args.blocks else
            sorted({int(m.group(1)) for n in have_nis for k in MAIN[n]
                    if (m := re.search(r"_blk(\d+)_", k))}))
    RG = "unseen" if args.regime == "unseen" else "seen"
    BOTH = args.regime == "both"
    rg_label = {"seen": "seen (e0 판별자)",
                "unseen": "unseen (e99, except_0 미학습)"}[RG]
    D = jload(join(RES, f"nis_diversity_v6{V}.json"), required=False)
    S = jload(join(RES, f"shortest_v6{V}.json"), required=False)
    print(f"[build] variant={V}  post={args.post}  regime={args.regime}  "
          f"n_is={have_nis}  blocks={blks}")

    ncsv = write_csv(TAB, nis, blks, join(RES, "nis_plot_data.csv"))

    # ---- headline: how flat is it, in numbers the reader can check ------------
    def flat_summary(regime, arm):
        out = {}
        for met, _, _ in MET:
            sps = [spread([(m.get(met) if (m := get(MAIN[n], regime, b, arm)) else None)
                           for n in have_nis]) for b in blks]
            sps = [x for x in sps if x == x]
            out[met] = max(sps) if sps else float("nan")
        return out

    fl_iw = flat_summary(RG, "adj+modelD")
    fl_io = flat_summary(RG, "modelD")
    # determinism check on the two n_is-free arms
    inv_bad = []
    for arm in sorted(NIS_FREE):
        for b in blks:
            for met, _, _ in MET:
                if spread([(m.get(met) if (m := get(MAIN[n], RG, b, arm)) else None)
                           for n in have_nis]) not in (0.0, float("nan")):
                    inv_bad.append((arm, b, met))
    inv_txt = ("두 arm 의 모든 셀이 <b>n 에 걸쳐 완전히 동일</b>하다 &mdash; "
               "시드 고정과 결정성이 확인된다."
               if not inv_bad else
               f"<b class='bad'>경고: {len(inv_bad)}개 셀이 n 에 따라 달라진다</b> "
               f"({inv_bad[:4]}&hellip;) &mdash; n_is 를 쓰지 않는 arm 이므로 "
               f"이는 버그 신호다.")

    # Cells where n_is genuinely moves the metric: a range alone is not enough,
    # so require BOTH a range above 2x the v&a sampling SE AND a strictly one-way
    # trend. Listed explicitly so a flat headline cannot bury a real exception.
    N_EVAL = 1000
    SE2 = 2 * (0.25 / N_EVAL) ** 0.5      # 2 x worst-case binomial SE ~= 0.032
    real = []
    for b in blks:
        for c, lab, _ in ARMS:
            if c in NIS_FREE:
                continue
            vs = [(m.get("valid_and_arrival") if (m := get(MAIN[n], RG, b, c))
                   else None) for n in have_nis]
            mk, _, _ = trend(vs)          # trend() returns (marker, n_up, n_down)
            sp = spread(vs)
            # Monotonicity is the primary evidence, magnitude only secondary. With
            # 5 points, all 4 steps landing in one nominated direction has
            # probability 1/16 under pure noise, so a strictly one-way row is
            # informative even when its range sits under 2x SE -- filtering on the
            # range alone hid exactly such a cell (blk4 Adj+IW) at first.
            if mk in ("&uarr;", "&darr;") and sp == sp:
                real.append((b, lab, vs[0], vs[-1], sp, mk, sp > SE2))
    real.sort(key=lambda r: -r[4])
    if real:
        items = "".join(
            f"<li><b>blk{b} {lab}</b>: {a:.3f} &rarr; {z:.3f} &mdash; "
            f"{'상승' if mk == '&uarr;' else '하락'} 4/4, 범위 {s:.3f}"
            f"{' <b>(표본오차 2배 초과)</b>' if big else ' (표본오차 2배 이내)'}</li>"
            for b, lab, a, z, s, mk, big in real)
        n_big = sum(1 for r in real if r[6])
        real_note = (f'<p><b>n 이 실제로 움직이는 셀.</b> 5개 점의 <b>4개 구간이 모두 같은 '
                     f'방향</b>인 행만 골랐다 &mdash; 순수 잡음이라면 한 방향으로 정렬될 확률이 '
                     f'1/16 이므로, 범위가 작아도 정보가 있다. 범위가 '
                     f'{SE2:.3f}(표본오차 2배)를 넘는 셀은 {n_big}개다.</p><ul>{items}</ul>'
                     f'<p class="small">거꾸로, 범위가 커도 방향이 혼재(&sim;)면 잡음으로 읽어야 '
                     f'한다 &mdash; 표에서 <b>범위와 추세를 반드시 함께</b> 볼 이유다.</p>')
    else:
        real_note = ('<p class="small">4개 구간이 모두 같은 방향인 셀은 <b>없다</b> &mdash; '
                     '이 체제에서 n_is 효과는 어느 블록에서도 방향성이 없다.</p>')

    div_note = ""
    if D:
        dd = {(r["blk"], r["n_is"]): r for r in D if r["adj_prop"]}
        b0 = min(blks)
        lo, hi = min(have_nis), max(have_nis)
        if (b0, lo) in dd and (b0, hi) in dd:
            div_note = (f"blk{b0} 에서 n 을 <b>{lo}&rarr;{hi}</b>({hi // lo}배)로 "
                        f"늘려도 고유 후보는 "
                        f"<b>{dd[(b0, lo)]['uniq']:.1f}&rarr;{dd[(b0, hi)]['uniq']:.1f}</b> "
                        f"밖에 안 늘고, ESS/n 은 "
                        f"{dd[(b0, lo)]['ess_frac']:.3f}&rarr;{dd[(b0, hi)]['ess_frac']:.3f} 다. ")

    css = re.search(r"<style>(.*?)</style>",
                    open(join(ROOT, "report_src/iclr_iw_exp_ko.html")).read(),
                    re.S).group(1)

    def mt(met, lab, d, regime=None, use=None):
        regime = regime or RG
        T = use or MAIN
        return (f"<h3>{lab} &mdash; {regime}</h3><table>"
                + metric_table(T, [n for n in nis if n in T], blks, regime, met, d,
                               S if met in ("valid_and_arrival", "dtw_sum",
                                            "lcs", "gen_len") else None)
                + "</table>")

    main_tables = "\n".join(mt(m, lab, d) for m, lab, d in MET)
    unseen_tables = ("\n".join(mt(m, lab, d, "unseen") for m, lab, d in MET)
                     if BOTH else "")
    app_tables = ""
    if "P1P3L" in TAB:
        app_tables = "\n".join(
            mt(m, lab, d, RG, TAB["P1P3L"]) for m, lab, d in MET)
    raw_tables = "\n".join(mt(m, lab, d, RG, TAB["raw"]) for m, lab, d in MET) \
        if "raw" in TAB else ""

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Porto v6 {V} — importance sample 개수 n_is 스윕</title><style>{css}
tr.iw td {{ background: #f4f8fb; }}
tr.invar td {{ background: #f7f7f7; color: #666; }}
tr.ref td {{ background: #f2f2f0; color: #555; font-style: italic; }}
b.bad {{ color: #c33; }}
</style></head><body>
<h1>Porto v6 {V} &mdash; importance sample 개수 <span class="mono">n_is</span> 스윕</h1>

{H2("무엇을 왜 재는가")}
<p>IW 가이던스는 매 reveal 마다 mean-field proposal 에서 <span class="mono">n_is</span> 개의
후보 블록을 뽑고, 판별자 가중치 <span class="mono">(D/(1&minus;D))<sup>&gamma;</sup></span>
로 재가중한 뒤 위치 하나를 공개한다. <span class="mono">n_is</span> 는
<b>연산량을 그대로 결정하는 유일한 하이퍼파라미터</b>다 &mdash; 판별자 forward 가
<span class="mono">batch &times; n_is</span> 회 돈다.</p>
<p>이 스윕을 돌린 이유는 <b>사전 진단이 평평함을 예측했기</b> 때문이다.
<span class="mono">tools/diag/diag_proposal_diversity.py</span> 로 재 보니, adjacency
마스킹을 켠 상태에서 <span class="mono">n_is=100</span> 개를 뽑아도 <b>서로 다른</b> 후보는
blk&le;4 에서 <b>2.6&ndash;3.9 개</b>뿐이었다. 시나리오 그래프의 out-degree 가 평균 2.74 라
reveal 위치의 합법 후속이 애초에 3 개 정도이기 때문이다. 그렇다면 지표는
<span class="mono">n_is</span> 에 <b>평평해야</b> 하고, 그건 곧 가이던스를 훨씬 싸게
&mdash; 나아가 <b>전수 열거</b>로 &mdash; 바꿀 수 있다는 뜻이다.
평평함 자체가 결과다.</p>
<table>
<tr><th>축</th><th>값</th><th>비고</th></tr>
<tr><td><span class="mono">n_is</span></td><td>{", ".join(map(str, have_nis))}</td>
 <td>연산량 &prop; n_is. n=100 은 gate-D 실행분을 그대로 재사용</td></tr>
<tr><td>블록</td><td>{", ".join(map(str, blks))}</td>
 <td>blk1&ndash;2 = 최적 구간, blk16&ndash;64 = 후보 다양성이 실제로 생기는 구간</td></tr>
<tr><td>arm</td><td>base / Adj-only / IW-only / Adj+IW</td>
 <td>앞의 둘은 후보를 뽑지 않으므로 <b>n 에 불변</b> &mdash; 결정성 점검용</td></tr>
<tr><td>regime</td><td>seen (e0) / unseen (e99)</td>
 <td>base 와 Adj-only 는 판별자와 무관해 seen 실행분을 공유</td></tr>
<tr><td>후처리</td><td>raw / rawL / P1P3 / P1P3L</td>
 <td>주 표는 <b>{args.post}</b>. 다른 변형은 부록</td></tr>
<tr><td>평가</td><td colspan=2 class="ctr">동일 1,000 OD 쌍 (SPLIT_SEED 777),
 <span class="mono">order=l2r</span>, <span class="mono">w_gamma=1.0</span>,
 first-hitting 스케줄</td></tr>
</table>

{H2("한눈에")}
<p>{div_note}따라서 <b>지표는 n 에 평평할 것으로 예측했고, 실제로 평평하다.</b>
<span class="mono">{args.post}</span> / {RG} 에서 n 을
{min(have_nis)}&ndash;{max(have_nis)} 로 바꿀 때 블록별 최대 변동폭은</p>
<table>
<tr><th>arm</th><th class="num">v&amp;a</th><th class="num">DTW (m)</th>
 <th class="num">LCS</th><th class="num">gen_len</th></tr>
<tr class="iw"><td>Adj+IW</td>
 <td class="num">{fl_iw["valid_and_arrival"]:.3f}</td>
 <td class="num">{fl_iw["dtw_sum"]:.0f}</td>
 <td class="num">{fl_iw["lcs"]:.2f}</td>
 <td class="num">{fl_iw["gen_len"]:.1f}</td></tr>
<tr class="iw"><td>IW-only</td>
 <td class="num">{fl_io["valid_and_arrival"]:.3f}</td>
 <td class="num">{fl_io["dtw_sum"]:.0f}</td>
 <td class="num">{fl_io["lcs"]:.2f}</td>
 <td class="num">{fl_io["gen_len"]:.1f}</td></tr>
</table>
<p class="small">비교 기준 &mdash; v&amp;a 의 짝지지 않은 표본 오차는
<span class="mono">sqrt(p(1&minus;p)/1000) &asymp; 0.013</span> 이다. 변동폭이 그 수준이면
<b>n 의 효과가 아니라 몬테카를로 잡음</b>이다.</p>
<p class="small"><b>결정성 점검.</b> base 와 Adj-only 는
<span class="mono">n_is</span> 를 소비하지 않는다(base 는
<span class="mono">plan_guided</span> 를 호출조차 하지 않고, Adj-only 는
<span class="mono">disc=None</span> 으로 마스킹된 marginal 에서 곧바로 공개한다).
{inv_txt}</p>
{real_note}

{H2(f"결과 &mdash; {args.post}, {rg_label}")}
<p class="small">굵은 값이 그 행에서 최고. <b>범위</b> = n 에 걸친 최대&minus;최소,
<b>추세</b> = n 증가 방향(&uarr;/&darr; 전 구간 일방, &nearr;/&searr; 다수, &sim; 혼재).
범위만 보면 잡음과 실제 추세를 못 가르므로 <b>두 열을 함께</b> 읽어야 한다.
회색 이탤릭 = 학습 없는 shortest path (BFS) 참조선, 회색 = n 에 불변인 arm.</p>
{main_tables}

{f'{H2(f"결과 &mdash; {args.post}, unseen (e99, except_0 미학습)")}' + unseen_tables if BOTH else ""}

{H2("후보 다양성 &mdash; 왜 평평한가")}
<p>지표가 평평한 이유는 <b>n 을 늘려도 후보 집합이 자라지 않기</b> 때문이다.
아래는 <span class="mono">plan_guided</span> 의 <span class="mono">diag_log</span> 에서
reveal 마다 기록한 <b>서로 다른 후보 블록 수</b>와 가중치 ESS 다
(<span class="mono">tools/diag/diag_proposal_diversity.py</span>, OD 100 쌍).</p>
{f"<table>{diversity_table(D, nis)}</table>" if D else
 "<p class='small muted'>진단 JSON 이 없다 &mdash; scripts/run_v6_nis.sh 단계 C 를 돌릴 것.</p>"}
<p><b>이 표가 스윕 전 예측을 절반만 확인한다.</b> 두 구간을 나눠 읽어야 한다.</p>
<ul>
<li><b>작은 블록(1&ndash;2) &mdash; proposal 한계.</b> 예측대로다. 마스킹 ON 에서 고유 후보가
 n=10 에서 1.6&ndash;1.8, n=150 에서도 <b>2.9&ndash;4.2</b> 에 머문다. 시나리오 그래프
 out-degree(평균 2.74)가 상한이라 n 을 15 배 늘려도 후보 집합이 자라지 않는다.
 <b>여기서 지표가 평평한 것은 후보가 없어서다.</b></li>
<li><b>큰 블록(16&ndash;64) &mdash; 판별자 한계.</b> 여기서는 후보가 <b>실제로</b> 늘어난다:
 blk64 에서 5.1 &rarr; <b>59.8</b> 개, 포화 조짐이 없다. 그런데도 Adj+IW 의 지표는
 여전히 평평하다(v&amp;a 범위 0.010). ESS/n 도 0.94 &rarr; 0.90 로 겨우 내려간다.
 <b>즉 후보 60 개를 줘도 판별자가 그것들을 갈라내지 못한다.</b> 원인이 다르므로
 처방도 다르다.</li>
</ul>
<p class="small">이 구분이 중요하다. n=100 한 점만 보고 있었을 때는 &ldquo;후보 다양성 부재&rdquo;
하나로 설명했지만, 스윕을 보면 <b>작은 블록에서만</b> 그 설명이 맞다. 큰 블록의 평평함은
판별자가 배포 과제(같은 <span class="mono">p&theta;</span> 가 같은 스텝에 낸 후보들의 순위)
에 둔감하다는 증거다 &mdash; 전역 val acc 0.69&ndash;0.73 은 건강하지만 그건 훨씬 쉬운 과제다.</p>

{H2("연산량 &mdash; n_is 는 사실상 무료다")}
<p>스윕을 돌리기 전에 벽시계 시간을 먼저 쟀다(blk2, seen, OD 100 쌍, 4 arm &times;
4 후처리 변형 전부 포함, 단일 GPU):</p>
<table>
<tr><th class="num">n_is</th><th class="num">wall</th><th>비고</th></tr>
<tr><td class="num">10</td><td class="num">459 s</td><td rowspan=2 class="ctr">
 후보 <b>15 배</b>에 시간 <b>+3.3%</b></td></tr>
<tr><td class="num">150</td><td class="num">474 s</td></tr>
</table>
<p>즉 이 규모에서 병목은 판별자 배치가 아니라 <b>reveal 마다 도는 backbone forward</b>와
CPU 채점이다. <span class="mono">batch &times; n_is</span> = 100&times;150 = 15,000 개
시퀀스도 GPU 를 다 못 채운다. 그러므로 <b>&ldquo;n_is 를 줄여서 빨라진다&rdquo;는 주장은
성립하지 않는다</b> &mdash; FLOP 은 줄지만 시간은 그대로다. 이 스윕의 값어치는 속도가
아니라 아래 세 가지다.</p>

{H2("IW-only 는 왜 아직 n 에 반응하는가")}
<p>모든 arm 중 <b>IW-only(마스킹 없음)</b> 만 n 에 뚜렷하게 움직인다 &mdash; v&amp;a 범위
{fl_io["valid_and_arrival"]:.3f}, 그리고 <b>단조 증가</b>다. 그 증가분이 어디서 오는지
분해하면 답이 분명하다.</p>
<table>{decomp_table(MAIN, nis, blks, RG, "modelD")}</table>
<p><b>전부 <span class="mono">valid</span> 에서 온다. arrival 은 평평하다.</b>
마스킹이 없으면 후보를 많이 뽑을수록 <b>그 중 하나가 우연히 합법일 확률</b>이 올라간다.
즉 n_is 가 사 주는 것은 &ldquo;더 나은 계획&rdquo;이 아니라 <b>합법성을 운으로 맞추는 것</b>이다.
Adj+IW 는 마스킹이 합법성을 이미 보장하므로 그 이득이 존재할 자리가 없고, 남은 일은
합법 후보들 사이의 순위 매기기뿐이라 곧바로 포화한다.</p>
<p class="small">그리고 이렇게 n 을 15 배 써서 얻은 IW-only 의 v&amp;a
({min(have_nis)}&rarr;{max(have_nis)}) 조차 Adj+IW 가 <b>n=10</b> 으로 내는 값에
한참 못 미친다. 마스킹은 n_is 로 대체할 수 있는 것이 아니다.</p>

{H2("따라서 무엇을 할 수 있는가")}
<ul>
<li><b>작은 블록에서는 전수 열거로 바꿔야 한다.</b> 이게 이 스윕의 핵심 함의다.
 마스킹 ON 의 합법 후보 공간이 blk2 에서
 <span class="mono">&asymp;2.74&sup2; &asymp; 7.5</span> 개뿐이므로, 100 개를 뽑는 대신
 <b>전부 세면</b> SNIS 가 추정이 아니라 <b>정확 계산</b>이 되어 몬테카를로 오차가
 0 이 된다. 위 표대로 시간은 어차피 n 에 둔감하니 <b>공짜로 정확해지는</b> 셈이다.
 지금은 100 번 뽑아 같은 3&ndash;4 개를 중복으로 세고 있다.</li>
<li><b>D-CBG 와의 비교에서 n_is 는 변호할 필요가 없는 축이다.</b> D-CBG exact 는 매 스텝
 전체 vocab({1387}) 을 훑는다. 우리가 n=10 으로도 같은 성능을 내는 것이 확인됐으므로,
 &ldquo;IW 는 n_is 를 크게 잡아야 작동한다&rdquo; 는 반론이 봉쇄된다. 리뷰어가
 <span class="mono">n_is</span> 를 튜닝 자유도로 지적할 여지가 없다.</li>
<li><b>큰 블록의 병목은 n_is 가 아니라 판별자다.</b> blk64 에서 후보를 60 개까지
 다양화해도 지표가 안 움직인다. 그러니 여기서 할 일은 n 을 더 키우는 게 아니라
 <b>판별자를 배포 과제로 학습</b>시키는 것이다 &mdash; 지금 학습 대상은 &ldquo;except 데이터
 vs 무조건 모델 pool&rdquo; 인데, 실제로 하는 일은 &ldquo;같은 스텝의 후보들 순위 매기기&rdquo;로
 훨씬 어렵다. 다만 성능 최적 블록이 1&ndash;2 라 이 방향의 실익은 제한적이다.</li>
</ul>
<p class="small"><b>남는 위험.</b> 이 스윕은 <span class="mono">n_is</span> 가 튜닝
자유도가 아님을 보이지만, 동시에 <b>IW 재가중이 마스킹 위에서 기여하는 폭이 좁다</b>는
것도 보여준다 &mdash; Adj-only 대비 Adj+IW 의 이득이 blk2 v&amp;a 0.777&rarr;0.799 이고,
그 이득이 n 에 무관하다. 즉 이득의 출처는 후보 개수가 아니라 판별자 신호 자체이며,
그것을 키우는 것이 다음 과제다.</p>

<hr>
<p class="small"><b>파이프라인.</b> A <span class="mono">scripts/run_v6_nis.sh</span>
(<span class="mono">tools/eval/three_way_postproc.py -n_is N</span>, 태그
<span class="mono">v6{V}{{B}}seenN{{N}}</span> / <span class="mono">v6{V}{{B}}unsN{{N}}</span>
&mdash; gate-D 레코드를 덮지 않는다) &rarr;
B <span class="mono">postproc_loopcut.py</span> + <span class="mono">collect_va_table.py -shape 1</span>
&rarr; C <span class="mono">diag_proposal_diversity.py -nis_list</span>.
n=100 은 재실행하지 않고 gate-D 태그에서 수집한다.
plot 용 CSV: <span class="mono">sets_v6/{V}/res/nis_plot_data.csv</span> ({ncsv} 행).</p>
{f'{H2(f"부록 &mdash; raw (후처리 없음), {RG}")}' + raw_tables if raw_tables else ""}
{f'{H2(f"부록 &mdash; P1P3L (repair + simplification), {RG}")}' + app_tables if app_tables else ""}
</body></html>"""

    rsfx = "" if BOTH else f"_{args.regime}"
    out_html = join(ROOT, f"report_src/nis_sweep_{V}{rsfx}_ko.html")
    io.open(out_html, "w", encoding="utf-8").write(doc)
    print(f"written {out_html} ({len(doc)} bytes)")
    print(f"[flat] Adj+IW  max spread over n: " + "  ".join(
        f"{m}={fl_iw[m]:.3f}" for m, _, _ in MET))
    print(f"[flat] IW-only max spread over n: " + "  ".join(
        f"{m}={fl_io[m]:.3f}" for m, _, _ in MET))
    if inv_bad:
        print(f"[WARN] {len(inv_bad)} n-invariant cells differ across n: {inv_bad[:8]}")
    else:
        print("[ok] base / Adj-only identical across every n_is")
