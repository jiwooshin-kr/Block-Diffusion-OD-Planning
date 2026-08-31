"""
Porto v6 {LE|LP} — D-CBG(Schiff et al.) vs IW 가이던스 비교 보고서.

  python report_src/build_porto_LE_dcbg.py [-variant LE] [-nis 100]

입력 (모든 수치는 JSON/로그에서 파싱하며 손으로 옮겨 적은 값은 없다):
  sets_v6/{V}/res/va_table_v6{V}_dcbg.json   arm x 7블록 x seen/unseen  (run_v6_dcbg.sh 단계 C)
  sets_v6/{V}/res/va_table_v6{V}.json        IW arm 보충 (dcbg 표에 없을 때)
  sets_v6/{V}/res/ceiling_v6{V}.json         지표 천장 (정답이 확률적 표본이라)
  sets_v6/{V}/res/paired_*.json              짝지은 검정 (tools/collect/paired_test.py)
  sets_v6/{V}/log/dcbgeval*_job*.log         D-CBG arm 의 invE / remE
  sets_v6/{V}/log/eval_job*.log              IW arm 의 invE / remE
출력:
  report_src/porto_{V}_dcbg_ko.html  ->  pdfs/Porto_{V}/porto_{V}_dcbg_ko.pdf
  PDF 변환:
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
      --no-pdf-header-footer --print-to-pdf=pdfs/Porto_LE/porto_LE_dcbg_ko.pdf \
      "file://$PWD/report_src/porto_LE_dcbg_ko.html"

arm 은 표에서 자동 발견한다 — gamma 스윕(`dcbg@g3`)이나 1차 근사(`dcbgfo`)처럼
나중에 붙는 arm 이 이 파일을 고치지 않아도 그대로 실린다.
"""

import argparse
import glob
import io
import json
import os
import re
from os.path import dirname, join

ROOT = dirname(dirname(os.path.abspath(__file__)))
BLKS = [1, 2, 4, 8, 16, 32, 64]

IW_ARMS = [("base", "base", "none"), ("adjonly", "Adj-only", "iw"),
           ("modelD", "IW-only", "iw"), ("adj+modelD", "Adj+IW", "iw")]

# Non-learned reference row (tools/eval/shortest_baseline.py), rendered directly
# under `base`. Its numbers do not depend on block size, regime, or the
# post-processing variant, so the same record is repeated in every table; it is
# excluded from the per-metric "best" bolding because it is not a competing method.
SHORTEST_CFG = "__shortest__"
SHORTEST_LAB = "shortest path (BFS)"
SHORTEST = [None]

# (json key, 표시 라벨, 방향(+1=클수록 좋음), ceiling json key 또는 None)
# 참조가 필요 없는 valid/arrival/v&a 를 먼저, 그 다음 soft 형상 지표(DTW/LCS),
# 마지막에 len-match(과거 표의 'EM'). EM 은 경로 동일이 아니라 valid & 길이 일치이므로
# 이름을 고쳤다 — 과거 표와의 연속성 때문에 지우지 않고 병기한다.
# 메인 지표는 3개 — valid&arrival(실행 가능성), DTW(평균 기하 오차), LCS/|ref|(순서 보존 커버리지).
# valid / arrival 은 v&a 의 구성 요소라 참고로 같이 싣고, gen_len 은 LCS 가 recall 이라
# 길이 팽창에 관대한 점을 독자가 항상 함께 보도록 붙인다(참조 평균 27.6).
# 'len-match'(구 EM)는 제외했다 — 이유는 아래 지표 절에 적었다.
# Ceiling columns are deliberately absent. The ceiling is measured on the OD subset
# that has >= 2 draws, whose reference paths are much shorter (19.9 vs 28.3 hops on
# LE), so a ceiling in sum/count units is not comparable with these tables.
MET = [("valid", "valid", +1, None), ("arrival", "arrival", +1, None),
       ("valid_and_arrival", "valid &amp; arrival", +1, None),
       ("dtw_sum", "DTW&darr; (m)", -1, None), ("lcs", "LCS", +1, None),
       ("gen_len", "gen_len", 0, None)]

MET_LABEL = {"valid_and_arrival": "v&amp;a", "len_match": "len-match",
             "dtw_sum": "DTW&darr; (m)", "dtw": "DTW/pair&darr;",
             "lcs": "LCS", "lcs_norm": "LCS/|ref|"}
MET_ORDER = {"valid_and_arrival": 0, "dtw_sum": 1, "lcs": 2,
             "dtw": 3, "lcs_norm": 4, "len_match": 5}

# 조건부 절이 빠져도 번호가 비지 않도록 h2 번호를 자동 증가시킨다.
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
                             f"(run_v6_dcbg.sh 단계 C 가 끝나야 생성된다)")
        return None


def get(J, regime, blk, cfg):
    """unseen 에 없는 arm(base/adjonly)은 seen 실행분을 공유한다."""
    return J.get(f"{regime}_blk{blk}_{cfg}") or J.get(f"seen_blk{blk}_{cfg}")


# ---------------------------------------------------------------- arm 발견
def arm_label(cfg):
    """dcbg / dcbg+adjp / dcbgfo / dcbgfo+adjp [+ @gN] -> 사람이 읽는 이름."""
    base, _, g = cfg.partition("@")
    name = {"dcbg": "D-CBG", "dcbg+adjp": "D-CBG+Adj",
            "dcbgfo": "D-CBG-fo", "dcbgfo+adjp": "D-CBG-fo+Adj"}.get(base, base)
    gam = g[1:] if g.startswith("g") else (g or "1")
    return f"{name} (&gamma;={gam})"


def arm_sort_key(cfg):
    """마스킹 유무로 묶고 그 안에서 gamma 오름차순. exact 먼저, 1차 근사 나중."""
    base, _, g = cfg.partition("@")
    gam = float(g[1:]) if g.startswith("g") else 1.0
    return (1 if "fo" in base else 0, 1 if base.endswith("+adjp") else 0, gam)


def discover_arms(J):
    found = set()
    for k in J:
        m = re.match(r"^(?:seen|unseen)_blk\d+_(.+)$", k)
        if m:
            found.add(m.group(1))
    out = [(c, lab, fam) for c, lab, fam in IW_ARMS if c in found]
    dc = sorted((c for c in found if c.startswith("dcbg")), key=arm_sort_key)
    return out + [(c, arm_label(c), "dcbg") for c in dc]


def present_arms(J):
    return [(c, lab, fam) for c, lab, fam in discover_arms(J)
            if any(get(J, r, b, c) for r in ("seen", "unseen") for b in BLKS)]


def make_pairs(arms):
    """헤드라인 표용 IW-vs-D-CBG 짝. 마스킹 유무를 맞춰 짝지고 gamma 별로 한 줄씩."""
    have = {c for c, _, _ in arms}
    out = []
    for c, lab, fam in arms:
        if fam != "dcbg":
            continue
        base = c.split("@")[0]
        iw = "adj+modelD" if base.endswith("+adjp") else "modelD"
        if iw in have:
            cond = "Lemma-3 마스킹" if base.endswith("+adjp") else "마스킹 없음"
            out.append((iw, c, f"{cond}<br>{lab}"))
    return out


# ---------------------------------------------------------------- invE / remE
LOGPAT = re.compile(
    r"^(?P<tag>v6\w+?(?P<blk>\d+)(?P<who>seen|uns)_(?P<cfg>[^ ]+?)_(?P<var>raw|P1|P3|P1P3))\s+"
    r"arr=(?P<arr>[\d.]+)\s+valid=(?P<valid>[\d.]+).*?invE=(?P<invE>[\d.]+)%"
    r".*?remE=(?P<remE>[\d.]+)%\s+patch=(?P<patch>[\d.]+)")


def scan_logs(V):
    """로그에서 지표를 긁는다. 재계산 없이 이미 찍힌 값을 쓴다.
      ER    {(regime, blk, cfg): (invE, remE)}          raw 변형
      PATCH {(regime, blk, cfg, variant): patch}        후처리가 추가한 평균 노드 수
    후처리는 노드를 추가만 하므로 patch = gen_len(변형) - gen_len(raw) 와 일치한다
    (교차검증: LE blk1 seen dcbg 로그 patch=6.05, gen_len 28.25 -> 34.30)."""
    ER, PATCH = {}, {}
    for pat in (f"sets_v6/{V}/log/dcbgeval*_job*.log", f"sets_v6/{V}/log/eval_job*.log"):
        for f in sorted(glob.glob(join(ROOT, pat))):
            for line in io.open(f, encoding="utf-8", errors="replace"):
                m = LOGPAT.match(line.strip())
                if not m:
                    continue
                regime = "seen" if m.group("who") == "seen" else "unseen"
                key = (regime, int(m.group("blk")), m.group("cfg"))
                PATCH[key + (m.group("var"),)] = float(m.group("patch"))
                if m.group("var") == "raw":
                    ER[key] = (float(m.group("invE")), float(m.group("remE")))
    return ER, PATCH


def patch_table(PATCH, ER, arms_present, regime="seen"):
    """후처리가 추가한 평균 노드 수. P1 = 불법 edge splice, P3 = 끝점 patch."""
    rows = ['<tr><th rowspan=2>구성</th><th rowspan=2>raw invE</th>'
            + "".join(f'<th class="num" colspan=1>blk {b}</th>' for b in BLKS) + "</tr>"
            + '<tr><th class="num" colspan=7>P1P3 합계 &nbsp;<span class="muted">'
            '(P1 splice + P3 끝점)</span></th></tr>']
    def _pt(b, c, var):
        return PATCH.get((regime, b, c, var), PATCH.get(("seen", b, c, var)))

    for c, lab, fam in arms_present:
        cells, any_ = "", False
        inv = _er(ER, regime, 2, c)
        for b in BLKS:
            t = _pt(b, c, "P1P3")
            p1 = _pt(b, c, "P1")
            p3 = _pt(b, c, "P3")
            if t is None:
                cells += '<td class="num muted">&ndash;</td>'
                continue
            any_ = True
            sub = (f'<br><span class="muted">{p1:.1f}+{p3:.1f}</span>'
                   if p1 is not None and p3 is not None else "")
            cells += f'<td class="num">{t:.1f}{sub}</td>'
        if any_:
            invc = (f'<td class="num">{inv[0]:.2f}%</td>' if inv
                    else '<td class="num muted">&ndash;</td>')
            rows.append(f'<tr class="{fam}"><td>{lab}</td>{invc}{cells}</tr>')
    return "\n".join(rows)


# ---------------------------------------------------------------- 표
def met_header(CEIL):
    """지표 헤더에 천장을 병기한다 — 정답이 확률적 표본이라 상한이 1/0 이 아니다."""
    out = ""
    for _, lab, _, ck in MET:
        sub = (f'<br><span class="muted">천장 {CEIL[ck]:.3f}</span>'
               if CEIL and ck and ck in CEIL else "")
        out += f'<th class="num">{lab}{sub}</th>'
    return out


def main_table(J, regime, arms, CEIL=None):
    rows = ["<tr><th>blk</th><th>구성</th>" + met_header(CEIL) + "</tr>"]
    for b in BLKS:
        grp = {c: v for c, v in ((c, get(J, regime, b, c)) for c, _, _ in arms) if v}
        if not grp:
            continue
        base = grp.get("base")
        # Computed before the reference row is inserted: BFS wins v&a by
        # construction, so letting it into `best` would take the bold away from
        # every model arm and hide which method actually leads.
        best = {m: ((max if d > 0 else min)(v[m] for v in grp.values() if m in v)
                    if d != 0 else None)
                for m, _, d, _ in MET}
        order = list(arms)
        if SHORTEST[0] is not None:
            grp[SHORTEST_CFG] = SHORTEST[0]
            i = next((k + 1 for k, (c, _, _) in enumerate(order) if c == "base"), 0)
            order.insert(i, (SHORTEST_CFG, SHORTEST_LAB, "ref"))
        first = True
        for c, lab, fam in order:
            if c not in grp:
                continue
            r = (f'<tr class="{fam}">'
                 + (f"<td rowspan={len(grp)}>{b}</td>" if first else "") + f"<td>{lab}</td>")
            first = False
            for m, _, _, _ in MET:
                v = grp[c].get(m)
                if v is None:
                    r += '<td class="num muted">&ndash;</td>'
                    continue
                if m == "gen_len":
                    r += f'<td class="num muted">{v:.1f}</td>'
                    continue
                sv = f"<b>{v:.3f}</b>" if v == best[m] else f"{v:.3f}"
                d = (f' <span class="muted">({v - base[m]:+.3f})</span>'
                     if base and c != "base" and m in base else "")
                r += f'<td class="num">{sv}{d}</td>' 
            rows.append(r + "</tr>")
    return "\n".join(rows)


def pair_table(J, arms_present, regime="seen"):
    """헤드라인: 같은 조건(마스킹 유무)에서 IW vs D-CBG 를 직접 맞붙인다."""
    use = make_pairs(arms_present)
    if not use:
        return ""
    rows = ['<tr><th rowspan=2>조건 / D-CBG 설정</th><th rowspan=2>blk</th>'
            '<th class="num" colspan=3>valid &amp; arrival</th>'
            '<th class="num" colspan=3>LCS/|ref|</th>'
            '<th class="num" colspan=3>DTW&darr;</th></tr>'
            '<tr>' + ('<th class="num">IW</th><th class="num">D-CBG</th>'
                      '<th class="num">차</th>') * 3 + '</tr>']
    for iw_c, dc_c, lab in use:
        ks = [b for b in BLKS if get(J, regime, b, iw_c) and get(J, regime, b, dc_c)]
        for i, b in enumerate(ks):
            x, y = get(J, regime, b, iw_c), get(J, regime, b, dc_c)
            r = f'<tr>{f"<td rowspan={len(ks)}>{lab}</td>" if i == 0 else ""}<td>{b}</td>'
            for k, sign in (("valid_and_arrival", +1), ("lcs_norm", +1), ("dtw", -1)):
                d = x[k] - y[k]
                cls = "win" if d * sign > 0 else ("lose" if d * sign < 0 else "")
                r += (f'<td class="num">{x[k]:.3f}</td><td class="num">{y[k]:.3f}</td>'
                      f'<td class="num {cls}">{d:+.3f}</td>')
            rows.append(r + "</tr>")
    return "\n".join(rows)


def paired_table(PJ):
    """짝지은 검정 표. 같은 1,000 OD 쌍이므로 독립 가정 SE 는 과대추정이다.
    이진 지표(v&a / len-match)는 정확 McNemar, 연속 지표(DTW / LCS)는 Wilcoxon
    signed-rank. 차이의 구간은 두 경우 모두 짝지은 부트스트랩.
    DTW 는 작을수록 좋으므로 우세 판정을 지표 방향에 맞춘다."""
    rows = ['<tr><th rowspan=2>비교</th><th rowspan=2>지표</th>'
            '<th rowspan=2>regime</th><th rowspan=2>blk</th>'
            '<th class="num" colspan=2>값</th><th class="num" rowspan=2>&Delta;</th>'
            '<th class="num" rowspan=2>95% CI</th>'
            '<th class="num" colspan=2>쌍 개수</th>'
            '<th class="num" rowspan=2>p</th></tr>'
            '<tr><th class="num">A</th><th class="num">B</th>'
            '<th class="num">A&lt;B</th><th class="num">A&gt;B</th></tr>']
    for J in PJ:
        if not J:
            continue
        lab = f'{J["arm_a"]}<br>vs {J["arm_b"]}'
        met = MET_LABEL.get(J.get("metric"), J.get("metric", "?"))
        ks = [k for k in (f"{r}_blk{b}" for r in ("seen", "unseen") for b in BLKS)
              if k in J["rows"]]
        for i, k in enumerate(ks):
            m = J["rows"][k]
            regime, blk = k.split("_blk")
            p_ = m["mcnemar_p"]
            low = bool(m.get("lower_better", False))
            better_a = (m["delta"] < 0) if low else (m["delta"] > 0)
            sig = ("win" if better_a else "lose") if p_ < 0.05 else ""
            lo, hi = m["ci95"]
            span = f" rowspan={len(ks)}"
            rows.append(
                f'<tr>{f"<td{span}>{lab}</td>" if i == 0 else ""}'
                f'{f"<td{span}>{met}</td>" if i == 0 else ""}'
                f'<td>{regime}</td><td class="num">{blk}</td>'
                f'<td class="num">{m["a"]:.3f}</td><td class="num">{m["b"]:.3f}</td>'
                f'<td class="num {sig}">{m["delta"]:+.3f}</td>'
                f'<td class="num">[{lo:+.3f}, {hi:+.3f}]</td>'
                f'<td class="num">{m["b_only"]}</td><td class="num">{m["c_only"]}</td>'
                f'<td class="num {sig}">{p_:.4f}</td></tr>')
    return "\n".join(rows)


def _er(ER, regime, b, c):
    """invE/remE lookup. base and Adj-only have no unseen run (they do not depend on
    the discriminator), so they fall back to the seen records, as in get()."""
    return ER.get((regime, b, c)) or ER.get(("seen", b, c))


def edge_table(ER, arms_present, regime="seen"):
    have = [c for c, _, _ in arms_present
            if any(_er(ER, regime, b, c) for b in BLKS)]
    if not have:
        return ""
    labs = {c: lab for c, lab, _ in arms_present}
    rows = ['<tr><th rowspan=2>blk</th>'
            + "".join(f'<th class="num" colspan=2>{labs[c]}</th>' for c in have) + "</tr>"
            + "<tr>" + '<th class="num">invE</th><th class="num">remE</th>' * len(have)
            + "</tr>"]
    for b in BLKS:
        cells, any_ = "", False
        for c in have:
            v = _er(ER, regime, b, c)
            if v:
                any_ = True
                cells += f'<td class="num">{v[0]:.2f}%</td><td class="num">{v[1]:.2f}%</td>'
            else:
                cells += '<td class="num muted">&ndash;</td><td class="num muted">&ndash;</td>'
        if any_:
            rows.append(f"<tr><td>{b}</td>{cells}</tr>")
    return "\n".join(rows)


def best_of(J, regime, fam, arms_present, metric="valid_and_arrival"):
    best, bb, bl = -1, None, None
    for c, lab, f in arms_present:
        if f != fam:
            continue
        for b in BLKS:
            m = get(J, regime, b, c)
            if m and m.get(metric, -1) > best:
                best, bb, bl = m[metric], b, lab
    return best, bb, bl


METRIC_SEC = r'''<p>참조 경로는 <b>C<sub>od</sub> 위 PSL 확률에서 뽑은 표본 하나</b>다. 따라서
&ldquo;참조와 정확히 같은가&rdquo;를 묻는 0/1 지표는 그럴듯한 대안 경로를 무의미한 경로와
똑같이 0점 준다. trajectory prediction / route choice 문헌의 관행대로
<b>관측된 경로 하나를 GT로 두고 soft 형상 지표</b>(DTW, LCS)로 재고,
실행 가능성은 <b>참조가 필요 없는</b> 지표로 따로 잰다.</p>
<table>
<tr><th>지표</th><th>정의</th><th>방향</th><th>참조 필요</th></tr>
<tr><td>valid</td><td>모든 연속 쌍이 시나리오 그래프의 edge</td>
 <td class="ctr">&uarr;</td><td class="ctr">아니오</td></tr>
<tr><td>arrival</td><td>생성 경로의 마지막 정점 = 목적지</td>
 <td class="ctr">&uarr;</td><td class="ctr">아니오</td></tr>
<tr><td>valid &amp; arrival</td><td>둘 다 &mdash; 실행 가능성의 <b>전제 조건</b> 지표.
 참조도 길이도 개입하지 않아 이 문서의 교란 요인 전부에서 자유롭다</td>
 <td class="ctr">&uarr;</td><td class="ctr">아니오</td></tr>
<tr><td><b>DTW (m)</b></td>
 <td>좌표 DTW. ground distance = <b>haversine 대권거리(미터)</b>, 표준 symmetric1 recursion
 <span class="mono">D[i,j] = c + min(D[i&minus;1,j], D[i,j&minus;1], D[i&minus;1,j&minus;1])</span>
 (Sakoe &amp; Chiba 1978), <b>정규화 없이 누적 합</b></td>
 <td class="ctr">&darr;</td><td class="ctr">예</td></tr>
<tr><td><b>LCS</b></td>
 <td>최장 공통 <b>부분수열</b> 길이(정점 id 기준, 교과서 DP). 순서를 지키며 겹치는 최대
 노드 수 &mdash; 집합 교집합이 아니고 연속일 필요도 없다. <b>정규화 없는 원값</b></td>
 <td class="ctr">&uarr;</td><td class="ctr">예</td></tr>
<tr><td>gen_len / ref_len<br><span class="small">(맥락용, 지표 아님)</span></td>
 <td>생성 / 참조 경로의 평균 노드 수</td>
 <td class="ctr">&mdash;</td><td class="ctr">&mdash;</td></tr>
</table>
<p class="small"><b>왜 정규화하지 않는가.</b> DTW 를 <span class="mono">max(n, m)</span> 으로 나누면
&ldquo;정렬 쌍당 평균&rdquo;이 되어 해석은 쉬워지지만, <b>분모가 생성 경로 길이에 의존</b>한다.
참조 근처에 머물면서 길게 가는 경로는 분자가 거의 늘지 않고 분모만 커져 <b>점수가 좋아진다</b>.
실측으로 드러났다: 루프 제거(순수하게 낭비만 삭제)가 <span class="mono">Adj+IW</span> 의
정규화 DTW 를 0.167 &rarr; 0.188 로 <b>악화</b>시켰지만 누적 합은 9.35 &rarr; 6.77 로 개선됐다.
누적 합은 참조 위에 머무는 패딩에 보상도 벌점도 주지 않고(추가 정렬 쌍의 비용이 &asymp; 0),
참조에서 벗어난 이탈만 청구한다. LCS 는 참조가 모든 arm 에서 동일하므로 원값이 이미 비교 가능하다.
다만 원값은 <span class="mono">E[l]</span> 이라 긴 참조 경로가 더 크게 기여하고,
<span class="mono">LCS/|ref|</span> = <span class="mono">E[l/r]</span> 은 OD 쌍마다 동등한 한 표를 준다
&mdash; 두 형태의 1위가 112개 셀 중 30&#37;에서 달랐으므로 사소한 차이가 아니다.
정규화 형태도 JSON(<span class="mono">dtw</span>, <span class="mono">lcs_norm</span>)에 남겨 두었다.</p>
<p class="small"><b>ground distance 를 haversine 으로 바꾼 이유.</b> 2026-08-21 이전에는
<span class="mono">(|&Delta;lat| + |&Delta;lng|) &times; 100</span> 이었다. 두 축에 같은 상수를 곱하므로
<b>비등방성이 전혀 보정되지 않는다</b> &mdash; Porto(위도 41.158&deg;)에서 1&deg; lat = 111.1 km 인데
1&deg; lng = 83.8 km 이므로 경도가 1.33배 과대평가된다. 게다가 단위가 없어 578 m 의 이격이
<span class="mono">0.787</span> 로 나온다. haversine 은 <span class="mono">cos&phi;<sub>1</sub>cos&phi;<sub>2</sub></span>
항이 그 축소를 공식 안에서 처리하므로 축별 km/도 상수가 필요 없고 결과가 <b>바로 미터</b>다.
차원별 미터 스케일링 후 유클리드를 쓴 것과 전체 1,999개 edge 에서 <b>0.59&#37; 이내</b>로 일치한다
(이 2.78 &times; 4.17 km 영역의 곡률). 실측으로 순위는 세 방식(도 L1 / L1 미터 / haversine)에서
동일했고(Pearson r = 0.999) 값의 <b>해석 가능성</b>만 달라졌다. 그래프 edge 평균 길이가
<b>84 m</b> 이므로 DTW 값을 &ldquo;링크 몇 개분 벗어났나&rdquo;로 읽을 수 있다.</p>
<p class="small"><b>구현과 외부 대조.</b> DTW/LCS 는 <span class="mono">main_bd.py</span> 의 직접 구현이다
(<span class="mono">dtw_distance</span> 는 <span class="mono">normalize</span>,
<span class="mono">ground</span> 인자로 형태를 고르며 구 정의도 <span class="mono">ground="deg_l1"</span>
로 재현된다). <span class="mono">tools/diag/verify_shape_metrics.py</span> 가 참조 경로 400쌍에서
독립 구현과 대조한다 &mdash; DTW 는 별도 코드 경로로 만든 대권거리 행렬 + 교과서 recursion 과
<b>오차 0</b>, 동일성 <span class="mono">d(a,a)=0</span> 확인, LCS 는 1행 롤링 DP 와 <b>불일치 0/400</b>,
JSEV 는 <span class="mono">scipy&hellip;jensenshannon&sup2;</span> 와 3e&minus;16 이내.
<b>대조되지 않는 것은 정규화/ground 관례 자체</b>이므로 이 문서의 DTW 값을 다른 논문의 DTW 와
직접 비교하면 안 된다 &mdash; 논문 내부 arm 간 비교 전용이다.</p>
<p class="small"><b>제외한 지표.</b> 과거 표의 &ldquo;EM&rdquo;(= <span class="mono">len-match</span>)은
세 가지 이유로 뺐다. (1) 경로 동일이 아니라 <b>valid &and; 길이 일치</b>다 &mdash; 전혀 다른 경로도
홉 수만 같으면 1이고, 구 데이터(v4)에서조차 같은 OD 의 최단경로가 유일한 경우는 170/1000 뿐이었다.
(2) <b>목표 OD 가 아니라 &ldquo;생성 경로가 실현한 OD&rdquo;</b> 로 판정한다
(<span class="mono">od = (path[0], path[-1])</span>). v6 는 OD 가 전체 순서쌍이라 어떤 끝점이든 참조가
존재하므로 <b>미도달 경로도 EM=1 을 받는다</b> &mdash; 실측(blk2 seen)에서
<span class="mono">Adj+IW</span> 는 EM=1 인 170개 중 5개(2.9&#37;)가 미도달이었다. 따라서 EM 은
v&amp;a 의 부분집합조차 아니다. (3) 천장이 0.611 이라 절대값 해석도 불가하다.
<span class="mono">PC</span> 도 뺐다 &mdash; 참조가 최단경로가 아니면 정답조차 PC &lt; 1 이 되어 상한이
1 이 아니다. (두 열 모두 레코드 CSV 에는 계산돼 남아 있다.)</p>
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    ap.add_argument("-nis", type=int, default=100)
    ap.add_argument("-table", type=str, default="",
                    help="합침 표 경로 직접 지정(드라이런용)")
    ap.add_argument("-regime", type=str, default="both",
                    choices=["both", "seen", "unseen"],
                    help="both = seen/unseen 둘 다 (기본). seen|unseen = 그 체제만 담은 문서. "
                         "base 와 Adj-only 는 판별자와 무관해 unseen 실행분이 없으므로 "
                         "seen 실행분을 공유한다(원래 프로토콜).")
    ap.add_argument("-paired", type=int, default=0,
                    help="1 = 짝지은 유의성 검정 절을 싣는다. 기본 0(생략). "
                         "검정 JSON 은 sets_v6/{V}/res/paired_*.json 에 남아 있고 "
                         "scripts/run_paired_battery.sh 로 갱신한다.")
    args = ap.parse_args()
    V = args.variant

    J = jload(args.table or join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_dcbg.json"))
    IWJ = jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}.json"), required=False)
    if IWJ:                      # dcbg 표에서 빠진 IW arm 을 보충
        for k, v in IWJ.items():
            J.setdefault(k, v)
    CEIL = jload(join(ROOT, f"sets_v6/{V}/res/ceiling_v6{V}.json"), required=False)
    SHORTEST[0] = jload(join(ROOT, f"sets_v6/{V}/res/shortest_v6{V}.json"), required=False)
    RAWL = jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_dcbg_rawL.json"), required=False)
    P13L = jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_dcbg_P1P3L.json"), required=False)
    P13 = jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_dcbg_P1P3.json"), required=False)
    if P13 and IWJ:
        IW13 = jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_P1P3.json"), required=False)
        if IW13:
            for k, v in IW13.items():
                P13.setdefault(k, v)
    PJ = ([jload(f, required=False) for f in
           sorted(glob.glob(join(ROOT, f"sets_v6/{V}/res/paired_*.json")))]
          if args.paired else [])
    PJ = [x for x in PJ if x]
    # 메인 3지표만 표에 싣는다. len-match 는 지표에서 제외했으므로 개수만 각주로 밝힌다.
    MAIN_METS = {"valid_and_arrival", "dtw_sum", "lcs"}
    lm = [x for x in PJ if x.get("metric") == "len_match"]
    lm_a = lm_b = 0
    for x in lm:
        for m in x["rows"].values():
            if m["mcnemar_p"] < 0.05:
                if m["delta"] > 0:
                    lm_a += 1
                else:
                    lm_b += 1
    PJ = [x for x in PJ if x.get("metric") in MAIN_METS]
    PJ.sort(key=lambda x: (x.get("arm_b", ""), MET_ORDER.get(x.get("metric"), 9)))
    ER, PATCH = scan_logs(V)
    arms = present_arms(J)
    print(f"[build] arms present: {[c for c, _, _ in arms]}")
    print(f"[build] paired tests: {len(PJ)}   invE rows: {len(ER)}   patch rows: {len(PATCH)}")

    nv = 1387
    iw_s = best_of(J, "seen", "iw", arms)
    dc_s = best_of(J, "seen", "dcbg", arms)
    iw_u = best_of(J, "unseen", "iw", arms)
    dc_u = best_of(J, "unseen", "dcbg", arms)

    css = re.search(r"<style>(.*?)</style>",
                    open(join(ROOT, "report_src/iclr_iw_exp_ko.html")).read(), re.S).group(1)

    ceil_row = ""
    if False:  # ceilings intentionally not shown (see MET comment)
        c = CEIL
        ceil_row = (f'<p class="small"><b>지표 천장.</b> 정답 경로가 choice set 위 PSL 확률에서 뽑은 '
                    f'<b>확률적 표본 1개</b>이므로 같은 분포의 독립 추출 두 개도 서로 다르다. 그 일치율이 '
                    f'참조 대비 지표의 상한이다 &mdash; len-match <b>{c.get("em", float("nan")):.3f}</b>, '
                    f'DTW <b>{c.get("dtw", float("nan")):.3f}</b>, '
                    f'LCS/|ref| <b>{c.get("lcs_norm", float("nan")):.3f}</b>, '
                    f'경로 동일 <b>{c.get("exact_path", float("nan")):.3f}</b>. '
                    f'valid / arrival / v&amp;a 는 참조가 필요 없는 지표라 영향이 없다 '
                    f'(OD쌍 {c.get("n_pairs", "?")}개 기준).</p>')

    metric_sec = METRIC_SEC
    _rl = {"LE": "27.6", "LP": "28.4"}.get(V, "27.6")
    ref_len_note = f"{_rl} 홉"
    RG = "unseen" if args.regime == "unseen" else "seen"   # 단일 체제 표에 쓰는 값
    BOTH = args.regime == "both"
    rg_label = {"seen": "seen (e0 판별자/분류기)",
                "unseen": "unseen (e99, except_0 미학습)"}[RG]
    p13l_block = ""
    if P13L:
        p13l_block = (
            "<h3>repair + simplification (P1P3L)</h3><table>"
            + main_table(P13L, RG, arms)
            + '</table><p class="small">두 표의 차이가 <b>repair 가 만든 우회 중 '
              '루프였던 부분</b>이다. P1P3 는 실행 가능하게 만들면서 노드를 추가하고, 그 위에 '
              'simplification 을 얹으면 그 낭비가 걷힌다 &mdash; v&amp;a 는 1.000 그대로다.</p>')
    # Reference-row note. Every number is read from the record and from the best
    # model arm of the same table, and the win/lose wording is derived from those
    # numbers rather than asserted, so the text cannot drift from the table.
    shortest_note = ""
    if SHORTEST[0]:
        S = SHORTEST[0]

        def _best(metric, sign):
            cands = [(sign * m[metric], lab, b) for c, lab, f in arms for b in BLKS
                     if (m := get(J, RG, b, c)) and metric in m]
            if not cands:
                return float("nan"), "?", "?"
            v, lab, b = min(cands)
            return sign * v, lab, b

        lcs_b, lcs_lab, lcs_blk = _best("lcs", -1)
        dtw_b, dtw_lab, dtw_blk = _best("dtw_sum", +1)
        dtwp_b, _, _ = _best("dtw", +1)
        lcs_win = lcs_b > S["lcs"]
        dtw_win = dtw_b < S["dtw_sum"]
        # A per-pair cross-check, because dtw_sum has no divisor and BFS paths are
        # the shortest in the table: if normalising flipped the verdict, the DTW
        # comparison would be a length artefact rather than a geometric one.
        dtwp_win = dtwp_b < S["dtw"]
        flip = " (정규화를 <span class='mono'>/max(n,m)</span>로 바꾸면 판정이 뒤집힌다: "
        flip += (f"per-pair 는 모델 {dtwp_b:.1f} m 대 BFS {S['dtw']:.1f} m)"
                 if dtwp_win != dtw_win else "")
        same = (f" per-pair 로 정규화해도 같다 &mdash; 모델 {dtwp_b:.1f} m 대 BFS "
                f"{S['dtw']:.1f} m &mdash; 즉 경로 길이 차이가 만든 착시가 아니다.")
        dtw_cross = flip if dtwp_win != dtw_win else same
        verdicts = (f'LCS <b>{S["lcs"]:.2f}</b> 대 모델 최고 <b>{lcs_b:.2f}</b>'
                    f'({lcs_lab}, blk{lcs_blk}) &mdash; 모델 '
                    f'{"우세" if lcs_win else "열세"}, '
                    f'DTW <b>{S["dtw_sum"]:.0f} m</b> 대 모델 최고 '
                    f'<b>{dtw_b:.0f} m</b>({dtw_lab}, blk{dtw_blk}) &mdash; 모델 '
                    f'{"우세" if dtw_win else "열세"}.')
        shortest_note = f'''<p class="small"><b>회색 행 = shortest path (BFS).</b>
학습이 전혀 없는 참조선이다. <span class="mono">except_0</span> 시나리오 그래프에서
BFS 로 홉 최소 경로를 뽑고, 모델 arm 과 <b>완전히 같은 지표</b>로 채점했다
(<span class="mono">tools/eval/shortest_baseline.py</span>). 블록 크기 &middot; regime &middot;
후처리 변형에 모두 무관해 표마다 같은 값이 반복된다 &mdash; BFS 경로는 이미 valid &amp;
arriving 이라 repair 가 고칠 것이 없고, 단순 경로라 simplification 이 자를 것도 없다.
<b>따라서 v&amp;a = {S["valid_and_arrival"]:.3f} 을 공짜로 얻는다.</b>
이 행의 역할은 <b>v&amp;a 가 헤드라인이 될 수 없음을 명시</b>하는 것이다: 제약 충족은
탐색으로 풀리는 문제이므로, 실제 질문은 <b>관측된 통행 행태를 재현하는가</b>이며 그건
형상 지표만이 잰다. 이 표에서의 판정은 {verdicts}{dtw_cross}
모델이 LCS 에서 이기는 이유는 <b>참조 경로가 최단경로가 아니기</b> 때문이다: 평균
{S["ref_len"]:.2f} 홉으로 BFS {S["gen_len"]:.2f} 홉보다
<b>{S["ref_len"] - S["gen_len"]:+.2f} 홉</b> 길고, 1,000쌍 중 BFS 보다 짧은 경우가
<b>{S["ref_shorter_than_bfs"]}</b>건(같은 길이 {S["ref_equal_to_bfs"]}건)이다 &mdash;
관측 행태는 언제나 최단경로 이상으로 우회하며, 모델은 그 우회를 배우고 BFS 는 정의상 못 배운다.
DTW 는 그 우회를 <b>기하 거리</b>로 재므로 사정이 다르다: 참조를 못 맞춘 우회는 짧게
끊는 것보다 <b>더</b> 벌점을 받는다. 마지막 주의 &mdash; 동수 최단경로가 여럿인 쌍이
{S["n_tied"]}건이라 이 행은 &ldquo;최선의&rdquo; 최단경로가 아니라
<b>임의로 택한 최단경로 하나</b>의 점수다. 동수 중 참조에 가장 가까운 것을 고르면
BFS 행은 더 좋아진다.</p>'''

    pt = pair_table(J, arms, RG)
    et = edge_table(ER, arms, RG)
    gammas = sorted({(c.split("@")[1][1:] if "@" in c else "1")
                     for c, _, f in arms if f == "dcbg"}, key=float)

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Porto v6 {V} — D-CBG vs IW 가이던스</title><style>{css}
tr.dcbg td {{ background: #fbf7ef; }}
tr.iw td {{ background: #f4f8fb; }}
tr.ref td {{ background: #f2f2f0; color: #555; font-style: italic; }}
td.win {{ color: #0a6; font-weight: 600; }}
td.lose {{ color: #c33; font-weight: 600; }}
</style></head><body>

<h1>Porto v6 {V} — D-CBG vs 중요도 가중(IW) 가이던스</h1>
<p class="small">참조 데이터 <span class="mono">v6-{V}1.0-0.05</span> (Link
{'Elimination' if V == 'LE' else 'Penalization'}, PSL). 평가는 <span class="mono">except_0</span>의
예약 1,000쌍(<span class="mono">SPLIT_SEED=777</span>), reveal 순서 <span class="mono">l2r</span>,
D-CBG <span class="mono">&gamma; &isin; {{{', '.join(gammas)}}}</span>,
IW <span class="mono">n<sub>is</sub> = {args.nis}</span>, <span class="mono">w_gamma = 1.0</span>.
이 문서의 모든 수치는 <span class="mono">report_src/build_porto_LE_dcbg.py</span>가
JSON과 실행 로그에서 생성한다.</p>

{H2("무엇을 비교했나")}
<p>두 방법 모두 같은 백본 <span class="mono">p<sub>&theta;</sub></span>를 예외 시나리오 쪽으로 기울인다.
차이는 <b>기울이는 방식</b>이다.</p>
<ul>
<li><b>IW</b> — 블록 후보를 <span class="mono">n<sub>is</sub>={args.nis}</span>개 뽑아
판별자 가중치 <span class="mono">(D/(1&minus;D))<sup>&gamma;</sup></span>로 재가중한 x&#772;에서 한 위치를 공개.
자기정규화 중요도 표집(SNIS)이다.</li>
<li><b>D-CBG</b> (Schiff et al.) — 공개 위치의 어휘 전체에 대해
<span class="mono">softmax(log p<sub>&theta;</sub> + &gamma;&middot;log p<sub>&phi;</sub>(y|x<sub>t</sub>))</span>.
노이즈 조건부 분류기가 필요하다(IW 판별자는 시간 조건도 MASK 토큰도 없어 재사용 불가).</li>
</ul>
{ceil_row}
{H2("지표 정의와 검증")}
{metric_sec}

{H2("프로토콜 일치")}
<table>
<tr><th>항목</th><th>IW 판별자</th><th>D-CBG 분류기</th></tr>
<tr><td>positives</td><td colspan=2 class="ctr">except 시나리오 실데이터 <b>1%</b>
 (<span class="mono">RandomState(777+e)</span>, e=0은 평가용 1,000개 예약분 제외)</td></tr>
<tr><td>negatives</td><td colspan=2 class="ctr">그 블록의 무조건 생성 pool <b>20,000개</b>
 (<span class="mono">uncond_pool_blk{{B}}.pth</span>, 동일 파일)</td></tr>
<tr><td>학습</td><td colspan=2 class="ctr">4,000 step, bs 128 (pos 64 + neg 64), Adam lr 1e&minus;3,
 grad clip 1.0, label smoothing 0.05</td></tr>
<tr><td>adjacency 조건</td><td colspan=2 class="ctr">GraphSAGE + edge 채널 +
 <span class="mono">deg_ratio</span>, logit bound <b>4.0</b> (동일)</td></tr>
<tr><td>입력</td><td>깨끗한 부분 경로 <span class="mono">[dst, ori, v&hellip;]</span>,
 절단 위치 <span class="mono">j ~ U{{2,&hellip;,L}}</span></td>
 <td>블록 정렬 canvas + 블록 j 까지 clean commit + 현재 블록만
 <span class="mono">t~U(1e-3,1)</span> 마스킹 + 미래 절단</td></tr>
<tr><td>가이던스 강도</td><td><span class="mono">w_gamma = 1.0</span> (SNIS target 그 자체이며
 구 데이터에서 1&rarr;4 변화가 v&amp;a &plusmn;0.015로 방향성 없음)</td>
 <td><span class="mono">&gamma; &isin; {{{', '.join(gammas)}}}</span> (민감하므로 스윕)</td></tr>
<tr><td>reveal</td><td colspan=2 class="ctr">한 스텝에 <b>한 위치</b>, 순서 <span class="mono">l2r</span>
 (좌측 최말단 MASK), first-hitting 노이즈 스케줄 공통</td></tr>
<tr><td>평가</td><td colspan=2 class="ctr">동일 1,000쌍, 후처리 raw / P1 / P3 / P1P3 동일 구현</td></tr>
</table>

{H2(f"결과 &mdash; raw (후처리 없음), {rg_label}")}
<p class="small">괄호는 <span class="mono">base</span> 대비 증감. 굵은 값이 그 블록의 최고.
파란 행 = IW 계열, 노란 행 = D-CBG 계열.
<span class="mono">base</span>와 <span class="mono">Adj-only</span>는 판별자와 무관해
seen 실행분을 공유한다.</p>
{shortest_note}
<table>{main_table(J, RG, arms, CEIL)}</table>
{f'{H2("결과 &mdash; raw (후처리 없음), unseen 시나리오 (e99, except_0 미학습)")}<table>' + main_table(J, "unseen", arms, CEIL) + "</table>" if BOTH else ""}

{f'''{H2(f"simplification 만 적용 (rawL) &mdash; 루프 제거, repair 없음 &mdash; {rg_label}")}
<p>후처리는 성질이 다른 <b>두 축</b>이다. 섞어서 보면 잘못된 결론이 나온다.</p>
<ul>
<li><b>repair</b> (P1 불법 edge splice / P3 끝점 patch) &mdash; <b>실패를 보정</b>하므로
 feasibility 지표를 <b>바꾼다</b>(v&amp;a 가 전 arm 1.000 으로 포화해 모델 신호가 사라진다).
 노드를 <b>추가</b>만 한다.</li>
<li><b>simplification</b> (루프 제거) &mdash; 아무것도 고치지 않으므로 <b>arrival 은 정확히 불변,
 validity 는 비감소</b>(잘린 구간의 연속쌍이 원본의 부분집합이라 불법 edge 가 사라질 수도 있다).
 노드를 <b>제거</b>만 한다. 따라서 헤드라인 지표를 세탁하지 않고 무조건 적용해도 안전하다.</li>
</ul>
<p>정당성: v6 참조 경로는 <b>1,000/1,000 이 simple path</b>(루프 0개)이므로 목표 분포는
non-simple 경로에 질량을 주지 않는다 &mdash; 루프는 순수한 낭비다. 2026-08-21 부터
sampler 가 반환 직전에 이를 기본 적용한다(<span class="mono">models_seq/bd_models.py::simplify_path</span>,
<span class="mono">simplify=False</span> 로 끌 수 있음). 루프 제거는 반환된 경로의 <b>순수 함수</b>라
저장된 레코드에 사후 적용한 결과가 sampler 안에서 적용한 것과 동일하므로, 아래 수치는 재생성 없이
얻은 것이다.</p>
<table>{main_table(RAWL, RG, arms)}</table>
{f'<h3>unseen</h3><table>' + main_table(RAWL, "unseen", arms) + "</table>" if BOTH else ""}
<p class="small">위 표를 §{_H2N[0] - 2} 의 raw 와 비교하면 <b>모든 arm 의 <span class="mono">gen_len</span> 이
27&ndash;30 홉으로 수렴</b>한다(참조 {ref_len_note}). raw 에서는 마스킹 arm 이 합법 보행 때문에
37&ndash;45 홉까지 늘어나 있었다 &mdash; 그 초과분의 상당 부분이 루프였다.
<span class="mono">v&amp;a</span> 는 예측대로 불변이거나 미세하게 개선된다(validity 비감소).''' if RAWL else ""}

{f'''{H2("헤드라인 &mdash; 같은 조건에서 맞붙이면")}
<p class="small">&ldquo;차&rdquo; = IW &minus; D-CBG. 초록이 IW 우세, 빨강이 D-CBG 우세
(DTW는 작을수록 좋으므로 방향을 반전해 판정). {RG} 기준.</p>
<table>{pt}</table>''' if pt else ""}

{f'''{H2("짝지은 유의성 검정")}
<p class="small">두 arm 은 <b>같은 1,000 OD 쌍</b>에서 평가되므로 독립 표본이 아니다. 독립 가정
SE = &radic;(p(1&minus;p)/n) 은 짝지은 차이의 불확실성을 과대평가하므로, 이진 지표
(v&amp;a, len-match)는 <b>정확 McNemar</b>(불일치 쌍만 사용), 연속 지표(DTW, LCS)는
<b>Wilcoxon signed-rank</b>로 p 를 내고, 차이의 구간은 두 경우 모두
<b>짝지은 부트스트랩</b>으로 낸다. &Delta; = A &minus; B (<b>DTW 는 작을수록 좋으므로
&Delta; &lt; 0 이 A 우세</b>, 나머지는 &Delta; &gt; 0 이 A 우세).
초록 = A 유의 우세, 빨강 = B 유의 우세 (p &lt; 0.05). 형상 지표는 빈 생성 경로를 제외해
짝을 맞춘다. 생성: <span class="mono">tools/collect/paired_test.py</span>,
<span class="mono">scripts/run_paired_battery.sh</span></p>
<table>{paired_table(PJ)}</table>
<p class="small"><b>지금 형상 지표 검정이 없는 이유.</b> 위 표에는
<span class="mono">valid &amp; arrival</span> 검정만 있다. 이 지표는 ground distance 나 정규화와
무관하므로 그대로 유효하다. 반면 저장돼 있는 DTW/LCS 검정은 <b>구 지표 정의</b>(도 L1 &times; 100 의
<span class="mono">max(n,m)</span> 정규화, <span class="mono">LCS/|ref|</span>) 기준이라
이 문서의 표(haversine 누적 합, LCS 원값)와 <b>짝이 맞지 않아 의도적으로 제외</b>했다.
새 정의로 재실행하면 채워진다 &mdash;
<span class="mono">VARIANT={V} A=adj+modelD B=dcbg+adjp SFX=... METRICS="va dtw_sum lcs"
bash scripts/run_paired_battery.sh -variant rawL</span>.
<span class="mono">len-match</span>(구 &ldquo;EM&rdquo;) 검정도 지표 자체를 폐기했으므로 뺐다
(IW 유의 우세 {lm_a}셀 / D-CBG {lm_b}셀로 결론 방향은 같았다). 구 JSON 은
<span class="mono">sets_v6/{V}/res/paired_*.json</span> 에 그대로 남아 있다.</p>''' if PJ else ""}

{H2("연산량")}
<p>reveal 1회당 분류기/판별자 호출 수. D-CBG exact 는 공개 위치에서
<span class="mono">|V| = {nv}</span>개를 전수 열거한다.</p>
<table>
<tr><th>방법</th><th class="num">reveal당 호출</th><th class="num">IW 대비</th></tr>
<tr><td>D-CBG 1차 근사 (<span class="mono">-approx 1</span>)</td>
 <td class="num">1 (fwd+bwd)</td><td class="num">1/{args.nis}</td></tr>
<tr class="iw"><td>IW (<span class="mono">n<sub>is</sub>={args.nis}</span>)</td>
 <td class="num">{args.nis}</td><td class="num">1&times;</td></tr>
<tr class="dcbg"><td>D-CBG exact</td><td class="num">{nv}</td>
 <td class="num">{nv / args.nis:.1f}&times;</td></tr>
</table>
<p class="small">즉 exact D-CBG 는 IW보다 <b>{nv / args.nis:.0f}배 많은</b> 호출을 쓰고,
1차 근사는 <b>{args.nis}배 적게</b> 쓴다. 두 arm 이 IW를 연산 예산의 위아래로 감싼다.</p>

{f'''{H2(f"legality &mdash; 불법 edge 비율 (raw, {RG})")}
<p class="small"><span class="mono">invE</span> = 시나리오 그래프에 없는 edge의 비율,
<span class="mono">remE</span> = 그중 &ldquo;시나리오에서 제거된&rdquo; edge의 비율.</p>
<table>{et}</table>
<p>두 분류기 모두 logit bound가 <b>4.0</b>이므로 가이던스 항의 최대 진폭이
<b>4 nats (&asymp; 55배)</b>다. 즉 <b>어느 쪽도 불법 토큰의 확률을 0으로 만들 수 없다</b> &mdash;
가이던스 단독으로는 legality가 세워지지 않고, 이를 세우는 것은 Lemma-3 마스킹이다.
&gamma;를 올리면 이 진폭이 커져 legality는 개선되지만(실측: D-CBG validity blk2 0.382&rarr;0.596)
하드 제약에는 못 미친다(<span class="mono">Adj-only</span> 0.777 vs
<span class="mono">D-CBG &gamma;=3</span> 0.578, v&amp;a 기준).</p>''' if et else ""}

{f'''{H2(f"후처리 P1P3 &mdash; legality 를 통제한 형상 비교 ({RG})")}
<p>raw 표에는 함정이 하나 있다. <b>DTW/LCS 는 legality 를 재지 않으므로 validity 가 다른 arm 사이에서는
단독으로 해석할 수 없다.</b> 실제로 raw DTW 에서는 <span class="mono">base</span>(v&amp;a 0.12&ndash;0.28,
즉 대부분 불법 경로)가 7블록 중 6개에서 가이던스 arm 을 이긴다. 원인은 두 방법의 <b>실패 방식이 다르다</b>는
것이다 &mdash; <span class="mono">base</span>는 불법 점프로 짧게 질러가지만 도착은 하고(arrival 0.93&ndash;0.99),
마스킹 arm 은 합법이지만 자주 못 닿는다(arrival 0.69&ndash;0.80). DTW 는 양 끝점을 강제로 정렬하므로
<b>미도달을 매우 세게 벌한다</b>.</p>
<p>후처리 <b>P1P3</b>(P1 = 불법 edge 를 최단경로로 splice, P3 = 끝점을 목적지로 patch)를 적용하면
<b>모든 arm 의 v&amp;a 가 1.000 으로 포화</b>하여 legality 와 도달이 통제된다. 그 조건에서
&ldquo;모두 실행 가능한 계획을 낸 상태에서 누가 참조에 더 가까운가&rdquo;를 물을 수 있고, 이것이 본 연구의
주장에 맞는 질문이다.</p>
<h3>repair 만 (P1P3)</h3>
<table>{main_table(P13, RG, arms, CEIL)}</table>
{p13l_block}
<p class="small"><b>후처리가 얼마나 들어갔나.</b> P1/P3 는 <b>노드를 추가만</b> 하므로
&ldquo;추가된 평균 노드 수&rdquo;가 후처리 비용을 그대로 나타낸다
(<span class="mono">splice()</span>/<span class="mono">endpoint()</span>가 세어 로그에
<span class="mono">patch=</span>로 찍는 값이며, <span class="mono">gen_len</span>(P1P3) &minus;
<span class="mono">gen_len</span>(raw) 와 정확히 일치한다 &mdash; 예: LE blk1 seen
<span class="mono">dcbg</span> 로그 patch 6.05, gen_len 28.25&rarr;34.30). 참조 경로 평균 길이는
<b>{ref_len_note}</b> 이므로 이 값을 그 비율로 읽으면 된다. 큰 값은 그만큼 모델 출력이
후처리에 의존했다는 뜻이다.</p>
<table>{patch_table(PATCH, ER, arms, RG)}</table>
<p class="small">각 칸은 <b>P1P3 합계</b>이고 아래 작은 글씨가 <span class="mono">P1 + P3</span> 분해다.
<b>마스킹 유무로 후처리의 성격이 갈린다</b> &mdash; 마스킹 없는 arm
(<span class="mono">base</span>, <span class="mono">IW-only</span>, <span class="mono">D-CBG</span>)은
validity 가 낮아 <b>P1(불법 edge splice)이 지배</b>하고, 마스킹 arm
(<span class="mono">Adj-only</span>, <span class="mono">Adj+IW</span>,
<span class="mono">D-CBG+Adj</span>)은 validity 가 0.99+ 라 P1 이 거의 0 이고 대신 arrival 이
0.69&ndash;0.80 이라 <b>P3(끝점 patch)가 지배</b>한다.</p>
<p class="small"><b>부호가 뒤집힌다.</b> raw 에서 <span class="mono">base</span>가 DTW 6승이었는데
P1P3 에서는 <span class="mono">Adj+IW</span>가 6승이 된다(LE seen 기준). P3 가 끝점을 patch 하면
<span class="mono">Adj+IW</span>의 DTW 가 크게 개선되는 반면(예: blk8 0.198&rarr;0.149)
<span class="mono">base</span>는 거의 변하지 않는다(0.175&rarr;0.172) &mdash; 불법 edge 를 splice 해도
지리적 위치가 거의 안 바뀌기 때문이다. 즉 raw DTW 의 열세는 <b>길이 팽창이 아니라 미도달</b> 때문이었다.
구조적 천장도 없다: 참조 경로 자체가 시나리오 그래프에서 합법이며 그 DTW 천장은
<b>{CEIL["dtw"]:.3f}</b> 다.</p>''' if P13 else ""}

{H2("왜 마스킹이 두 방법에서 다른 의미인가")}
<p>S = {{시나리오 그래프에서 합법인 경로}}, 목표 분포 &pi; = 예외 시나리오 데이터 분포이라 하면
데이터에 불법 edge가 있을 수 없으므로 <b>&pi;(S<sup>c</sup>) = 0</b>이다.</p>
<ul>
<li><b>IW</b>: 마스크는 <b>proposal</b>에만 걸린다
 (<span class="mono">bd_models.py::_denoise_block_mask_guided</span>의
 <span class="mono">p_cand = pm / z</span>). mean-field proposal이라 정규화 상수
 <span class="mono">z</span>는 그 행의 모든 후보에 공통이므로 자기정규화에서 소거되고,
 합법 후보의 가중치는 변하지 않는다. 따라서 <b>추정 대상은 여전히 &pi;</b>이며 제거된 영역은
 목표 질량이 0인 곳뿐이다 &mdash; <b>새 근사를 추가하지 않는 순수한 분산 감소</b>다.
 (같은 장부 정리가 ancestral proposal에서는 후보별 상수
 <span class="mono">&prod;<sub>j</sub>&alpha;<sub>j</sub></span>가 소거되지 않아
 <span class="mono">-anc_correct</span>로 보정된다. masked mean-field는 그 보정항이 정확히 1인 경우다.)</li>
<li><b>D-CBG</b>: proposal과 target이 분리돼 있지 않아 마스크가 <b>sampler의 법칙 자체</b>에 걸린다
 (<span class="mono">guided[~legal] = -1e9</span>). bounded logit 때문에
 <span class="mono">p<sub>&phi;</sub> &gt; 0</span>이고 softmax라
 <span class="mono">p<sub>&theta;</sub> &gt; 0</span>이므로 <b>p(S<sup>c</sup>) &gt; 0</b>이고,
 마스킹은 <b>목표 분포를 재정의</b>한다. D-CBG의 유도는 그 변경을 허락하지 않는다.</li>
</ul>
<p class="small"><b>정직한 각주.</b> IW 쪽 support 보존에는 수치 안전장치 하나가 있다 &mdash;
모델이 합법 후속 전체와 <span class="mono">END</span>/<span class="mono">PAD</span>에
10<sup>&minus;9</sup> 미만의 질량을 주면 마스크를 포기하고 원래 분포로 되돌아간다
(<span class="mono">z &gt; 1e-9</span> 분기). D-CBG에도 동일한 분기가 있어
(<span class="mono">legal[z0 &lt;= 1e-9] = True</span>) <b>비교에는 대칭</b>이다.
한편 <span class="mono">xbar_sum &gt; 1e-12</span> / <span class="mono">p_sel.sum &gt; 1e-12</span>
두 가드는 row-max 이동(<span class="mono">wgt = exp(g &minus; g.max())</span>)으로 최대 가중치가 정확히 1이
되므로 <b>발동할 수 없다</b>.</p>
<p class="small"><b>큰 블록에서의 한계 (소유).</b> 블록이 커지면 IW의 mean-field proposal이
내부적으로 비일관한 후보 블록을 내놓아 열화한다 &mdash; 실측으로 blk&ge;16에서 v&amp;a가
D-CBG에 밀린다. 구 데이터에서 ancestral proposal(legal-by-construction 순차 제안)이 blk64에서
v&amp;a +0.052 / len-match +0.064로 이를 메웠으나, 그 방식은 &ldquo;proposal 은 모델 자신의
marginal 에서 값싸게 뽑는다&rdquo;는 전제를 흐리고 블록 길이만큼 순차 루프라 채택하지 않았다.
원인이 정식화가 아니라 값싼 proposal 의 한계임을 밝히고 그대로 둔다. 형상 지표(DTW/LCS)에서는
큰 블록 셀이 대체로 무승부라 이 열화는 <b>도달률 현상</b>이다.</p>

{H2("실행")}
<p class="small"><b>파이프라인</b> <span class="mono">scripts/run_v6_dcbg.sh</span> (VARIANT={V}) &mdash;
A 분류기 {len(BLKS)}블록 &times; {{e0, e99}} = 14개(<span class="mono">-adj 1 -neg model -frac 1
-steps 4000</span>) &rarr;
B <span class="mono">tools/eval/eval_dcbg.py</span> &times; (블록 &times; regime &times; arm),
<span class="mono">-order l2r</span> &rarr;
C <span class="mono">tools/collect/collect_va_table.py -shape 1</span>로 IW arm과 한 표로 합침.
IW arm은 <span class="mono">scripts/run_v6_all.sh</span> 단계 D의 산출물을 그대로 읽는다.
&gamma; 스윕은 분류기를 재사용하고 평가만 다시 하며, arm 이름에
<span class="mono">@g3</span> 같은 접미사를 붙여 앞선 &gamma;의 레코드를 덮지 않는다.</p>
<p class="small"><b>v6에서 고친 것.</b> <span class="mono">eval_dcbg.py</span> /
<span class="mono">train_dcbg_classifier.py</span>는 v4 전용으로 하드코딩돼 있었다.
특히 <span class="mono">n_vertex = 1390</span>이 상수였는데 v6는 <b>1387</b>이다.
<span class="mono">dcbg_plugin</span>이 백본의 canvas를 그대로 분류기에 넣으므로
(<span class="mono">F.one_hot(seq, clf.vocab)</span>) 이 값이 어긋나면
<span class="mono">END/PAD/MASK</span> id가 밀려 <b>에러 없이 조용히 틀린다</b>.
<span class="mono">A_norm.shape[0]</span>에서 읽도록 바꾸고
<span class="mono">clf.vocab == backbone.vocab_size</span> assert를 넣었다.</p>

</body></html>"""

    rsfx = "" if BOTH else f"_{args.regime}"
    out_html = join(ROOT, f"report_src/porto_{V}_dcbg{rsfx}_ko.html")
    io.open(out_html, "w", encoding="utf-8").write(doc)
    print(f"written {out_html} ({len(doc)} bytes)")
    print(f"seen  best: IW {iw_s[0]:.3f} (blk{iw_s[1]} {iw_s[2]}) / "
          f"D-CBG {dc_s[0]:.3f} (blk{dc_s[1]} {dc_s[2]})")
    print(f"unseen best: IW {iw_u[0]:.3f} (blk{iw_u[1]} {iw_u[2]}) / "
          f"D-CBG {dc_u[0]:.3f} (blk{dc_u[1]} {dc_u[2]})")
