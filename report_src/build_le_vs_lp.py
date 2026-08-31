"""
LE vs LP: does the choice-set generation method change how much guidance helps?

  python report_src/build_le_vs_lp.py

Inputs
  sets_v6/le_vs_lp_facts.json                        tools/diag/le_vs_lp_facts.py
  sets_v6/{V}/res/va_table_v6{V}_nis{n}_rawL.json     scripts/run_v6_nis.sh
  sets_v6/{V}/res/shortest_v6{V}.json                tools/eval/shortest_baseline.py
  sets_v6/{V}/res/nis_diversity_v6{V}.json           tools/diag/diag_proposal_diversity.py
Output
  report_src/le_vs_lp_ko.html  ->  pdfs/Porto_LE_vs_LP/le_vs_lp_ko.pdf

  PDF:
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
      --no-pdf-header-footer --print-to-pdf=pdfs/Porto_LE_vs_LP/le_vs_lp_ko.pdf \
      "file://$PWD/report_src/le_vs_lp_ko.html"

The report is organised as a hypothesis with evidence for and against, because the
evidence is genuinely mixed: the prediction about the reference paths is confirmed
cleanly, the prediction about discriminator training is not, and the per-block
guidance gains only partly line up.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import io
import json
import os
import re
from os.path import dirname, join

import numpy as np

ROOT = dirname(dirname(os.path.abspath(__file__)))
VARS = ["LE", "LP"]
NIS = [10, 30, 50, 100, 150]
# Blocks present in the n_is sweep are DISCOVERED from the tables, not hardcoded:
# the sweep was first run on {1,2,16,64} and later extended to all seven, and a
# hardcoded list would silently drop the new rows.
BLKS_SWEEP = []
BLKS_ALL = [1, 2, 4, 8, 16, 32, 64]

_H2N = [0]


def H2(t):
    _H2N[0] += 1
    return f"<h2>{_H2N[0]}. {t}</h2>"


def jload(p, required=True):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        if required:
            raise SystemExit(f"MISSING {p}")
        return None


def get(J, regime, blk, cfg):
    return J.get(f"{regime}_blk{blk}_{cfg}") or J.get(f"seen_blk{blk}_{cfg}")


def trend_mark(vals):
    """Direction across n_is; a range alone cannot separate trend from noise."""
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return ""
    d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    up, dn = sum(x > 0 for x in d), sum(x < 0 for x in d)
    if up == 0 and dn == 0:
        return ""
    if dn == 0:
        return "&uarr;"
    if up == 0:
        return "&darr;"
    if up >= 3 * max(dn, 1):
        return "&nearr;"
    if dn >= 3 * max(up, 1):
        return "&searr;"
    return "&sim;"


if __name__ == "__main__":
    F = jload(join(ROOT, "sets_v6/le_vs_lp_facts.json"))
    S = {V: jload(join(ROOT, f"sets_v6/{V}/res/shortest_v6{V}.json")) for V in VARS}
    D = {V: jload(join(ROOT, f"sets_v6/{V}/res/nis_diversity_v6{V}.json"),
                  required=False) for V in VARS}
    T = {V: {n: jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_nis{n}_rawL.json"),
                      required=False) for n in NIS} for V in VARS}
    for V in VARS:
        T[V] = {n: j for n, j in T[V].items() if j}
    have = {V: sorted(T[V]) for V in VARS}
    found = set()
    for V in VARS:
        for n, J in T[V].items():
            for k in J:
                if (m := re.search(r"_blk(\d+)_", k)):
                    found.add(int(m.group(1)))
    BLKS_SWEEP[:] = sorted(found)
    print(f"[build] sweep blocks discovered: {BLKS_SWEEP}")
    print(f"[build] n_is present: { {V: have[V] for V in VARS} }")

    # ---------- guidance gain: Adj+IW minus Adj-only ----------
    def gain(V, b, n, met="valid_and_arrival"):
        J = T[V].get(n)
        if not J:
            return None
        a, o = get(J, "seen", b, "adj+modelD"), get(J, "seen", b, "adjonly")
        if not a or not o:
            return None
        return a[met] - o[met]

    gain_rows = []
    for b in BLKS_SWEEP:
        cells = ""
        for V in VARS:
            vs = [gain(V, b, n) for n in have[V]]
            cells += "".join(
                (f'<td class="num {"win" if v > 0 else ("lose" if v < 0 else "")}">'
                 f'{v:+.3f}</td>') if v is not None
                else '<td class="num muted">&ndash;</td>' for v in vs)
            ok = [v for v in vs if v is not None]
            cells += (f'<td class="num"><b>{np.mean(ok):+.3f}</b></td>'
                      f'<td class="num">{trend_mark(vs)}</td>'
                      if ok else '<td class="num muted">&ndash;</td><td></td>')
        gain_rows.append(f"<tr><td>{b}</td>{cells}</tr>")
    gain_head = ("<tr><th rowspan=2>blk</th>"
                 + "".join(f'<th class="num" colspan={len(have[V]) + 2}>{V}</th>'
                           for V in VARS) + "</tr><tr>"
                 + "".join("".join(f'<th class="num">n={n}</th>' for n in have[V])
                           + '<th class="num">평균</th><th class="num">추세</th>'
                           for V in VARS) + "</tr>")

    # ---------- shortestness comparison ----------
    SH = {V: F[V]["shortestness"] for V in VARS}
    PT = {V: F[V]["perturbation"] for V in VARS}
    RV = {V: F[V]["ref_validity"] for V in VARS}
    DV = {V: F[V]["disc_val"] for V in VARS}

    def row2(lab, fmt, key, src, better=None, note=""):
        """One metric, LE and LP side by side, plus which side the hypothesis wants."""
        a, b_ = src["LE"][key], src["LP"][key]
        mark = ""
        if better == "LE_lower":
            mark = "&#10004;" if a < b_ else "&#10008;"
        elif better == "LE_higher":
            mark = "&#10004;" if a > b_ else "&#10008;"
        elif better == "same":
            # 1% relative tolerance: these are means over scenarios, so 3798.0 vs
            # 3798.1 is the same perturbation, not a difference between datasets.
            tol = 0.01 * max(abs(a), abs(b_), 1e-12)
            mark = "&#10004;" if abs(a - b_) <= tol else "&#10008;"
        return (f'<tr><td>{lab}</td><td class="num">{fmt.format(a)}</td>'
                f'<td class="num">{fmt.format(b_)}</td>'
                f'<td class="num">{fmt.format(a - b_)}</td>'
                f'<td class="ctr">{mark}</td><td class="small">{note}</td></tr>')

    disc_rows = ""
    for b in BLKS_ALL:
        c = ""
        for V in VARS:
            d = DV[V].get(str(b)) or DV[V].get(b)
            c += ("".join(f'<td class="num">{d[k]:.3f}</td>'
                          for k in ("train_acc", "val_acc", "logit_std"))
                  if d else '<td class="num muted" colspan=3>&ndash;</td>')
        disc_rows += f"<tr><td>{b}</td>{c}</tr>"
    dv_mean = {V: np.mean([x["val_acc"] for x in DV[V].values()]) for V in VARS}
    dv_lo = {V: min(x["val_acc"] for x in DV[V].values()) for V in VARS}
    dv_hi = {V: max(x["val_acc"] for x in DV[V].values()) for V in VARS}

    # ---------- diversity, to keep the n_is story attached ----------
    div_rows = ""
    if all(D[V] for V in VARS):
        for b in BLKS_SWEEP:
            c = ""
            for V in VARS:
                for n in NIS:
                    r = [x for x in D[V] if x["blk"] == b and x["n_is"] == n
                         and x["adj_prop"]]
                    c += (f'<td class="num">{r[0]["uniq"]:.1f}</td>' if r
                          else '<td class="num muted">&ndash;</td>')
            div_rows += f"<tr><td>{b}</td>{c}</tr>"

    # Per-block verdict, generated from the numbers. Hardcoding this text broke
    # once already: it was written for a 4-block sweep and silently became wrong
    # when blk 4/8/32 were added.
    def gmean(V, b):
        ok = [x for x in (gain(V, b, n) for n in have[V]) if x is not None]
        return np.mean(ok) if ok else float("nan")

    per_blk = [(b, gmean("LE", b), gmean("LP", b)) for b in BLKS_SWEEP]
    per_blk = [(b, a, c) for b, a, c in per_blk if a == a and c == c]
    n_lp = sum(1 for _, a, c in per_blk if c > a)
    n_le = len(per_blk) - n_lp
    verdict_head = (f"혼재한다. 가설이 예측하는 방향(LP &gt; LE)은 "
                    f"{len(per_blk)}개 블록 중 <b>{n_lp}개</b>에서만 맞고, "
                    f"{n_le}개에서는 <b>반대</b>다.")
    verdict_items = "\n".join(
        f'<li><b>blk{b}</b>: LE {a:+.3f} / LP {c:+.3f} &mdash; '
        f'{"<b>가설 방향</b>" if c > a else "<b>반대 방향</b>"}'
        f' (차 {c - a:+.3f})'
        + (" &mdash; 성능 <b>운용점</b>" if b == 2 else "")
        + (" &mdash; 둘 다 0 에 가깝다, 마스킹만으로 포화" if max(abs(a), abs(c)) < 0.010
           else "") + "</li>"
        for b, a, c in per_blk)

    css = re.search(r"<style>(.*?)</style>",
                    open(join(ROOT, "report_src/iclr_iw_exp_ko.html")).read(),
                    re.S).group(1)

    # headline numbers used in prose, read from the data so they cannot drift
    g2 = {V: np.mean([x for x in (gain(V, 2, n) for n in have[V]) if x is not None])
          for V in VARS}
    g64 = {V: np.mean([x for x in (gain(V, 64, n) for n in have[V]) if x is not None])
           for V in VARS}
    g16 = {V: np.mean([x for x in (gain(V, 16, n) for n in have[V]) if x is not None])
           for V in VARS}
    g1 = {V: np.mean([x for x in (gain(V, 1, n) for n in have[V]) if x is not None])
          for V in VARS}

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Porto v6 — LE vs LP: choice set 생성 방식과 가이던스 효과</title><style>{css}
tr.iw td {{ background: #f4f8fb; }}
td.win {{ color: #0a6; font-weight: 600; }}
td.lose {{ color: #c33; font-weight: 600; }}
td.ctr, th.ctr {{ text-align: center; }}
</style></head><body>
<h1>LE vs LP &mdash; choice set 생성 방식이 가이던스 효과를 바꾸는가</h1>

{H2("가설")}
<p>v6 의 두 데이터셋은 <b>예외 시나리오가 다른 것이 아니라 choice set 생성 방식이
다르다</b>. OD 쌍마다 choice set 을 먼저 만들고 그 중 하나를 참조 경로로 뽑는데,
만드는 방법이 이렇게 갈린다.</p>
<table>
<tr><th>데이터셋</th><th>choice set 생성</th><th>구성원의 성질</th></tr>
<tr><td><b>LE</b> (link elimination)</td>
 <td>edge 를 <b>삭제</b>하며 최단경로를 반복 수집</td>
 <td>모든 구성원이 <b>어떤 부분그래프에서의 최단경로</b></td></tr>
<tr><td><b>LP</b> (link penalization)</td>
 <td>edge 비용에 <b>벌점</b>을 주며 최단경로를 반복 수집. edge 는 삭제되지 않는다</td>
 <td>수정된 비용 아래의 최단경로 &mdash; 홉 수로는 어느 부분그래프에서도 최단이 아니다</td></tr>
</table>
<p>한편 <b>except_e 시나리오는 edge 를 삭제</b>한다. 그러면 LE 쪽에는 구조적 혼입이
생긴다 &mdash; <b>LE 의 학습 분포에는 이미 &ldquo;edge 가 빠진 그래프의 최단경로&rdquo;가
가득하다.</b> 시나리오가 edge 를 지워 만든 경로가 LE 의 평범한 데이터와 구별이 잘 안
된다는 뜻이고, 판별자의 positive 클래스가 그만큼 덜 특징적이 된다. LP 에는 삭제를
닮은 것이 학습 분포에 없으므로 시나리오 신호가 더 깨끗해야 하고, <b>따라서 가이던스가
LP 에서 더 잘 들어야 한다.</b> 이것이 검증 대상이다.</p>
<p class="small">간접적인 논거임을 분명히 해 둔다 &mdash; LE 의 choice set 삭제와
시나리오 삭제는 같은 연산이지만 같은 edge 집합이 아니다. 그래서 예측을 세 갈래로
쪼개 각각 측정했다.</p>

{H2("전제 확인 &mdash; 시나리오 교란은 두 데이터셋에서 동일하다")}
<p>먼저 확인해야 할 것은 &ldquo;LE 와 LP 가 시나리오 자체에서 다르지 않다&rdquo;는 점이다.
다르면 이후 비교가 전부 오염된다. except_e 그래프를 normal 과 비교했다
(<span class="mono">{PT["LE"]["n_scen"]}</span> 개 시나리오 평균).</p>
<table>
<tr><th>항목</th><th class="num">LE</th><th class="num">LP</th>
 <th class="num">차</th><th class="ctr">가설 예상</th><th>비고</th></tr>
{row2("normal edge 수", "{:.0f}", "normal_edges", PT, "same")}
{row2("시나리오 edge 수", "{:.1f}", "scenario_edges_mean", PT, "same")}
{row2("삭제된 edge", "{:.1f}", "removed_mean", PT, "same")}
{row2("추가된 edge", "{:.1f}", "added_mean", PT, "same")}
{row2("삭제 비율", "{:.4f}", "removed_frac", PT, "same")}
{row2("except_0 &cap; except_1 삭제 겹침", "{:.0f}", "overlap_exc0_exc1", PT,
      None, "198개 중. 시나리오는 서로 독립에 가깝다")}
</table>
<p><b>동일하다.</b> 둘 다 3996 개 중 <b>198 개(4.95%)</b>를 삭제하고 추가는 0 이다.
그러므로 &ldquo;LE 는 edge 를 지우는 데이터&rdquo;라는 말은 <b>시나리오</b>에 대해서는 LP 에도
똑같이 적용된다. 가설이 성립할 자리는 시나리오가 아니라 <b>choice set 쪽뿐</b>이며,
이 표는 그 점을 확정한다.</p>
<p class="small">부수 확인 &mdash; 참조 경로가 자기 시나리오 그래프에서 합법인가?
LE {RV["LE"]["invalid_paths"]}/{RV["LE"]["n_pairs"]}, LP
{RV["LP"]["invalid_paths"]}/{RV["LP"]["n_pairs"]} 가 불법이고 불법 edge 는
{RV["LE"]["illegal_edges"]}/{RV["LE"]["total_edges"]} 와
{RV["LP"]["illegal_edges"]}/{RV["LP"]["total_edges"]} 다. 즉 <b>둘 다 0</b>.
&ldquo;LP 는 벌점만 줬으니 참조 경로가 벌점 링크를 계속 쓴다&rdquo;는 가능성은 배제된다 &mdash;
벌점은 choice set 을 만들 때만 쓰이고, 뽑힌 경로는 시나리오 그래프에서 온전히 합법이다.</p>

{H2("예측 1 (확인됨) &mdash; LE 참조 경로가 최단경로에 더 가깝다")}
<p>가설의 가장 직접적인 결과다. LE 구성원이 <b>부분그래프의 최단경로</b>라면, 뽑힌 참조
경로도 최단경로에서 덜 벗어나야 한다. LP 는 벌점을 피해 우회하므로 더 벗어나야 한다.
BFS 홉 최소 경로를 기준으로 재면 <b>네 지표가 모두 같은 방향</b>을 가리킨다.</p>
<table>
<tr><th>항목</th><th class="num">LE</th><th class="num">LP</th>
 <th class="num">차</th><th class="ctr">가설 예상</th><th>비고</th></tr>
{row2("참조 경로 길이 (홉)", "{:.2f}", "ref_len", SH, "LE_lower",
      "BFS 는 두 데이터셋에서 23.94 홉으로 동일(OD 쌍이 같다)")}
{row2("BFS 대비 초과 홉", "{:.2f}", "hop_excess_mean", SH, "LE_lower", "")}
{row2("우회 비율 |ref|/|BFS|", "{:.3f}", "detour_ratio_mean", SH, "LE_lower", "")}
{row2("BFS 와의 DTW (m)", "{:.0f}", "dtw_to_bfs", SH, "LE_lower",
      "기하 이탈. 작을수록 최단경로에 가깝다")}
{row2("BFS 와의 LCS", "{:.2f}", "lcs_with_bfs", SH, "LE_higher",
      "최단경로와 공유하는 노드 수. 클수록 가깝다")}
{row2("BFS 와 길이가 같은 비율", "{:.3f}", "frac_equal_to_bfs", SH, None,
      "이 지표만 차이가 없다 &mdash; 아래 주석 참조")}
{row2("BFS 보다 짧은 비율", "{:.3f}", "frac_shorter_than_bfs", SH, None,
      "정의상 0. BFS 가 홉 최소이므로")}
</table>
<p><b>예측대로다.</b> LE 참조 경로는 평균
{SH["LE"]["hop_excess_mean"]:.2f} 홉만 우회하는데 LP 는
{SH["LP"]["hop_excess_mean"]:.2f} 홉 우회한다. BFS 와의 DTW 도 LE
{SH["LE"]["dtw_to_bfs"]:.0f} m 대 LP {SH["LP"]["dtw_to_bfs"]:.0f} m,
LCS 도 LE {SH["LE"]["lcs_with_bfs"]:.2f} 대 LP {SH["LP"]["lcs_with_bfs"]:.2f} 로
LE 가 일관되게 더 &ldquo;최단경로 같다&rdquo;. <b>choice set 생성 방식이 참조 경로의 성격에
실제로 각인된다</b>는 뜻이고, 가설의 전제가 데이터에서 확인된 것이다.</p>
<p class="small">&ldquo;BFS 와 길이가 같은 비율&rdquo;만 차이가 없다(9.2% 대 9.3%). 당연하다 &mdash;
LE 구성원이 최단인 것은 <b>choice set 을 만들 때 쓴 부분그래프</b>에서이고, 우리가 비교하는
BFS 는 <b>except_0</b> 그래프에서의 최단경로다. 두 부분그래프가 다르므로 정확히 일치할
이유가 없다. 평균적으로 더 가까울 뿐이며, 위의 연속 지표들이 바로 그 &ldquo;평균적으로&rdquo;를
잡아낸다.</p>

{H2("예측 2 (기각됨) &mdash; 판별자 학습 난이도는 차이가 없다")}
<p>가설이 맞다면 LE 의 판별자 학습이 더 어려워야 한다. positive(except_0 경로)가
negative(무조건 모델 pool)와 덜 구별되기 때문이다. e0 판별자의 학습 로그에서
마지막 스텝의 정확도를 긁어 비교했다.</p>
<table>
<tr><th rowspan=2>blk</th><th class="num" colspan=3>LE</th>
 <th class="num" colspan=3>LP</th></tr>
<tr><th class="num">train</th><th class="num">val</th><th class="num">logit_std</th>
 <th class="num">train</th><th class="num">val</th><th class="num">logit_std</th></tr>
{disc_rows}
</table>
<p><b>차이가 없다.</b> val 정확도 평균이 LE <b>{dv_mean["LE"]:.3f}</b>
({dv_lo["LE"]:.3f}&ndash;{dv_hi["LE"]:.3f}), LP <b>{dv_mean["LP"]:.3f}</b>
({dv_lo["LP"]:.3f}&ndash;{dv_hi["LP"]:.3f}) 로 사실상 같고 블록별 대소도 뒤섞인다.
logit_std 도 1.87&ndash;2.25 로 겹친다.</p>
<p>그러므로 가설이 맞다 해도 그 효과는 <b>학습 과제의 난이도로는 나타나지 않는다.</b>
이건 이상한 일이 아니다 &mdash; 학습/검증 과제는 &ldquo;경로 하나를 보고 시나리오 데이터인지
판정&rdquo;인데, 배포 과제는 &ldquo;같은 스텝에 같은 <span class="mono">p&theta;</span> 가 낸
후보들의 순위 매기기&rdquo;로 훨씬 어렵고 다른 능력을 요구한다. 효과가 있다면 후자에서
보일 것이다.</p>

{H2("예측 3 (부분 확인) &mdash; 가이던스 기여")}
<p>배포 과제에서의 판별자 기여를 직접 재려면 <b>마스킹만 쓴 arm 을 기준선</b>으로 삼아야
한다. <span class="mono">Adj+IW &minus; Adj-only</span> 가 곧 재가중이 순수하게 보탠
몫이다(v&amp;a, rawL, seen). n_is 스윕 전체를 함께 실어 <b>n 에 의존하지 않는 값인지</b>도
같이 본다.</p>
<table>
{gain_head}
{"".join(gain_rows)}
</table>
<p><b>{verdict_head}</b></p>
<ul>
{verdict_items}
</ul>
<p>즉 <b>블록 크기가 데이터셋보다 큰 변수</b>다. 그리고 그 의존이 무작위가 아니라
<b>블록 크기에 따라 부호가 뒤집힌다</b>: 작은&middot;중간 블록에서는 LE 가, 큰 블록에서는
LP 가 앞선다. 가설은 데이터셋 전체에 걸친 우열을 예측했으므로 이 구조는 가설로 설명되지
않는다. 게다가 <b>운용점은 blk2</b>(v&amp;a 0.79&ndash;0.80, 큰 블록은 0.44&ndash;0.53)이고
거기서의 차이는 {abs(g2["LP"] - g2["LE"]):.3f} 로 v&amp;a 표본오차 0.013 수준이다.</p>
{f'''<p class="small">후보 다양성(마스킹 ON, 고유 후보 수)을 나란히 두면 blk64 의 차이가
왜 그쪽에서 나는지 보인다 &mdash; 후보가 많아지는 구간에서만 재가중이 쓸 재료를 갖는다.</p>
<table>
<tr><th rowspan=2>blk</th>{"".join(f'<th class="num" colspan={len(NIS)}>{V}</th>' for V in VARS)}</tr>
<tr>{"".join("".join(f'<th class="num">n={n}</th>' for n in NIS) for V in VARS)}</tr>
{div_rows}
</table>''' if div_rows else ""}

{H2("결론")}
<p>가설은 <b>메커니즘으로는 실재하고, 결과로는 약하다.</b></p>
<ul>
<li><b>실재한다:</b> choice set 생성 방식이 참조 경로에 각인된다는 것이 네 지표에서
 일관되게 확인된다. LE 참조 경로는 최단경로에 더 가깝고(초과 홉
 {SH["LE"]["hop_excess_mean"]:.2f} 대 {SH["LP"]["hop_excess_mean"]:.2f}),
 이는 LE 가 삭제 기반이라는 정의의 직접적 귀결이다.</li>
<li><b>약하다:</b> 그 각인이 판별자 학습 난이도로는 전혀 나타나지 않고
 (val acc {dv_mean["LE"]:.3f} 대 {dv_mean["LP"]:.3f}), 가이던스 기여로도 <b>블록 크기에 따라 부호가
 뒤집힌다</b>(작은&middot;중간 블록 LE 우세 {n_le}개 / 큰 블록 LP 우세 {n_lp}개). 가설은
 데이터셋 전체의 우열을 예측하므로 이 구조를 설명하지 못한다. 운용점 blk2 에서의 차이는
 {abs(g2["LP"] - g2["LE"]):.3f} 로 표본오차 안이다.</li>
<li><b>따라서 논문에서 쓸 수 있는 형태는</b> &ldquo;LP 에서 가이던스가 더 잘 듣는다&rdquo;가
 아니라, <b>&ldquo;두 choice-set 생성 방식 모두에서 같은 결론이 재현된다&rdquo;</b>는 강건성
 진술이다. 실제로 blk2 에서 Adj+IW 가 최선인 것, 마스킹이 기여의 대부분을 설명하는 것,
 그리고 <b>운용 블록에서 n_is 가 평평한 것</b>이 LE/LP 양쪽에서 동일하게 나타난다.
 (단 &ldquo;n_is 에 평평&rdquo;은 <b>전 블록에 걸친 진술이 아니다</b> &mdash; LE blk4 와
 LP blk64 에서는 기여가 n 에 따라 단조 증가한다. n_is 스윕 보고서의 해당 절 참조.)</li>
</ul>
<p class="small"><b>제대로 검정하려면.</b> 지금 비교는 데이터셋 2 개 &times; 블록 {len(per_blk)} 개의
관측치라 블록 효과와 데이터셋 효과가 분리되지 않는다. 가설을 정면으로 재려면 LE 의
choice set 삭제 edge 집합과 except_e 삭제 집합의 <b>겹침을 통제</b>해야 한다 &mdash;
겹침이 큰 OD 쌍과 작은 OD 쌍으로 나눠 가이던스 기여를 비교하면, 같은 데이터셋 안에서
가설을 직접 검정할 수 있다. 그 정보는 상류
<span class="mono">porto_openstreetmap/path_data/v6</span> 의 choice set 생성 산출물에
있어야 한다.</p>

<hr>
<p class="small"><b>출처.</b> 데이터셋 사실
<span class="mono">tools/diag/le_vs_lp_facts.py</span> &rarr;
<span class="mono">sets_v6/le_vs_lp_facts.json</span> &middot;
최단경로 기준선 <span class="mono">tools/eval/shortest_baseline.py</span> &middot;
가이던스 기여 <span class="mono">scripts/run_v6_nis.sh</span> 의 rawL 표 &middot;
후보 다양성 <span class="mono">tools/diag/diag_proposal_diversity.py</span> &middot;
판별자 정확도는 <span class="mono">sets_v6/{{LE,LP}}/log/disc_e0_blk*.log</span> 의
마지막 스텝. 평가는 두 데이터셋 모두 동일한 1,000 OD 쌍(SPLIT_SEED 777),
<span class="mono">order=l2r</span>, <span class="mono">w_gamma=1.0</span>.</p>
</body></html>"""

    out = join(ROOT, "report_src/le_vs_lp_ko.html")
    io.open(out, "w", encoding="utf-8").write(doc)
    print(f"written {out} ({len(doc)} bytes)")
    for V in VARS:
        print(f"[{V}] hop_excess {SH[V]['hop_excess_mean']:.2f}  "
              f"dtw_to_bfs {SH[V]['dtw_to_bfs']:.0f}  "
              f"lcs_with_bfs {SH[V]['lcs_with_bfs']:.2f}  "
              f"disc_val {dv_mean[V]:.3f}")
    print("[gain mean] " + "  ".join(
        f"blk{b} LE={np.mean([x for x in (gain('LE', b, n) for n in have['LE']) if x is not None]):+.3f}"
        f"/LP={np.mean([x for x in (gain('LP', b, n) for n in have['LP']) if x is not None]):+.3f}"
        for b in BLKS_SWEEP))
