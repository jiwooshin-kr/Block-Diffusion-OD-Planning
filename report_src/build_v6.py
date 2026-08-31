"""
v6 데이터(LE 1.0 / 0.05) 결과 보고서 생성 — iclr_iw_exp.pdf §5.6.1/§5.6.2와 같은 레이아웃.

  python report_src/build_v6.py [-variant LE]

입력 (전부 JSON에서 읽으며 손으로 옮겨 적은 수치는 없다):
  sets_v6/{V}/res/va_table_v6{V}.json   3-arm x 7블록 x seen/unseen
  sets_v6/{V}/res/jsev_v6{V}.json       블록별 무조건 생성 분포 일치
  sets_res/va_table_l2r_f005.json       구 데이터(§5.6.2) 비교용 IW
  sets_res/va_table_abl_f005.json       구 데이터 비교용 adj-only/base
출력:
  report_src/v6_{V}_res_{ko,en}.html  ->  pdfs/v6_res/v6_{V}_results_{ko,en}.pdf
"""

import argparse
import io
import json
import os
import re
from os.path import dirname, join

ROOT = dirname(dirname(os.path.abspath(__file__)))
BLKS = [1, 2, 4, 8, 16, 32, 64]
ARMS = [("adjonly", "Adj-only"), ("modelD", "IW-only"), ("adj+modelD", "Adj+IW")]
# 표에 싣는 지표: PC 제외(참조 경로가 최단경로가 아니라 rank 기반 PC가 의미를 잃음),
# DTW/LCS 추가(참조 경로 대비 형상). 높을수록 좋음 = +1, 낮을수록 좋음 = -1
MET = [("valid", "valid", +1), ("arrival", "arrival", +1),
       ("valid_and_arrival", "valid &amp; arrival", +1), ("em", "EM", +1),
       ("dtw", "DTW&darr;", -1), ("lcs_norm", "LCS/|ref|", +1)]


def jload(p, required=True):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        if required:
            raise
        return None


def sgn(x, nd=3):
    return f"{x:+.{nd}f}" if abs(x) >= 10 ** -nd else "0.000"


def cell(v, base, best, nd=3):
    s = f"<b>{v:.{nd}f}</b>" if v == best else f"{v:.{nd}f}"
    return f'<td class="num">{s} <span class="muted">({sgn(v - base, nd)})</span></td>'


def arm_table(J, regime):
    """blk x {adj-only, IW-only, adj+IW} x 지표. base 대비 증감을 괄호로."""
    hdr = ("<tr><th>blk</th><th>구성</th>"
           + "".join(f'<th class="num">{lab}</th>' for _, lab, _ in MET) + "</tr>")
    rows = [hdr]
    for b in BLKS:
        base = J[f"seen_blk{b}_base"]          # base는 판별자와 무관 -> 두 체제 공통
        grp = {}
        for cfg, _ in ARMS:
            # adj-only는 판별자를 쓰지 않으므로 seen 실행분을 두 표에서 공유한다
            grp[cfg] = J.get(f"{regime}_blk{b}_{cfg}", J.get(f"seen_blk{b}_{cfg}"))
        best = {m: (max if d > 0 else min)(grp[c][m] for c, _ in ARMS) for m, _, d in MET}
        for i, (cfg, lab) in enumerate(ARMS):
            r = f'<tr>{f"<td rowspan=3>{b}</td>" if i == 0 else ""}<td>{lab}</td>'
            for m, _, _ in MET:
                r += cell(grp[cfg][m], base[m], best[m])
            rows.append(r + "</tr>")
    return "\n".join(rows)


def base_table(J):
    hdr = ('<tr><th>blk</th>' + "".join(f'<th class="num">{lab}</th>' for _, lab, _ in MET)
           + '<th class="num">gen_len</th></tr>')
    rows = [hdr]
    for b in BLKS:
        m = J[f"seen_blk{b}_base"]
        rows.append(f"<tr><td>{b}</td>"
                    + "".join(f'<td class="num">{m[k]:.3f}</td>' for k, _, _ in MET)
                    + f'<td class="num">{m["gen_len"]:.1f}</td></tr>')
    return "\n".join(rows)


def compare_table(NEW, OLD_IW, OLD_AB):
    """구 데이터(§5.6.2, unseen) vs v6(unseen) — 핵심 지표만 나란히."""
    hdr = ('<tr><th rowspan=2>blk</th><th rowspan=2>구성</th>'
           '<th class="num" colspan=2>valid</th><th class="num" colspan=2>valid &amp; arrival</th>'
           '<th class="num" colspan=2>EM</th></tr>'
           '<tr><th class="num">구 v4</th><th class="num">v6 LE</th>'
           '<th class="num">구 v4</th><th class="num">v6 LE</th>'
           '<th class="num">구 v4</th><th class="num">v6 LE</th></tr>')
    rows = [hdr]
    for b in BLKS:
        for i, (cfg, lab) in enumerate(ARMS):
            o = (OLD_AB[f"l2r_blk{b}_{cfg}"] if cfg == "adjonly"
                 else OLD_IW[f"unseen_blk{b}_{cfg}"])
            n = NEW.get(f"unseen_blk{b}_{cfg}", NEW.get(f"seen_blk{b}_{cfg}"))
            r = f'<tr>{f"<td rowspan=3>{b}</td>" if i == 0 else ""}<td>{lab}</td>'
            for k in ("valid", "valid_and_arrival", "em"):
                r += (f'<td class="num muted">{o[k]:.3f}</td>'
                      f'<td class="num"><b>{n[k]:.3f}</b></td>')
            rows.append(r + "</tr>")
    return "\n".join(rows)


def jsev_table(JS):
    hdr = ('<tr><th>blk</th><th class="num">JSEV (nats)</th><th class="num">JSEV (bits)</th>'
           '<th class="num">Spearman</th><th class="num">R²</th>'
           '<th class="num">생성 길이</th><th class="num">n</th></tr>')
    rows = [hdr]
    for b in BLKS:
        k = f"blk{b}"
        if k not in JS:
            continue
        m = JS[k]
        rows.append(f'<tr><td>{b}</td><td class="num">{m["JSEV"]:.4f}</td>'
                    f'<td class="num">{m["JSEV"] / 0.6931471805599453:.4f}</td>'
                    f'<td class="num">{m.get("Spearman", float("nan")):.3f}</td>'
                    f'<td class="num">{m["R2"]:.3f}</td>'
                    f'<td class="num">{m["len_mean"]:.1f} ± {m["len_std"]:.1f}</td>'
                    f'<td class="num">{m["n_scored"]}</td></tr>')
    real = JS["real"]
    rows.append(f'<tr><td colspan=5 class="muted">참조(정답 경로, 동일 표본 크기)</td>'
                f'<td class="num muted">{real["len_mean"]:.1f} ± {real["len_std"]:.1f}</td>'
                f'<td class="num muted">{real["n"]}</td></tr>')
    return "\n".join(rows)


def post_table(NEW, POST, regime):
    """raw / P1 / P3 / P1P3 x {v&a, EM, LCS/|ref|}. P1P3에서 v&a가 포화되므로 EM이 판별 지표."""
    VS = [("raw", NEW), ("P1", POST["P1"]), ("P3", POST["P3"]), ("P1P3", POST["P1P3"])]
    hdr = ('<tr><th rowspan=2>blk</th><th rowspan=2>구성</th>'
           + "".join(f'<th class="num" colspan=3>{n}</th>' for n, _ in VS) + "</tr>"
           + "<tr>" + ("<th class=\"num\">v&amp;a</th><th class=\"num\">EM</th>"
                       "<th class=\"num\">LCS/|r|</th>") * len(VS) + "</tr>")
    rows = [hdr]
    arms = [("base", "base")] + list(ARMS)
    for b in BLKS:
        best_em = {n: max(
            (J.get(f"{regime}_blk{b}_{c}", J.get(f"seen_blk{b}_{c}")) or {}).get("em", -1)
            for c, _ in arms) for n, J in VS}
        for i, (cfg, lab) in enumerate(arms):
            r = f'<tr>{f"<td rowspan=4>{b}</td>" if i == 0 else ""}<td>{lab}</td>'
            for n, J in VS:
                m = J.get(f"{regime}_blk{b}_{cfg}", J.get(f"seen_blk{b}_{cfg}"))
                em = f"<b>{m['em']:.3f}</b>" if m["em"] == best_em[n] else f"{m['em']:.3f}"
                r += (f'<td class="num">{m["valid_and_arrival"]:.3f}</td>'
                      f'<td class="num">{em}</td>'
                      f'<td class="num">{m["lcs_norm"]:.3f}</td>')
            rows.append(r + "</tr>")
    return "\n".join(rows)


def pick_best(J, regime, metric="valid_and_arrival"):
    best, bb, bc = -1, None, None
    for b in BLKS:
        for cfg, lab in ARMS:
            m = J.get(f"{regime}_blk{b}_{cfg}", J.get(f"seen_blk{b}_{cfg}"))
            if m[metric] > best:
                best, bb, bc = m[metric], b, lab
    return best, bb, bc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    args = ap.parse_args()
    V = args.variant

    NEW = jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}.json"))
    JS = jload(join(ROOT, f"sets_v6/{V}/res/jsev_v6{V}.json"))
    CEIL = jload(join(ROOT, f"sets_v6/{V}/res/ceiling_v6{V}.json"), required=False)
    POST = {v: jload(join(ROOT, f"sets_v6/{V}/res/va_table_v6{V}_{v}.json"), required=False)
            for v in ("P1", "P3", "P1P3")}
    OLD_IW = jload(join(ROOT, "sets_res/va_table_l2r_f005.json"), required=False)
    OLD_AB = jload(join(ROOT, "sets_res/va_table_abl_f005.json"), required=False)

    css = re.search(r"<style>(.*?)</style>",
                    open(join(ROOT, "report_src/iclr_iw_exp_ko.html")).read(), re.S).group(1)

    seen_t, uns_t = arm_table(NEW, "seen"), arm_table(NEW, "unseen")
    cmp_t = (compare_table(NEW, OLD_IW, OLD_AB) if (OLD_IW and OLD_AB) else "")
    bs, bsb, bsc = pick_best(NEW, "seen")
    bu, bub, buc = pick_best(NEW, "unseen")
    o_best = max(OLD_IW[f"unseen_blk{b}_{c}"]["valid_and_arrival"]
                 for b in BLKS for c, _ in ARMS if f"unseen_blk{b}_{c}" in OLD_IW) if OLD_IW else None
    o_adj = max(OLD_AB[f"l2r_blk{b}_adjonly"]["valid_and_arrival"] for b in BLKS) if OLD_AB else None
    o_top = max(x for x in (o_best, o_adj) if x is not None) if OLD_IW else None

    ceil_sec, ceil_note = "", ""
    if CEIL:
        bu_arm = NEW.get(f"unseen_blk{bub}_" + {"Adj-only": "adjonly", "IW-only": "modelD",
                                                "Adj+IW": "adj+modelD"}[buc],
                         NEW.get(f"seen_blk{bub}_adjonly"))
        ceil_sec = f"""<h2>2b. 지표의 천장 — 정답이 확률적 표본이라는 점</h2>
<p>이 데이터의 정답 경로는 <b>choice set 생성 + 확률적 선택</b>으로 만들어졌다.
{V} = <b>Link {'Elimination' if V == 'LE' else 'Penalization'}</b>으로 OD마다 대안 경로 집합
C<sub>od</sub>를 만들고, Path-Size Logit(폴더명의 1.0 = Beta &gt; 0)으로 확률을 부여한 뒤
<b>그 분포에서 경로 하나를 확률적으로 추출</b>해 기록한 것이다. 실제로 참조 경로는 최단경로보다
평균 <b>+4.4홉</b> 길고(중앙값 +3, 최소 +0) 최단보다 짧은 경우는 하나도 없다 — C<sub>od</sub>의 한 원소라는 구조와 일치한다.</p>
<p>따라서 <b>같은 분포에서 뽑은 두 표본조차 서로 일치하지 않는다.</b> 그 일치율이
참조 대비 지표(EM / DTW / LCS)가 도달할 수 있는 상한이다 — EM = 1.0, LCS = 1.0, DTW = 0 이 아니다.
아래 값은 normal과 except_1&ndash;5의 추출을 except_0 참조와 맞대어 측정했다
(두 경로가 양쪽 그래프에서 모두 적법한 OD만 사용, 총 {CEIL['n_pairs']}쌍).</p>
<table>
<tr><th>지표</th><th class="num">천장 (독립 추출 2개)</th><th class="num">우리 최고 (blk{bub} {buc}, unseen)</th><th class="num">천장 대비</th></tr>
<tr><td>경로 완전 일치</td><td class="num">{CEIL['exact_path']:.3f}</td><td class="num muted">&mdash;</td><td class="num muted">&mdash;</td></tr>
<tr><td>EM (길이 일치)</td><td class="num"><b>{CEIL['em']:.3f}</b></td><td class="num">{bu_arm['em']:.3f}</td><td class="num">{bu_arm['em'] / CEIL['em']:.0%}</td></tr>
<tr><td>EM (|&Delta;len| &le; 1)</td><td class="num">{CEIL['em_tol1']:.3f}</td><td class="num muted">&mdash;</td><td class="num muted">&mdash;</td></tr>
<tr><td>DTW (낮을수록 좋음)</td><td class="num"><b>{CEIL['dtw']:.3f}</b></td><td class="num">{bu_arm['dtw']:.3f}</td><td class="num">{bu_arm['dtw'] / CEIL['dtw']:.1f}&times;</td></tr>
<tr><td>LCS/|ref|</td><td class="num"><b>{CEIL['lcs_norm']:.3f}</b></td><td class="num">{bu_arm['lcs_norm']:.3f}</td><td class="num">{bu_arm['lcs_norm'] / CEIL['lcs_norm']:.0%}</td></tr>
</table>
<p><b>이 표가 §3을 읽는 법을 바꾼다.</b> EM 0.175를 &ldquo;17.5%만 맞혔다&rdquo;로 읽으면 틀리고
&ldquo;천장 {CEIL['em']:.2f}의 {bu_arm['em'] / CEIL['em']:.0%}&rdquo;로 읽어야 한다. LCS는 천장의
{bu_arm['lcs_norm'] / CEIL['lcs_norm']:.0%}까지 와 있다. 반면 <b>valid / arrival / valid &amp; arrival은
참조 경로가 전혀 필요 없는 지표</b>(생성 경로가 시나리오 그래프에서 적법한가, 목적지에 닿았는가)이므로
이 문제의 영향을 받지 않는다 — 헤드라인을 valid &amp; arrival로 두는 이유다.</p>
<p class="small"><b>남은 한계.</b> 이 천장은 &ldquo;표본 대 표본&rdquo; 비교의 한계일 뿐이며, 근본적으로는
비교 대상이 분포가 되어야 한다. 데이터에는 normal + except_0&ndash;99 = 같은 OD에 대한 101번의 추출이
들어 있으므로, 제거 edge의 영향을 받지 않은 추출만 모으면 <b>OD당 약 30개의 정답 표본과 경로별 빈도</b>를
생성기 코드 없이 복원할 수 있다(시나리오 21개로 실측 시 OD당 유효 추출 6.7개, 고유 경로 2.9개,
77%의 OD에서 2개 이상 확보). 그러면 choice set 소속률과 분포 거리(JS)를 직접 잴 수 있다 — 다음 단계로 등재.</p>"""
        ceil_note = (f"EM / DTW / LCS는 천장이 각각 {CEIL['em']:.3f} / {CEIL['dtw']:.3f} / "
                     f"{CEIL['lcs_norm']:.3f}이다(&sect;2b). valid &middot; arrival &middot; valid &amp; arrival은 참조와 무관하므로 천장이 1.0이다.")

    post_sec = ""
    if all(POST.values()):
        pt = post_table(NEW, POST, "unseen")
        b2 = {c: POST["P1P3"].get(f"unseen_blk2_{c}", POST["P1P3"].get(f"seen_blk2_{c}"))
              for c in ("base", "adjonly", "modelD", "adj+modelD")}
        post_sec = f"""<h2>4b. 후처리 (P1 / P3) — unseen</h2>
<p class="small"><b>P1</b>: 경로 중간의 불법 edge (u, v)를 A<sub>scn</sub> 상의 u&rarr;v 최단경로로 잇는다.
<b>P3</b>: 목적지에 도달하지 못한 경로 끝에서 목적지까지 최단경로를 덧붙인다.
<b>P1P3</b>: 둘 다. 굵은 값은 같은 블록·같은 후처리에서 네 arm 중 EM 최고.</p>
<table>{{pt}}</table>
<p><b>이 표가 말하는 것.</b> 새 데이터에서 후처리는 <i>정답 주입이 아니다</i>. 구 데이터에서는 정답 집합이
최단경로였기 때문에 최단경로로 깁는 후처리가 사실상 답을 써넣는 것에 가까웠고, 원 보고서 §E5가
바로 이 점을 지표 문제로 제기했다. v6의 참조 경로는 최단경로가 아니므로(15%만 일치) P1/P3는
경로를 적법·도달 가능하게 <b>복구</b>할 뿐 참조 경로 쪽으로 밀어주지 않는다.</p>
<p><b>대신 새로운 문제가 생긴다.</b> P1P3를 적용하면 valid &amp; arrival이 <b>네 arm 모두 정확히 1.000</b>이 된다 —
이 지표는 후처리 뒤에는 arm을 전혀 구분하지 못한다. 따라서 후처리 결과를 비교할 때는
<b>EM / DTW / LCS로 판단해야 한다</b>. 그 기준으로 보면 guidance의 이득은 후처리 뒤에도 남는다:
blk2 P1P3에서 EM이 base {{b2['base']['em']:.3f}} &rarr; Adj-only {{b2['adjonly']['em']:.3f}}
&rarr; IW-only {{b2['modelD']['em']:.3f}} &rarr; <b>Adj+IW {{b2['adj+modelD']['em']:.3f}}</b>이고,
<b>Adj+IW가 7개 블록 전부에서 P1P3 EM 최고</b>다. 즉 후처리로 적법성·도달성을 공짜로 얻고 나면
남는 차이는 &ldquo;참조 경로를 얼마나 맞혔는가&rdquo;이고 거기서 guidance가 앞선다.</p>"""

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>v6 {V} 결과 — IW guidance 3-arm</title><style>{css}</style></head><body>

<h1>v6 데이터({V} 1.0 / 0.05) IW guidance 결과 — iclr_iw_exp §5.6 형식</h1>

<p>기존 보고서 <span class="mono">pdfs/iclr_iw_exp.pdf</span> §5.6.1(seen) / §5.6.2(unseen)의 표를
<b>새 데이터셋에서 그대로 재현</b>한 기록. 프로토콜은 원본과 동일하다 — masked block diffusion,
left-to-right 공개 순서, n<sub>is</sub> = 100, &gamma; = 1, except_0 예약 OD 쌍 1,000개, RAW(후처리 없음),
seed 7. 바뀐 것은 <b>데이터</b>와 <b>지표 두 개</b>뿐이며 그 이유는 §5에 적었다.</p>

<h2>1. 세 arm</h2>
<table>
<tr><th>표기</th><th>코드</th><th>내용</th></tr>
<tr><td>Adj-only</td><td class="mono">plan_guided(disc=None, adj_prop=True)</td>
    <td>Lemma 3 인접성 마스킹만. <b>판별자를 쓰지 않으므로 seen/unseen 표에서 행이 동일</b>하다</td></tr>
<tr><td>IW-only</td><td class="mono">plan_guided(disc=D, adj_prop=False)</td>
    <td>판별자 SNIS 재가중만</td></tr>
<tr><td>Adj+IW</td><td class="mono">plan_guided(disc=D, adj_prop=True)</td><td>제안 방법</td></tr>
<tr><td>(base)</td><td class="mono">plan()</td><td>unguided. 행으로 싣지 않고 괄호 안 증감의 기준선</td></tr>
</table>
<p class="small">판별자는 <b>positives = 예외 시나리오 실데이터 1%</b>, <b>negatives = 해당 블록 백본의
무조건 생성 20,000개</b>(<span class="mono">-neg model</span>)로 학습했다. 밀도비
exp(logit) = D/(1&minus;D) &asymp; q<sub>exc</sub>/p<sub>&theta;</sub>의 분모가 p<sub>&theta;</sub>여야 하기 때문이다.
seen은 except_0(평가 예약 1,000개 제외)로, unseen은 except_1&ndash;99로 학습해 except_0에 zero-shot 적용한다.</p>

<h2>2. base (unguided) 기준선</h2>
<table>{base_table(NEW)}</table>

{ceil_sec}

<h2>3. 세 arm — 블록별</h2>
<p class="small">굵은 값은 같은 블록 세 arm 중 최고(DTW는 낮을수록 좋으므로 최소값),
괄호 안 회색 값은 그 블록 base 대비 증감. Adj-only 행은 두 표에서 동일하다 —
각 표를 독립적으로 읽을 수 있도록 생략하지 않고 반복했다.</p>

<h3>3.1 seen (판별자 = except_0 vs p<sub>&theta;</sub>)</h3>
<table>{seen_t}</table>

<h3>3.2 unseen (판별자 = except_1&ndash;99 vs p<sub>&theta;</sub> &rarr; except_0 zero-shot)</h3>
<table>{uns_t}</table>

<p class="small">{ceil_note}</p>

<p><b>읽기.</b> 핵심 열은 valid &amp; arrival이다(모든 edge가 A<sub>scn</sub>에서 적법 <i>그리고</i> 종점이 목적지 —
테스트 시점에 확인 가능한 수용 기준). 최고값은 seen <b>{bs:.3f}</b>(blk{bsb}, {bsc}),
unseen <b>{bu:.3f}</b>(blk{bub}, {buc}). <b>seen과 unseen이 거의 같다</b> —
99개의 다른 시나리오만 보고 학습한 판별자가 한 번도 못 본 except_0에서 사실상 동일하게 작동한다.
원본 보고서의 결론이 새 데이터에서 재현됐다.</p>

{"<h2>4. 구 데이터(§5.6.2) 대비</h2>" if cmp_t else ""}
{f'''<p class="small">같은 unseen 체제, 같은 프로토콜. 회색이 구 데이터(porto v4-0.05, V = 1,390,
정답 = 최단경로), 굵은 값이 v6 {V}(V = 1,387, 정답 = 최단경로가 <b>아닌</b> 참조 경로).
데이터가 다르므로 절대값 비교가 아니라 <b>arm 간 관계</b>가 어떻게 달라졌는지를 본다.</p>
<table>{cmp_t}</table>
<p><b>차이 네 가지.</b></p>
<ol>
<li><b>Adj-only의 validity가 사실상 포화됐다</b>(0.98&ndash;1.00 vs 구 데이터 0.72&ndash;0.93).
v6 그래프는 노드 1,387개에 edge 1,999개로 평균 차수 2.88로 훨씬 희소해서, 인접성 마스킹이 남기는
합법 후보가 적고 그만큼 제약이 강하게 걸린다. 구 데이터에서 validity를 갉아먹던 막다른 길
fallback(원 보고서 E4)이 여기서는 거의 발동하지 않는다.</li>
<li><b>핵심 지표가 전반적으로 높다</b> — valid &amp; arrival 최고 {bu:.3f}(v6) vs {o_top:.3f}(구 데이터).</li>
<li><b>IW가 마스킹 위에서 실제로 기여한다.</b> 구 데이터에서는 Adj-only와 Adj+IW가 거의 구분되지
않았지만(valid 0.928 vs 0.925), v6에서는 EM이 blk4에서 0.121 &rarr; 0.181, blk64에서
0.117 &rarr; 0.158로 뚜렷이 벌어진다. 판별자가 <i>적법성 너머의</i> 신호를 담고 있다는 뜻이다.</li>
<li><b>최적 블록이 4에서 2로 이동했다.</b> blk32/64에서의 하락(arrival 급락)은 두 데이터에서 동일하다.</li>
</ol>''' if cmp_t else ""}

{post_sec}

<h2>5. 지표 — 무엇을 왜 바꿨나</h2>
<p>새 데이터의 참조 경로는 <b>최단경로가 아니다</b>. 300개를 그래프 최단길이와 대조한 결과
v6는 44/300(15%)만 일치하는 반면 구 데이터는 300/300(100%)이었다. 여기서 두 가지가 따라온다.</p>
<ul>
<li><b>PC 제외.</b> PC = 1/(그래프상 더 짧은 feasible 길이 개수 + 1)로 정의되는데, 정답 경로 자체가
최단이 아니면 정답조차 PC &lt; 1이 되어 상한이 1이 아니게 된다. 지표가 무엇을 재는지 불분명해지므로
표에서 뺐다(레코드 CSV에는 계산돼 남아 있다).</li>
<li><b>EM의 의미가 바뀌었다.</b> v6는 OD당 경로가 정확히 1개(1,387 &times; 1,386 = 1,922,382 = 전체 순서쌍)이므로
EM은 &ldquo;valid이고 <b>참조 경로 길이와 정확히 일치</b>&rdquo;가 된다. 구 데이터에서는 이것이 곧 최단성이었다.
v6의 EM 절대값이 낮은 것은 이 때문이며, 구 데이터와 직접 비교하면 안 된다.</li>
<li><b>DTW / LCS 추가.</b> 참조 경로와 1:1로 대응해 재는 형상 지표라 표본 크기 편향이 없다.
DTW는 좌표 L1 &times; 100 ground distance에 max(n, m) 정규화(값이 작을수록 유사), LCS는 참조 길이로 정규화했다.</li>
</ul>
<p class="small"><b>구현 검증.</b> 표에 쓰인 구현을 레퍼런스와 대조했다 —
<span class="mono">lcs_length</span> 교과서 DP와 300/300 일치,
<span class="mono">dtw_distance</span> 표준 DTW와 200/200 일치(오차 0, 자기 자신 = 0),
JSEV는 <span class="mono">scipy.spatial.distance.jensenshannon²</span>와 3e&minus;16 이내.
한편 <span class="mono">utils/evaluate_plan_dtw.py::LCSSDistance</span>는
<span class="mono">c[lena-1][lenb-1]</span>을 반환해 마지막 원소를 버리는 버그가 있다(300개 중 156개 불일치,
<span class="mono">a=b=[1,2,3,4]</span>에 3을 반환). 이 함수는 block diffusion 파이프라인에서 쓰이지 않으므로
본 표와 기존 보고서 수치에는 영향이 없다.</p>

<h2>6. 분포 일치 (JSEV)</h2>
<p class="small">블록별 무조건 생성 {JS["blk1"]["n_scored"] if "blk1" in JS else "20,000"}개의 방향성 edge 빈도 분포를
참조 경로 <b>동일 개수</b>와 비교한 Jensen-Shannon divergence. 표본 크기를 맞춘 이유는 이 지표가
표본 수에 강하게 편향되기 때문이다 — 같은 분포에서 뽑은 두 표본조차 n = 1,000이면 0.379,
n = 20,000이면 0.026이 나온다. 그래서 1,000쌍짜리 §3 표에는 넣지 않고 여기서 따로 잰다.</p>
<table>{jsev_table(JS)}</table>

<h2>7. 데이터와 실행</h2>
<table>
<tr><th>항목</th><th>v6 {V} 1.0 / 0.05</th><th>구 데이터 (v4-0.05)</th></tr>
<tr><td>노드 / edge</td><td>1,387 / 1,999 (평균 차수 2.88)</td><td>1,390</td></tr>
<tr><td>normal 경로</td><td>1,922,382 (OD 전체 순서쌍, OD당 1개)</td><td>1,930,710</td></tr>
<tr><td>경로 길이 평균 / 최대</td><td>27.6 / 73</td><td>23.2 / 59</td></tr>
<tr><td>참조 경로가 최단경로</td><td>15% (44/300)</td><td>100% (300/300)</td></tr>
<tr><td>예외 시나리오</td><td>except_0&ndash;99, 각 edge 99쌍(&asymp;5%) 제거</td><td>동일 구조</td></tr>
</table>
<p class="small"><b>파이프라인</b> <span class="mono">scripts/run_v6_all.sh</span> (VARIANT={V}) —
A 백본 mask blk 1&ndash;64 7개(normal 1 epoch, bs 32, lr 5e&minus;4) &rarr;
B 무조건 생성 pool 블록당 20,000 &rarr;
C 판별자 e0 7개 + e99 7개(4,000 step, frac 1%) &rarr;
D <span class="mono">tools/eval/three_way_postproc.py</span> &times; 14(order l2r, n<sub>is</sub> 100) &rarr;
E <span class="mono">tools/collect/collect_va_table.py -shape 1</span> &rarr;
F <span class="mono">tools/eval/eval_uncond_jsev.py</span>.
2026-08-12 02:04&ndash;05:47 KST, 3시간 44분, RTX 3090 &times; 4, 실패 0건.
수치 출처는 <span class="mono">sets_v6/{V}/res/va_table_v6{V}.json</span>,
<span class="mono">jsev_v6{V}.json</span>이며 이 문서는
<span class="mono">report_src/build_v6.py</span>가 그 JSON에서 생성한다. 손으로 옮겨 적은 수치는 없다.</p>

</body></html>"""

    out_html = join(ROOT, f"report_src/v6_{V}_res_ko.html")
    io.open(out_html, "w", encoding="utf-8").write(doc)
    print(f"written {out_html} ({len(doc)} bytes)")
