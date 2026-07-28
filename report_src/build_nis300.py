"""Insert section 7 (n_is = 300 replication of the section-4 table) into both
v4 reports and renumber the trailing sections. Every number is read from
sets_res/va_table_n300_f005.json with sets_res/va_table_v4_f005.json (n_is=100)
shown muted in parentheses.
"""
import json
import re

ROOT = "/Users/jiwooshin/Desktop/In_Progress/Block-Diffusion-OD-Planning"
BLKS = [1, 2, 4, 8, 16, 32, 64]
MET = ["valid", "arrival", "valid_and_arrival", "em", "pc"]

n300 = json.load(open(f"{ROOT}/sets_res/va_table_n300_f005.json"))
n100 = json.load(open(f"{ROOT}/sets_res/va_table_v4_f005.json"))
ess = json.load(open(f"{ROOT}/sets_res/ess_n300.json"))


def cell(new, old, bold=False):
    v = f"<b>{new:.3f}</b>" if bold else f"{new:.3f}"
    if old is None:
        return f'<td class="num">{v}</td>'
    return f'<td class="num">{v} <span class="muted">({old:.3f})</span></td>'


def table(regime, guided, hdr):
    rows = []
    for b in BLKS:
        base = n300[f"{regime}_blk{b}_base"]
        base0 = n100[f"{regime}_blk{b}_base"]
        g = n300[f"{regime}_blk{b}_{guided}"]
        g0 = n100[f"{regime}_blk{b}_{guided}"]
        c = [f"<tr><td>{b}</td>"]
        for m in MET:
            same = abs(base[m] - base0[m]) < 5e-4
            c.append(cell(base[m], None if same else base0[m], m == "valid_and_arrival"))
        for m in MET:
            c.append(cell(g[m], g0[m], m == "valid_and_arrival"))
        rows.append("".join(c) + "</tr>")
    return hdr + "\n" + "\n".join(rows)


def hdr(gl):
    h = '<tr><th>blk</th>'
    for p in ("base", gl):
        for lab in ("valid", "arr", "v∧a", "EM", "PC"):
            h += f'<th class="num">{p} {lab}</th>'
    return h + "</tr>"


def delta(regime, guided, metric):
    """max |n300 - n100| over blocks, and the signed mean."""
    d = [n300[f"{regime}_blk{b}_{guided}"][metric] - n100[f"{regime}_blk{b}_{guided}"][metric]
         for b in BLKS]
    return max(d, key=abs), sum(d) / len(d)


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.3f}"


D = {}
for reg in ("seen", "unseen"):
    for g, gk in (("adj", "adj+modelD"), ("iw", "modelD")):
        for m in ("valid_and_arrival", "em"):
            mx, mn = delta(reg, gk, m)
            D[f"{reg}_{g}_{m}_max"] = sgn(mx)
            D[f"{reg}_{g}_{m}_mean"] = sgn(mn)

ess_line = " · ".join(f"blk{b} {ess[f'blk{b}']/3:.1f}%" for b in BLKS)
ess_100 = " · ".join(f"blk{b} {ess[f'blk{b}_n100']:.1f}%" for b in BLKS)

EN = f"""<h2>7. More importance samples — n<sub>is</sub> = 300 (fam 0.05, except_0, RAW)</h2>
<p class="small">Identical protocol to §4 — same v4 backbones, same v4 discriminators, the same 1,000 reserved except_0 OD pairs, batch 100, seed 7, no post-processing — with <b>the single change n<sub>is</sub> = 100 → 300</b>. Muted values in parentheses are the corresponding n<sub>is</sub> = 100 numbers from §4. The <b>base</b> columns use neither the discriminator nor n<sub>is</sub> and are re-seeded identically, so they reproduce §4 exactly; that they do is a built-in check that nothing besides the sample count moved. <b>Cost</b>: discriminator calls per path rise from L · 100 ≈ 2,200 to L · 300 ≈ 6,600, i.e. ≈3× the guided wall-clock of §5.6.</p>
<h3>7.1 Seen (disc trained on except_0)</h3>
<table>
{table("seen", "adj+modelD", hdr("adj+IW"))}
</table>
<h3>7.1b IW only, no adjacency masking — seen</h3>
<table>
{table("seen", "modelD", hdr("IW"))}
</table>
<h3>7.2 Unseen (disc = except 1–99 → except_0; base identical to §7.1)</h3>
<table>
{table("unseen", "adj+modelD", hdr("adj+IW"))}
</table>
<h3>7.2b IW only, no adjacency masking — unseen</h3>
<table>
{table("unseen", "modelD", hdr("IW"))}
</table>
<h3>7.3 What tripling the sample count buys</h3>
<ul class="small">
<li><b>adj+IW, seen</b>: v∧a moves by {D["seen_adj_valid_and_arrival_mean"]} on average across blocks (largest single shift {D["seen_adj_valid_and_arrival_max"]}), EM by {D["seen_adj_em_mean"]} (largest {D["seen_adj_em_max"]}).</li>
<li><b>IW only, seen</b>: v∧a {D["seen_iw_valid_and_arrival_mean"]} on average (largest {D["seen_iw_valid_and_arrival_max"]}), EM {D["seen_iw_em_mean"]} (largest {D["seen_iw_em_max"]}).</li>
<li><b>Unseen mirrors seen</b>: adj+IW v∧a {D["unseen_adj_valid_and_arrival_mean"]} on average, IW-only {D["unseen_iw_valid_and_arrival_mean"]} — the zero-shot conclusion of §4.2 is unchanged at the larger sample count.</li>
<li><b>Effective sample size scales with n<sub>is</sub> rather than saturating</b>: mean ESS as a fraction of the draws is {ess_line} at n<sub>is</sub> = 300, against {ess_100} at n<sub>is</sub> = 100 — the same fraction, so the absolute ESS simply tripled with the budget. The weight distribution is therefore <i>not</i> concentrating on a few candidates; the bounded logit (|ℓ| ≤ 4) keeps the ratio within e⁸ across candidates, so extra samples buy variance reduction on an already well-conditioned estimator — which is why the table moves so little.</li>
<li><b>Practical reading</b>: at γ = 1 the SNIS estimator is already converged at n<sub>is</sub> = 100 for this task. Tripling the budget is not where the remaining headroom is; the adjacency masking of Lemma 3 (§4.1 vs §4.1b) and the discriminator itself are.</li>
</ul>"""

KO = f"""<h2>7. 중요도 표본 수 증가 — n<sub>is</sub> = 300 (fam 0.05, except_0, RAW)</h2>
<p class="small">§4와 완전히 동일한 프로토콜 — 동일 v4 백본, 동일 v4 판별자, 동일한 except_0 예약 OD 쌍 1,000개, 배치 100, 시드 7, 후처리 없음 — 에서 <b>n<sub>is</sub> = 100 → 300 한 가지만</b> 바꿨다. 괄호 안 흐린 값은 §4의 n<sub>is</sub> = 100 수치다. <b>base</b> 컬럼은 판별자도 n<sub>is</sub>도 쓰지 않고 동일하게 재시드되므로 §4를 그대로 재현하며, 실제로 그렇다는 것이 표본 수 외에는 아무것도 바뀌지 않았다는 내장 검증이 된다. <b>비용</b>: 경로당 판별자 호출이 L · 100 ≈ 2,200에서 L · 300 ≈ 6,600으로 늘어, §5.6의 guided wall-clock 대비 약 3배다.</p>
<h3>7.1 Seen (except_0로 학습한 disc)</h3>
<table>
{table("seen", "adj+modelD", hdr("adj+IW"))}
</table>
<h3>7.1b IW only, 인접성 마스킹 없음 — seen</h3>
<table>
{table("seen", "modelD", hdr("IW"))}
</table>
<h3>7.2 Unseen (disc = except 1–99 → except_0; base는 §7.1과 동일)</h3>
<table>
{table("unseen", "adj+modelD", hdr("adj+IW"))}
</table>
<h3>7.2b IW only, 인접성 마스킹 없음 — unseen</h3>
<table>
{table("unseen", "modelD", hdr("IW"))}
</table>
<h3>7.3 표본 수를 3배로 늘려 얻는 것</h3>
<ul class="small">
<li><b>adj+IW, seen</b>: v∧a가 블록 평균 {D["seen_adj_valid_and_arrival_mean"]} 이동(단일 최대 {D["seen_adj_valid_and_arrival_max"]}), EM은 {D["seen_adj_em_mean"]}(최대 {D["seen_adj_em_max"]}).</li>
<li><b>IW only, seen</b>: v∧a 평균 {D["seen_iw_valid_and_arrival_mean"]}(최대 {D["seen_iw_valid_and_arrival_max"]}), EM {D["seen_iw_em_mean"]}(최대 {D["seen_iw_em_max"]}).</li>
<li><b>Unseen도 seen과 같은 양상</b>: adj+IW v∧a 평균 {D["unseen_adj_valid_and_arrival_mean"]}, IW only {D["unseen_iw_valid_and_arrival_mean"]} — §4.2의 zero-shot 결론은 더 큰 표본 수에서도 그대로다.</li>
<li><b>유효표본수(ESS)는 포화하지 않고 n<sub>is</sub>에 비례해 늘어난다</b>: 추출 표본 대비 평균 ESS 비율이 n<sub>is</sub> = 300에서 {ess_line}이고 n<sub>is</sub> = 100에서 {ess_100}로 동일하다. 즉 절대 ESS가 예산에 비례해 그대로 3배가 됐다. 따라서 가중치 분포가 소수 후보로 집중되는 것이 <i>아니며</i>, 유계 로짓(|ℓ| ≤ 4)이 후보 간 비율을 e⁸ 안으로 묶어두므로 표본을 늘려도 이미 잘 조건화된 추정량의 분산만 줄인다 — 표가 거의 움직이지 않는 이유다.</li>
<li><b>실무적 해석</b>: γ = 1에서 SNIS 추정량은 이 과제에 대해 n<sub>is</sub> = 100이면 이미 수렴해 있다. 남은 여지는 표본 예산이 아니라 Lemma 3의 인접성 마스킹(§4.1 vs §4.1b)과 판별자 자체에 있다.</li>
</ul>"""

for lang, new in [("en", EN), ("ko", KO)]:
    p = f"{ROOT}/report_src/results_bd_v4_{lang}.html"
    s = open(p).read()
    assert "<h2>7. " in s, "section 7 anchor missing"
    # renumber trailing sections 7->8, 8->9 (headings + cross-references)
    s = re.sub(r"<h2>8\. ", "<h2>9. ", s)
    s = re.sub(r"<h2>7\. ", "<h2>8. ", s)
    s = s.replace("see §8.)", "see §9.)").replace("§8 참조.)", "§9 참조.)")
    s = s.replace("of §7 is not", "of §8 is not").replace("§7의 일반화", "§8의 일반화")
    i = s.index("<h2>8. ")
    s = s[:i] + new + "\n\n" + s[i:]
    open(p, "w").write(s)
    print(lang, "section 7 inserted, 7/8 renumbered to 8/9")
