"""Insert the reveal-order section (l2r vs first-hitting) as the last results
section of both v4 reports; 'Notes & provenance' shifts to the end.
Numbers from sets_res/va_table_l2r_f005.json with the section-4 first_hit
values (va_table_v4_f005.json) muted in parentheses.
"""
import json
import re

ROOT = "/Users/jiwooshin/Desktop/In_Progress/Block-Diffusion-OD-Planning"
BLKS = [1, 2, 4, 8, 16, 32, 64]
MET = ["valid", "arrival", "valid_and_arrival", "em", "pc"]

L = json.load(open(f"{ROOT}/sets_res/va_table_l2r_f005.json"))
F = json.load(open(f"{ROOT}/sets_res/va_table_v4_f005.json"))
ESS = {1: 99.7, 2: 98.4, 4: 98.3, 8: 97.6, 16: 95.8, 32: 95.6, 64: 94.9}


def cell(new, old, bold=False):
    v = f"<b>{new:.3f}</b>" if bold else f"{new:.3f}"
    return f'<td class="num">{v} <span class="muted">({old:.3f})</span></td>'


def table(regime, guided, gl):
    h = '<tr><th>blk</th>'
    for p in ("base", gl):
        for lab in ("valid", "arr", "v∧a", "EM", "PC"):
            h += f'<th class="num">{p} {lab}</th>'
    rows = [h + "</tr>"]
    for b in BLKS:
        c = [f"<tr><td>{b}</td>"]
        for cfg in ("base", guided):
            k = f"{regime}_blk{b}_{cfg}"
            for m in MET:
                c.append(cell(L[k][m], F[k][m], m == "valid_and_arrival"))
        rows.append("".join(c) + "</tr>")
    return "\n".join(rows)


def d(regime, cfg, m, b):
    return L[f"{regime}_blk{b}_{cfg}"][m] - F[f"{regime}_blk{b}_{cfg}"][m]


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.3f}"


ess_line = " · ".join(f"blk{b} {ESS[b]:.1f}" for b in BLKS)

EN = f"""<h2>9. Reveal order inside a block — left-to-right vs first-hitting (fam 0.05, except_0, RAW)</h2>
<p class="small">Identical protocol to §4 — same v4 backbones, same v4 discriminators, the same 1,000 reserved except_0 OD pairs, batch 100, seed 7, <b>n<sub>is</sub> = 100</b>, no post-processing — with <b>the single change <span class="mono">-order l2r</span></b>. The mask kernel commits one masked position per denoising step; <span class="mono">first_hit</span> (the default everywhere else in this report) picks that position uniformly at random among the still-masked ones, <span class="mono">l2r</span> always picks the left-most. The first-hitting <i>time</i> schedule (t ← t·u<sup>1/#masked</sup>) is untouched, so this isolates the reveal <b>position</b> choice. Unlike §7, the <b>base</b> column moves too, because unguided planning runs the same reveal loop. Muted values in parentheses are the §4 first-hitting numbers.</p>
<p class="small"><b>Noise floor (blk1)</b>: with block size 1 there is exactly one masked position per block, so both orders commit the <i>same</i> position — yet the row is not identical, because <span class="mono">first_hit</span> spends a <span class="mono">torch.multinomial</span> draw on the position choice and <span class="mono">l2r</span> does not, so the two runs diverge in RNG stream from the first step. The blk1 row therefore measures pure sampling noise at this eval size: v∧a {sgn(d("seen","base","valid_and_arrival",1))} (base), {sgn(d("seen","modelD","valid_and_arrival",1))} (IW), {sgn(d("seen","adj+modelD","valid_and_arrival",1))} (adj+IW). Read every other row against that ±0.012 floor.</p>
<h3>9.1 Seen (disc trained on except_0)</h3>
<table>
{table("seen", "adj+modelD", "adj+IW")}
</table>
<h3>9.1b IW only, no adjacency masking — seen</h3>
<table>
{table("seen", "modelD", "IW")}
</table>
<h3>9.2 Unseen (disc = except 1–99 → except_0)</h3>
<table>
{table("unseen", "adj+modelD", "adj+IW")}
</table>
<h3>9.2b IW only, no adjacency masking — unseen</h3>
<table>
{table("unseen", "modelD", "IW")}
</table>
<h3>9.3 Reading</h3>
<ul class="small">
<li><b>Left-to-right wins at every block size ≥ 2, in every configuration, and the margin grows with block size.</b> v∧a gains (seen): base {sgn(d("seen","base","valid_and_arrival",2))} at blk2 → {sgn(d("seen","base","valid_and_arrival",64))} at blk64; IW-only {sgn(d("seen","modelD","valid_and_arrival",2))} → {sgn(d("seen","modelD","valid_and_arrival",64))}; adj+IW {sgn(d("seen","adj+modelD","valid_and_arrival",2))} → {sgn(d("seen","adj+modelD","valid_and_arrival",64))}. At blk64 adj+IW v∧a goes 0.162 → 0.393, a 2.4× improvement from a change that costs nothing.</li>
<li><b>The adj+IW gain is a valid/arrival trade that comes out far ahead.</b> Validity roughly doubles or better (blk4 0.389 → 0.765; blk64 0.229 → 0.828) while arrival falls (blk4 0.873 → 0.732; blk64 0.795 → 0.511). The product still rises by {sgn(d("seen","adj+modelD","valid_and_arrival",4))} and {sgn(d("seen","adj+modelD","valid_and_arrival",64))} respectively, and EM/PC — which the trade could have hurt — rise too ({sgn(d("seen","adj+modelD","em",64))} EM, {sgn(d("seen","adj+modelD","pc",64))} PC at blk64).</li>
<li><b>Why the interaction with Lemma-3 masking is so strong</b>: the adjacency mask can only constrain a candidate when a neighbour of the target position is <i>already revealed</i> — the code falls back to an all-ones mask when the left and right neighbours are still MASK. Under <span class="mono">l2r</span> the left neighbour is revealed <i>by construction</i> at every step (it was committed on the previous step), so every single commit is restricted to a legal continuation of a real prefix. Under <span class="mono">first_hit</span> many positions are committed while both neighbours are still masked, i.e. unconstrained; later reveals then have to reconcile with those free tokens, which is where the invalid edges come from. That is why the effect is far larger for adj+IW than for IW-only or base.</li>
<li><b>The effect is not only about guidance</b>: base and IW-only gain as well, and their gains also grow with block size (base {sgn(d("seen","base","valid_and_arrival",8))} at blk8 → {sgn(d("seen","base","valid_and_arrival",64))} at blk64). Interpretation, not measurement: revealing left-to-right means every conditional the denoiser is asked for is conditioned on a <i>contiguous complete prefix</i>, which is the structure a road-network path actually has; a scattered context is a harder query. Larger blocks have more scattered-context steps to get wrong, which matches the monotone growth of the gain.</li>
<li><b>Zero-shot generalization is untouched</b>: the largest seen-vs-unseen difference anywhere in the l2r tables is 0.017, in line with §4.2. The reveal order does not interact with which scenario the discriminator was trained on.</li>
<li><b>New best cell in the report</b>: adj+IW l2r at blk2 reaches v∧a <b>0.577</b> seen / 0.574 unseen, above the previous best of 0.540 (blk1 adj+IW, §4.1) — and unlike blk1 it keeps a usable arrival rate (0.678 vs 0.567).</li>
<li><b>Cost: none.</b> l2r removes a <span class="mono">multinomial</span> draw per step and changes no scorer-call count, so the §5.6 wall-clock table applies unchanged. Mean ESS at n<sub>is</sub> = 100 is {ess_line} — marginally below the first-hitting values (99.7 … 95.5), i.e. the weights spread slightly more, not less.</li>
<li><b>Caveat</b>: single seed per cell. The blk1 row bounds the noise at ≈0.012 on v∧a, so the blk≥8 gains (0.04–0.23) are far outside it, but the blk2/blk4 base and IW-only gains (0.005–0.024) are at or near the floor and should not be read as established individually — only the monotone trend across seven block sizes is.</li>
</ul>"""

KO = f"""<h2>9. 블록 내 공개 순서 — left-to-right vs first-hitting (fam 0.05, except_0, RAW)</h2>
<p class="small">§4와 완전히 동일한 프로토콜 — 동일 v4 백본·판별자, 동일한 except_0 예약 OD 쌍 1,000개, 배치 100, 시드 7, <b>n<sub>is</sub> = 100</b>, 후처리 없음 — 에서 <b><span class="mono">-order l2r</span> 하나만</b> 바꿨다. mask 커널은 denoising 스텝마다 마스킹된 위치 하나를 확정하는데, <span class="mono">first_hit</span>(본 보고서의 다른 모든 곳에서 쓴 기본값)은 남은 마스킹 위치 중 하나를 균등 확률로 뽑고 <span class="mono">l2r</span>은 항상 가장 왼쪽을 고른다. first-hitting <i>시간</i> 스케줄(t ← t·u<sup>1/#masked</sup>)은 그대로이므로 공개 <b>위치</b> 선택만 분리된다. §7과 달리 <b>base</b> 컬럼도 움직이는데, unguided 계획도 같은 공개 루프를 돌기 때문이다. 괄호 안 흐린 값은 §4의 first-hitting 수치다.</p>
<p class="small"><b>노이즈 하한 (blk1)</b>: 블록 크기가 1이면 블록당 마스킹 위치가 정확히 하나라 두 방식이 <i>같은</i> 위치를 확정한다. 그런데도 행이 동일하지 않은데, <span class="mono">first_hit</span>은 위치 선택에 <span class="mono">torch.multinomial</span> 추출을 한 번 쓰고 <span class="mono">l2r</span>은 쓰지 않아 첫 스텝부터 난수 스트림이 갈라지기 때문이다. 따라서 blk1 행은 이 평가 규모에서의 순수 표집 잡음을 측정한다: v∧a {sgn(d("seen","base","valid_and_arrival",1))}(base), {sgn(d("seen","modelD","valid_and_arrival",1))}(IW), {sgn(d("seen","adj+modelD","valid_and_arrival",1))}(adj+IW). 나머지 행은 이 ±0.012 하한에 견주어 읽어야 한다.</p>
<h3>9.1 Seen (except_0로 학습한 disc)</h3>
<table>
{table("seen", "adj+modelD", "adj+IW")}
</table>
<h3>9.1b IW only, 인접성 마스킹 없음 — seen</h3>
<table>
{table("seen", "modelD", "IW")}
</table>
<h3>9.2 Unseen (disc = except 1–99 → except_0)</h3>
<table>
{table("unseen", "adj+modelD", "adj+IW")}
</table>
<h3>9.2b IW only, 인접성 마스킹 없음 — unseen</h3>
<table>
{table("unseen", "modelD", "IW")}
</table>
<h3>9.3 해석</h3>
<ul class="small">
<li><b>블록 크기 2 이상에서, 모든 구성에서 left-to-right이 이기고, 격차는 블록이 커질수록 벌어진다.</b> v∧a 이득(seen): base blk2 {sgn(d("seen","base","valid_and_arrival",2))} → blk64 {sgn(d("seen","base","valid_and_arrival",64))}; IW only {sgn(d("seen","modelD","valid_and_arrival",2))} → {sgn(d("seen","modelD","valid_and_arrival",64))}; adj+IW {sgn(d("seen","adj+modelD","valid_and_arrival",2))} → {sgn(d("seen","adj+modelD","valid_and_arrival",64))}. blk64에서 adj+IW v∧a가 0.162 → 0.393으로 2.4배가 되는데, 비용은 전혀 들지 않는 변경이다.</li>
<li><b>adj+IW의 이득은 valid/arrival 맞교환이지만 결과적으로 크게 남는 장사다.</b> validity가 두 배 안팎으로 뛰고(blk4 0.389 → 0.765; blk64 0.229 → 0.828) arrival은 떨어진다(blk4 0.873 → 0.732; blk64 0.795 → 0.511). 그럼에도 곱은 각각 {sgn(d("seen","adj+modelD","valid_and_arrival",4))}, {sgn(d("seen","adj+modelD","valid_and_arrival",64))} 상승하고, 맞교환에 다칠 수 있었던 EM/PC도 함께 오른다(blk64에서 EM {sgn(d("seen","adj+modelD","em",64))}, PC {sgn(d("seen","adj+modelD","pc",64))}).</li>
<li><b>Lemma-3 마스킹과의 상호작용이 큰 이유</b>: 인접성 마스크는 대상 위치의 이웃이 <i>이미 공개된</i> 경우에만 후보를 제약할 수 있다 — 좌우 이웃이 아직 MASK면 코드가 전부 1인 마스크로 되돌아간다. <span class="mono">l2r</span>에서는 왼쪽 이웃이 <i>구조적으로</i> 매 스텝 공개돼 있으므로(직전 스텝에서 확정됐다) 모든 확정이 실제 prefix의 적법한 연장으로 제한된다. <span class="mono">first_hit</span>에서는 좌우가 모두 마스킹된 상태로 확정되는 위치가 많고, 그 토큰들은 아무 제약 없이 놓인 뒤 나중 공개가 그것에 맞춰야 한다 — invalid edge가 여기서 나온다. adj+IW에서 효과가 IW only나 base보다 훨씬 큰 이유다.</li>
<li><b>guidance만의 이야기가 아니다</b>: base와 IW only도 이득을 보고 그 이득 역시 블록 크기에 따라 커진다(base blk8 {sgn(d("seen","base","valid_and_arrival",8))} → blk64 {sgn(d("seen","base","valid_and_arrival",64))}). 측정이 아닌 해석을 붙이자면: 왼쪽부터 공개하면 denoiser에게 묻는 모든 조건부가 <i>연속된 완전한 prefix</i>를 조건으로 갖는데, 이것이 도로망 경로가 실제로 가진 구조다. 흩어진 문맥은 더 어려운 질의이고, 큰 블록일수록 그런 스텝이 많아 틀릴 여지가 크다 — 이득이 단조 증가하는 것과 일치한다.</li>
<li><b>zero-shot 일반화는 그대로다</b>: l2r 표 전체에서 seen–unseen 최대 차이가 0.017로 §4.2와 같은 수준이다. 공개 순서는 판별자가 어느 시나리오로 학습됐는지와 상호작용하지 않는다.</li>
<li><b>보고서 전체 최고 셀 경신</b>: adj+IW l2r blk2가 v∧a <b>0.577</b>(seen) / 0.574(unseen)로, 종전 최고 0.540(blk1 adj+IW, §4.1)을 넘는다. 게다가 blk1과 달리 쓸 만한 arrival(0.678 vs 0.567)을 유지한다.</li>
<li><b>비용: 없음.</b> l2r은 스텝당 <span class="mono">multinomial</span> 추출을 하나 없애고 채점 호출 수는 바꾸지 않으므로 §5.6의 wall-clock 표가 그대로 적용된다. n<sub>is</sub> = 100에서 평균 ESS는 {ess_line}로 first-hitting(99.7 … 95.5)보다 근소하게 낮다 — 가중치가 오히려 조금 더 퍼진다.</li>
<li><b>주의</b>: 칸마다 시드 하나다. blk1 행이 v∧a 잡음을 ≈0.012로 묶어주므로 blk8 이상의 이득(0.04–0.23)은 확실히 그 밖이지만, blk2/blk4의 base·IW only 이득(0.005–0.024)은 하한에 걸쳐 있어 개별적으로 확립된 것으로 읽으면 안 된다 — 일곱 개 블록에 걸친 단조 추세만이 근거다.</li>
</ul>"""

for lang, new in [("en", EN), ("ko", KO)]:
    p = f"{ROOT}/report_src/results_bd_v4_{lang}.html"
    s = open(p).read()
    assert "<h2>9. " in s and "<h2>10." not in s, "unexpected numbering"
    s = re.sub(r"<h2>9\. ", "<h2>10. ", s)          # Notes & provenance -> 10
    s = s.replace("see §9.)", "see §10.)").replace("§9 참조.)", "§10 참조.)")
    i = s.index("<h2>10. ")
    s = s[:i] + new + "\n\n" + s[i:]
    open(p, "w").write(s)
    print(lang, "section 9 inserted; notes -> 10")
