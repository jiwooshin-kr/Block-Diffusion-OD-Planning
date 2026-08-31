"""Paper figure: the full discriminator pipeline (BDDiscriminator).
Emits report_src/disc_pipeline_{en,ko}.html -> pdfs/DISC_PIPELINE{,_KO}.pdf
Every architectural claim traces to models_seq/bd_disc.py, tools/train/train_bd_disc.py,
models_seq/bd_models.py::_denoise_block_mask_guided, tools/eval/disc_gen_eval.py.
"""

W, H = 1200, 790
INK, MUT, LINE = "#1f2933", "#616e7c", "#9aa5b1"
ACC, ACCF = "#b45309", "#fff8ec"          # scenario-conditioning highlight
GRY = "#f7f9fa"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, lines, accent=False, mono_from=None, fill=None):
    st, fl = (ACC, ACCF) if accent else (LINE, fill or "#ffffff")
    o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fl}" '
         f'stroke="{st}" stroke-width="{1.6 if accent else 1.1}"/>']
    cx = x + w / 2
    ty = y + 21
    o.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" class="bt" '
             f'fill="{ACC if accent else INK}">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        cls = "mn" if (mono_from is not None and i >= mono_from) else "bl"
        o.append(f'<text x="{cx}" y="{ty + 19 + i * 15.5}" text-anchor="middle" '
                 f'class="{cls}">{esc(ln)}</text>')
    return "\n".join(o)


def vlink(x, y1, y2, label=None):
    o = [f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 7}" stroke="{MUT}" '
         f'stroke-width="1.3" marker-end="url(#ar)"/>']
    if label:
        o.append(f'<text x="{x + 7}" y="{(y1 + y2) / 2 + 4}" class="ed">{esc(label)}</text>')
    return "\n".join(o)


def path_arrow(d, dashed=False, color=None):
    c = color or MUT
    da = ' stroke-dasharray="4 3"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.3"{da} '
            f'marker-end="url({"#ara" if color else "#ar"})"/>')


def panel(x, w, num, title):
    return (f'<rect x="{x}" y="56" width="{w}" height="654" rx="9" fill="none" '
            f'stroke="{LINE}" stroke-width="1" stroke-dasharray="5 4"/>'
            f'<text x="{x + 14}" y="45" class="pt">{esc(num + "  " + title)}</text>')


def build(L):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'''<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,1 L9,5 L0,9 z" fill="{MUT}"/></marker>
  <marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,1 L9,5 L0,9 z" fill="{ACC}"/></marker>
</defs>''',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="24" y="30" class="ti">{esc(L["title"])}</text>']

    # ---------------- panel A : example construction -------------------
    AX, AW = 24, 250
    BXX, BW = 306, 520
    CX, CW = 858, 318
    s.append(panel(AX, AW, "①", L["pA"]))
    s.append(panel(BXX, BW, "②", L["pB"]))
    s.append(panel(CX, CW, "③", L["pC"]))

    ax, aw = AX + 16, AW - 32
    acx = ax + aw / 2
    s.append(box(ax, 76, aw, 62, L["a1t"], L["a1"]))
    s.append(box(ax, 158, aw, 62, L["a2t"], L["a2"]))
    s.append(vlink(acx - 40, 138, 250))
    s.append(vlink(acx + 40, 220, 250))
    s.append(box(ax, 250, aw, 58, L["a3t"], L["a3"], accent=False, mono_from=1))
    s.append(vlink(acx, 308, 328))
    s.append(box(ax, 328, aw, 58, L["a4t"], L["a4"], mono_from=1))
    s.append(vlink(acx, 386, 406))
    s.append(box(ax, 406, aw, 58, L["a5t"], L["a5"]))
    s.append(box(ax, 500, aw, 74, L["a6t"], L["a6"], accent=True))
    s.append(f'<text x="{acx}" y="{592}" text-anchor="middle" class="cap">{esc(L["acap"])}</text>')
    s.append(box(ax, 616, aw, 74, L["a7t"], L["a7"], fill=GRY))

    # ---------------- panel B : the network ----------------------------
    bx, bw = BXX + 20, BW - 40
    bcx = bx + bw / 2
    B = [(86, 52, L["b1t"], L["b1"], False, None),
         (158, 76, L["b2t"], L["b2"], True, 1),
         (254, 52, L["b3t"], L["b3"], False, None),
         (326, 88, L["b4t"], L["b4"], True, 1),
         (434, 58, L["b5t"], L["b5"], False, None),
         (512, 50, L["b6t"], L["b6"], False, None),
         (582, 46, L["b7t"], L["b7"], False, None),
         (648, 46, L["b8t"], L["b8"], True, None)]
    for i, (y, h, t, ln, acc, mf) in enumerate(B):
        s.append(box(bx, y, bw, h, t, ln, accent=acc, mono_from=mf))
        if i:
            py, ph = B[i - 1][0], B[i - 1][1]
            s.append(vlink(bcx, py + ph, y))

    # tokens/lengths from A5 into B1
    s.append(path_arrow(f"M {ax+aw} 435 H {BXX-14} V 112 H {bx-4}"))
    # scenario graph from A6 into the two accent blocks
    s.append(path_arrow(f"M {ax+aw} 537 H {BXX-6} V 196 H {bx-4}", dashed=True, color=ACC))
    s.append(path_arrow(f"M {BXX-6} 370 H {bx-4}", dashed=True, color=ACC))

    # ---------------- panel C : objective + two uses -------------------
    cx_, cw = CX + 16, CW - 32
    s.append(box(cx_, 86, cw, 92, L["c1t"], L["c1"], mono_from=1))
    s.append(box(cx_, 250, cw, 182, L["c2t"], L["c2"], mono_from=3))
    s.append(box(cx_, 500, cw, 154, L["c3t"], L["c3"], mono_from=4))
    bus = CX - 16
    s.append(f'<line x1="{bx+bw}" y1="671" x2="{bus}" y2="671" stroke="{MUT}" stroke-width="1.3"/>')
    s.append(f'<line x1="{bus}" y1="132" x2="{bus}" y2="671" stroke="{MUT}" stroke-width="1.3"/>')
    for y in (132, 341, 577):
        s.append(f'<line x1="{bus}" y1="{y}" x2="{cx_-7}" y2="{y}" stroke="{MUT}" '
                 f'stroke-width="1.3" marker-end="url(#ar)"/>')
    s.append(f'<text x="{(bx+bw+bus)/2}" y="664" text-anchor="middle" class="ed">{esc(L["e_logit"])}</text>')

    # legend + footnote + provenance
    s.append(f'<line x1="26" y1="732" x2="58" y2="732" stroke="{ACC}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    s.append(f'<text x="66" y="736" class="eda">{esc(L["leg"])}</text>')
    s.append(f'<text x="{W-24}" y="736" text-anchor="end" class="src">{esc(L["src"])}</text>')
    s.append(f'<text x="24" y="762" class="cap">{esc(L["foot"])}</text>')
    s.append(f'<text x="24" y="778" class="cap">{esc(L["foot2"])}</text>')
    s.append("</svg>")
    return "\n".join(s)


EN = dict(
    title="Discriminator pipeline for importance-weight guidance in block diffusion",
    src="models_seq/bd_disc.py · tools/train/train_bd_disc.py · bd_models.py::_denoise_block_mask_guided · tools/eval/disc_gen_eval.py",
    pA="Example construction", pB="BDDiscriminator  D(x | A_scn)", pC="Objective and the two uses",
    a1t="Positives — exceptional",
    a1=["except_e shortest paths, 1% slice", "(first 1,000 rows held out for eval)"],
    a2t="Negatives — reference p_ref",
    a2=["model: unconditional BD samples", "or data: real normal paths"],
    a3t="make_partial", a3=["random prefix in canvas form", "[dst, v0, …, v_j],  j ~ U{2,…,L}"],
    a4t="pad_batch", a4=["dedicated PAD = V, max_len 128", "→ (tokens, lengths)"],
    a5t="batch", a5=["64 positives + 64 negatives", "labels 1−ε / ε,  ε = 0.05"],
    a6t="scenario e drawn per step",
    a6=["A_scn  (V × V) adjacency", "deg_ratio = deg_scn / deg_norm"],
    acap="one scenario per step: e0 = {0}, e99 = {1,…,99}",
    a7t="Pool sizes (Porto, V = 1,390)",
    a7=["e0: 19,308 positives (1% of 1.93 M)", "e99: 1,911,492 positives",
        "model negatives: 20,000 per block"],
    b1t="Vertex embedding table",
    b1=["(V+1) × 100, node2vec init", "dedicated PAD row (padding_idx)"],
    b2t="GraphSAGE × 2  over A_scn",
    b2=["h ← ReLU( W [ h ‖ Â h ] ),  Â = row-norm(A+I)",
        "E_scn = E + h   (residual; PAD row untouched)",
        "⇒ scenario-conditioned vertex embeddings"],
    b3t="Gather + project",
    b3=["E_scn[tokens] → Linear 100 → 128", "+ learned positional embedding"],
    b4t="Transition-feature channel  → Linear 3 → 128",
    b4=["per position i:",
        "edge_exists = A_scn[ v_{i−1}, v_i ]   ∈ {0,1}",
        "is_transition = 1[ i ≥ 2 and not PAD ]",
        "deg_ratio[ v_i ]                (added to x)"],
    b5t="Transformer encoder × 2",
    b5=["pre-LN, 4 heads, dim 128", "attention masked on PAD keys"],
    b6t="LayerNorm + masked mean pooling",
    b6=["PAD positions excluded → batch-invariant"],
    b7t="MLP head  128 → 128 → 1   ⇒  z", b7=[],
    b8t="Bounded logit   ℓ = λ · tanh( z / λ ),   λ = 4", b8=["|ℓ| ≤ 4  ⇒  weight capped at e⁴"],
    e_logit="ℓ", leg="scenario input: A_scn, deg_ratio",
    c1t="Training",
    c1=["BCE-with-logits on smoothed labels",
        "Adam · grad-clip 1.0 · 4,000 steps · bs 128",
        "50/50 positives / negatives per step"],
    c2t="Use 1 — SNIS guidance at sampling time",
    c2=["at every first-hitting reveal:",
        "draw n_is = 100 candidate blocks from the",
        "denoiser marginal (adjacency-masked, Lemma 3)",
        "score [ committed prefix ‖ candidate ]",
        "truncate: after dst, before END/PAD/MASK",
        "w ∝ exp( γ ℓ ) = ( D / (1−D) )^γ",
        "        ≈ ( q_exc / p_ref )^γ",
        "self-normalise → x̄₀ → reveal one token",
        "cost: L · n_is discriminator calls per path"],
    c3t="Use 2 — validity probe (DISC_VALIDITY)",
    c3=["held-out positives:",
        "  except_0 reserve  /  except_1 outside the slice",
        "negatives: fresh seed-99 pool or seed-555 normal",
        "identical make_partial truncation as training",
        "accuracy@0 and AUC on the balanced set,",
        "plus e0-vs-e99 logit correlation"],
    foot="The scenario enters at exactly two places (amber): the GraphSAGE embedding and the edge-existence / degree-ratio channel. Both read the adjacency as an",
    foot2="input rather than memorising it, which is why a discriminator trained on except 1–99 transfers zero-shot to the unseen except_0 scenario.",
)

KO = dict(
    title="Block diffusion 중요도가중 guidance를 위한 판별자 파이프라인",
    src=EN["src"],
    pA="학습 예제 구성", pB="BDDiscriminator  D(x | A_scn)", pC="목적함수와 두 가지 사용처",
    a1t="Positive — exceptional",
    a1=["except_e 최단경로, 1% 슬라이스", "(앞 1,000행은 평가용으로 예약)"],
    a2t="Negative — 기준분포 p_ref",
    a2=["model: 무조건부 BD 생성 경로", "또는 data: 실제 normal 경로"],
    a3t="make_partial", a3=["canvas 형태 랜덤 prefix", "[dst, v0, …, v_j],  j ~ U{2,…,L}"],
    a4t="pad_batch", a4=["전용 PAD = V, max_len 128", "→ (tokens, lengths)"],
    a5t="배치", a5=["positive 64 + negative 64", "레이블 1−ε / ε,  ε = 0.05"],
    a6t="스텝마다 시나리오 e 추출",
    a6=["A_scn  (V × V) 인접행렬", "deg_ratio = deg_scn / deg_norm"],
    acap="스텝당 시나리오 하나: e0 = {0}, e99 = {1,…,99}",
    a7t="풀 크기 (Porto, V = 1,390)",
    a7=["e0: positive 19,308개 (1.93M의 1%)", "e99: positive 1,911,492개",
        "model negative: 블록당 20,000개"],
    b1t="정점 임베딩 테이블",
    b1=["(V+1) × 100, node2vec 초기화", "전용 PAD 행 (padding_idx)"],
    b2t="GraphSAGE × 2  (A_scn 위에서)",
    b2=["h ← ReLU( W [ h ‖ Â h ] ),  Â = row-norm(A+I)",
        "E_scn = E + h   (잔차; PAD 행은 그대로)",
        "⇒ 시나리오 조건부 정점 임베딩"],
    b3t="Gather + 사영",
    b3=["E_scn[tokens] → Linear 100 → 128", "+ 학습형 위치 임베딩"],
    b4t="전이 특징 채널  → Linear 3 → 128",
    b4=["위치 i마다:",
        "edge_exists = A_scn[ v_{i−1}, v_i ]   ∈ {0,1}",
        "is_transition = 1[ i ≥ 2 이고 PAD 아님 ]",
        "deg_ratio[ v_i ]                (x에 가산)"],
    b5t="Transformer 인코더 × 2",
    b5=["pre-LN, 4 heads, dim 128", "PAD 키에 대한 어텐션 마스킹"],
    b6t="LayerNorm + 마스킹 평균 풀링",
    b6=["PAD 위치 제외 → 배치 구성에 불변"],
    b7t="MLP head  128 → 128 → 1   ⇒  z", b7=[],
    b8t="유계 로짓   ℓ = λ · tanh( z / λ ),   λ = 4", b8=["|ℓ| ≤ 4  ⇒  가중치는 e⁴로 상한"],
    e_logit="ℓ", leg="시나리오 입력: A_scn, deg_ratio",
    c1t="학습",
    c1=["스무딩된 레이블에 대한 BCE-with-logits",
        "Adam · grad-clip 1.0 · 4,000 steps · bs 128",
        "스텝마다 positive / negative 50:50"],
    c2t="사용 1 — 샘플링 시 SNIS guidance",
    c2=["first-hitting reveal 마다:",
        "denoiser 주변분포에서 후보 블록 n_is = 100개 추출",
        "(인접성 마스킹, Lemma 3)",
        "[ 확정 prefix ‖ 후보 ] 를 채점",
        "절단: dst 뒤 / END·PAD·MASK 앞에서 자름",
        "w ∝ exp( γ ℓ ) = ( D / (1−D) )^γ",
        "        ≈ ( q_exc / p_ref )^γ",
        "자기정규화 → x̄₀ → 토큰 하나 공개",
        "비용: 경로당 판별자 호출 L · n_is 회"],
    c3t="사용 2 — 타당성 검증 (DISC_VALIDITY)",
    c3=["평가용 positive:",
        "  except_0 예약분  /  except_1 슬라이스 밖",
        "negative: seed-99 신규 풀 또는 seed-555 normal",
        "학습과 동일한 make_partial 절단 적용",
        "균형 집합에서 accuracy@0 · AUC,",
        "그리고 e0–e99 로짓 상관"],
    foot="시나리오는 정확히 두 곳(주황)으로만 들어간다: GraphSAGE 임베딩과 edge-existence / degree-ratio 채널. 둘 다 인접행렬을 암기하는 대신 입력으로",
    foot2="읽기 때문에, except 1–99로 학습한 판별자가 처음 보는 except_0 시나리오에 zero-shot으로 전이된다.",
)

CSS = """<style>
 @page { size: 318mm 209mm; margin: 0 }
 html,body { margin:0; padding:0; background:#fff }
 svg { display:block }
 text { font-family:"Helvetica Neue",Helvetica,Arial,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;
        fill:%s; -webkit-font-smoothing:antialiased }
 .ti { font-size:19px; font-weight:700; letter-spacing:-0.2px }
 .src{ font-size:10px; fill:#9aa5b1; font-family:ui-monospace,SFMono-Regular,Menlo,monospace }
 .pt { font-size:14.5px; font-weight:700; fill:#3e4c59 }
 .bt { font-size:12.5px; font-weight:700 }
 .bl { font-size:11px; fill:#3e4c59 }
 .mn { font-size:10.5px; fill:#3e4c59; font-family:ui-monospace,SFMono-Regular,Menlo,monospace }
 .ed { font-size:10px; fill:#7b8794 }
 .eda{ font-size:10px; fill:#b45309; font-weight:600 }
 .cap{ font-size:10.5px; fill:#7b8794 }
</style>""" % INK

if __name__ == "__main__":
    import os
    root = "/Users/jiwooshin/Desktop/In_Progress/Block-Diffusion-OD-Planning"
    for lang, L in [("en", EN), ("ko", KO)]:
        p = os.path.join(root, "report_src", f"disc_pipeline_{lang}.html")
        with open(p, "w") as f:
            f.write(f"<title>{esc(L['title'])}</title>\n{CSS}\n{build(L)}\n")
        print("written", p)
