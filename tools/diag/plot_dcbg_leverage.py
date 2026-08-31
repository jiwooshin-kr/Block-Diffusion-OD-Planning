"""
Visualize tools/diag/diag_dcbg_leverage.py output: how much leverage the D-CBG
classifier tilt actually has on the reveal distribution, as a function of
gamma, and WHY it is near-zero at gamma=1.

  python tools/diag/plot_dcbg_leverage.py -npz sets_res/dcbg_leverage_blk4_g1.0.npz
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-npz", type=str, default="sets_res/dcbg_leverage_blk4_g1.0.npz")
ap.add_argument("-out", type=str, default="scratchpad_dcbg_leverage.png")
args = ap.parse_args()

d = np.load(args.npz)
g = d["gammas"]
tv, flip = d["tv"], d["flip"]                     # (n, 2 scales, G)
n = tv.shape[0]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
fig.suptitle("D-CBG guidance leverage per reveal step "
             f"(mask kernel, blk4, l2r, exact enumeration, {n} reveal steps)",
             fontsize=12)

# (a) TV distance vs gamma
ax = axes[0, 0]
for si, (nm, c) in enumerate([("clf @ t*100 (as evaluated)", "tab:red"),
                              ("clf @ t*1 (training scale)", "tab:blue")]):
    m = tv[:, si, :].mean(0)
    q1, q3 = np.percentile(tv[:, si, :], [25, 75], axis=0)
    ax.plot(g, m, "o-", color=c, label=f"mean TV, {nm}")
    ax.fill_between(g, q1, q3, color=c, alpha=0.15)
ax.axvline(1.0, color="k", ls="--", lw=1)
ax.text(1.08, 0.5, "gamma used in eval (=1)", rotation=90, va="center", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("gamma"); ax.set_ylabel("TV(p_unguided, p_guided)")
ax.set_title("(a) How far the reveal distribution moves")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (b) argmax flip rate vs gamma
ax = axes[0, 1]
for si, (nm, c) in enumerate([("t*100", "tab:red"), ("t*1", "tab:blue")]):
    ax.plot(g, 100 * flip[:, si, :].mean(0), "o-", color=c, label=f"clf @ {nm}")
ax.axvline(1.0, color="k", ls="--", lw=1)
ax.set_xscale("log"); ax.set_xlabel("gamma"); ax.set_ylabel("argmax flip rate (%)")
ax.set_title("(b) Steps where guidance changes the top-1 token")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (c) why: model log-p spread vs classifier log-p spread over top-10 candidates
ax = axes[1, 0]
bins = np.linspace(-3, 2.2, 54)
ax.hist(np.log10(np.clip(d["logp_rng_top10"], 1e-3, None)), bins=bins, alpha=0.55,
        label=f"model log p_theta range (top-10)\nmean {d['logp_rng_top10'].mean():.1f} nats",
        color="tab:gray")
ax.hist(np.log10(np.clip(d["clf_rng_top10"][:, 0], 1e-3, None)), bins=bins, alpha=0.6,
        label=f"classifier log p_phi range (top-10)\nmean {d['clf_rng_top10'][:,0].mean():.2f} nats",
        color="tab:red")
ax.set_xlabel("log10(range over top-10 candidates)  [nats]")
ax.set_ylabel("reveal steps")
ax.set_title("(c) Signal the tilt must overcome (gamma multiplies red only)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (d) one concrete reveal step: top-10 candidate probabilities
ax = axes[1, 1]
if "ex_logp" in d:
    lp, cl = d["ex_logp"], d["ex_clf100"]
    x = np.arange(len(lp))
    w = 0.27
    for k, (gg, c) in enumerate([(0.0, "tab:gray"), (1.0, "tab:red"), (32.0, "tab:orange")]):
        z = lp + gg * cl
        p = np.exp(z - z.max()); p = p / p.sum()
        ax.bar(x + (k - 1) * w, p, w, color=c, label=f"gamma={gg:g}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in d["ex_idx"]], rotation=45, fontsize=7)
    ax.set_xlabel("top-10 candidate tokens (one reveal step)")
    ax.set_ylabel("reveal probability")
    ax.set_title(f"(d) Example step (t={float(d['ex_t']):.2f}): gamma=1 is invisible")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(args.out, dpi=150)
print(f"saved {args.out}")

# headline numbers
print(f"median p0 top-1 prob        : {np.median(d['p0_top1']):.4f}")
print(f"mean TV at gamma=1 (t*100)  : {tv[:, 0, list(g).index(1.0)].mean():.4f}")
print(f"flip%% at gamma=1 (t*100)    : {100*flip[:, 0, list(g).index(1.0)].mean():.2f}%")
