"""
paperC_fig_window.py — Figure 1 (the operating window): reads only the committed npz in processed_data/, runs no model.
One figure, 16 cm wide, Times New Roman, style from neuro_paper.mplstyle (in this directory) and plotstyle.py.
 A  extent of the planar (axis) front vs lattice coupling d_lat at refinement k = 1, 2, 4, 8 (paperC_continuum)
 B  direct continuum solve: propagated or not vs grid, FD4/FD6, plotted against d/dx^2 (paperC_pde)
 C  upper edge vs time step under the original and the settling release rule (paperC_continuum, _edges, _dt, _pde,
    _design)
 D  front speed approaching each edge of the window (paperC_edges, S1 strong edge; weak edge for reference); upper
    edge also under the settling rule at DT/32 (paperC_template, round 8, SP)
 E  weak-edge threshold by lattice direction, s d_low/2 relative to the axis (paperC_edges, W1)
 F  tolerance (descriptive, no fit): upper edge per realisation under criteria (i) and (ii)
    (paperC_tolerance2, round 7) and lower edge (round 6, paperC_tolerance thr, as stored in paperC_tolerance2 d_low6),
    medians as bars. Where round 7 resolved no upper edge (delta 0.4, f 0.5: no grid point 1-20 met criterion (i) or
    (ii) in 5/5) the panel says "upper edge not resolved" and shows the round-6 weak-edge-grid couplings >= 1 at which
    the front PROPAGATED (paperC_tolerance grids 'lo', dt = DT, 1200 s): these conditions are NOT drawn as closed
    (round 7's verdicts are unchanged). The scaling median x (1 + delta) is NOT drawn: S1 failed.
Release rule by marker (original: filled circle; settling: open square); reference lines thin black dashed with an
inline label (grey = lower edge only). Exports pdf + svg + png and Figure_window_stats.csv (every plotted value,
long format).
Run: python -s paperC_fig_window.py  -> Figure_window.pdf/.svg/.png, Figure_window_stats.csv
"""
import csv, json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.model import DT
from plotstyle import palette, sequential, panel_label

H = pathlib.Path(__file__).resolve().parent / "processed_data"
CM = 1 / 2.54
STYLE = pathlib.Path(__file__).resolve().parent / "neuro_paper.mplstyle"   # vendored, next to this script
plt.style.use(STYLE)                          # house style: 8/7/6 pt, outward 2.5 pt ticks, lw 1.0, 600 dpi
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})   # figures in Times New Roman
MS = 2; S = MS ** 2                           # line markers ms = 2; scatter area in pt^2 to match
C_I, C_II, C_LO = (palette(k) for k in ("criterion_i", "criterion_ii", "lower_edge"))
# One colour = one concept; colour only where markers cannot carry it. Release rule by
# marker (original: filled circle; settling: open square; both solid black lines). Front propagated: filled black dot
# (B and F alike); no front: black x. Reference lines thin black dashed with an inline label.
ORIG = dict(marker="o", ls="-", color="k", ms=MS)
SETT = dict(marker="s", ls="-", color="k", mfc="white", ms=MS + 0.5, mew=0.6)
PROP = dict(marker="o", s=S, color="k", lw=0)
NOFR = dict(marker="x", s=S + 2, color="k", lw=0.5)
REF = dict(color="k", lw=0.5, ls="--")
REFTXT = dict(fontsize=5.5, va="bottom")
LEG = dict(frameon=False, handletextpad=0.4, borderaxespad=0.2)
ROWS = []


def rec(panel, series, xs, ys, **cond):
    """Record plotted values for Figure_window_stats.csv (long format)."""
    for i, (x, y) in enumerate(zip(np.atleast_1d(xs), np.atleast_1d(ys))):
        ROWS.append(dict(panel=panel, series=series, x=x, y=y,
                         **{k: (v[i] if isinstance(v, (list, range)) else v) for k, v in cond.items()}))
P = lambda f: json.loads(str(np.load(H / f"{f}.npz", allow_pickle=True)["__params__"]))
c = np.load(H / "paperC_continuum.npz", allow_pickle=True)
e = np.load(H / "paperC_edges.npz", allow_pickle=True)
p = np.load(H / "paperC_pde.npz", allow_pickle=True)
fig, ax = plt.subplots(2, 3, figsize=(16 * CM, 10.5 * CM))
ax = ax.ravel()

a = ax[0]
for i, (k, col) in enumerate(zip(c["ks"], sequential(len(c["ks"])))):
    a.plot(c["g_lat"], c["ext_d"][i], "o-", ms=MS, color=col, label=f"h = 1/{k}")
    rec("A", f"h = 1/{k}", c["g_lat"], c["ext_d"][i], x_name="d/h^2", y_name="extent (cells)", rule="original")
a.set_xscale("log"); a.set_xlabel("lattice coupling $d/h^2$"); a.set_ylabel("extent (cells)")
a.axhline(20, **REF); a.text(2.2e3, 22, "propagation criterion", ha="right", **REFTXT)
a.legend(loc="upper right", **LEG)

a = ax[1]
keys = p["pde_keys"]; prop = p["pde_prop"]
for st in ("FD4", "FD6"):
    for j, kk in enumerate(keys):
        if kk[0] != st or kk[4] != "1":
            continue
        d, dx = float(kk[1]), float(kk[2])
        x = d / dx ** 2 * (1.13 if st == "FD6" else 0.885)   # FD4 left, FD6 right of each pair
        a.scatter(x, d, **(PROP if prop[j] else NOFR))
        rec("B", "front propagated" if prop[j] else "no front", d / dx ** 2, d, x_name="d/dx^2", y_name="d",
            stencil=st, dx=dx, seed=kk[3], dt_div=kk[4])
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("$d/\\Delta x^2$"); a.set_ylabel("coupling $d$")
a.scatter([], [], label="front propagated", **PROP)
a.scatter([], [], label="no front", **NOFR)
a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, columnspacing=1.0, **LEG)

a = ax[2]
cth, eth, tth, pth, gth = P("paperC_continuum")["thr_d"], P("paperC_edges")["thr"], P("paperC_dt")["thr"], \
    P("paperC_pde")["p3_thr"], P("paperC_design")["thr"]
orig = [(1, cth["(1, 1.0)"]), (2, eth["E_w5_dt2"]), (4, eth["E_w5_dt4"]), (8, eth["E_w5_dt8"])] + \
       [(f, tth[f"ax_dt{f}"]) for f in (16, 32, 64, 128)]
sett = sorted([(int(k), v) for k, v in pth.items()] + [(128, gth["R1_dt128"])])   # DT/32: P3 and R1 both measured
a.plot([DT / f for f, _ in orig], [v for _, v in orig], label="original rule", **ORIG)
a.plot([DT / f for f, _ in sett], [v for _, v in sett], label="settling rule", **SETT)
a.plot([DT / 32], [gth["R1_dt32"]], **{**SETT, "ls": "none"})
rec("C", "original rule", [DT / f for f, _ in orig], [v for _, v in orig], x_name="time step (s)", y_name="d*")
rec("C", "settling rule", [DT / f for f, _ in sett] + [DT / 32], [v for _, v in sett] + [gth["R1_dt32"]],
    x_name="time step (s)", y_name="d*")
a.set_xscale("log"); a.set_xlabel("time step (s)"); a.set_ylabel("upper edge $d^*$")
a.legend(loc="upper left", **LEG)

a = ax[3]
eps = e["eps"]
a.plot(eps, e["v_strong"], label="upper edge, original rule", **ORIG)
tp = np.load(H / "paperC_template.npz", allow_pickle=True)
a.plot(tp["eps"], tp["v_settle"], label="upper edge, settling rule", **SETT)
a.plot(eps, e["v_weak"], "^-", color=C_LO, mfc="white", ms=MS + 0.5, mew=0.6, label="lower edge, original rule")
for ser, xs, ys in (("upper edge, original rule", eps, e["v_strong"]), ("upper edge, settling rule", tp["eps"],
                    tp["v_settle"]), ("lower edge, original rule", eps, e["v_weak"])):
    rec("D", ser, xs, ys, x_name="epsilon", y_name="front speed (rows/s)")
a.set_xscale("log"); a.set_xlabel("distance to edge $\\epsilon$"); a.set_ylabel("front speed (rows/s)")
a.set_ylim(0, None); a.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), **LEG)

a = ax[4]
dev = P("paperC_edges")["devW"]
names = {"1,1": "(1,1)", "2,3": "(3,2)", "1,2": "(2,1)", "1,1,2": "tri.", "1,3": "(3,1)"}
ks = ["1,1", "2,3", "1,2", "1,1,2", "1,3"]
a.bar(range(len(ks)), [1 + dev[k] for k in ks], color=C_LO, width=0.7)
a.axhline(1, **REF); a.text(-0.85, 1.02, "axis", ha="left", **REFTXT)
rec("E", "lower edge by direction", [names[k] for k in ks], [1 + dev[k] for k in ks], x_name="front direction",
    y_name="m d_low/2 / axis d_low")
a.set_xticks(range(len(ks))); a.set_xticklabels([names[k] for k in ks]); a.set_xlim(-0.9, len(ks) - 0.5)
a.set_xlabel("front direction"); a.set_ylabel("$m\\,d_{low}/2$ / axis $d_{low}$")

a = ax[5]
t2 = P("paperC_tolerance2"); SA = t2["sig_A"]; g6 = P("paperC_tolerance")["grids"]
from paperC_tolerance import G_LO
edge = t2["edge"]
LV = [("uniform", [(0.0, 1, 0.0)])] + \
     [(f"$\\delta$ {v}", [(v, r, 0.0) for r in range(1, 6)]) for v in (0.1, 0.2, 0.4)] + \
     [(f"$f$ {v}", [(0.0, r, v * SA) for r in range(1, 6)]) for v in (0.25, 0.5)]
for x, (lab, cs) in enumerate(LV):
    lo = [t2["d_low6"][str(c)] for c in cs]
    ser = [(lo, x, "^", C_LO, "lower edge")] + \
          [([edge[str((c, cr))] for c in cs], x + dx, mk, col, f"upper edge ({cr})")
           for cr, dx, mk, col in (("i", -0.2, "o", C_I), ("ii", 0.2, "D", C_II))]
    for ys, xx, mk, col, name in ser:
        ys = np.array(ys, float)
        if not np.isfinite(ys).any():
            continue
        a.scatter(np.full(ys.size, xx), ys, marker=mk, s=S + 2, facecolors="none", edgecolors=col, lw=0.6,
                  label=name if x == 1 else None)
        rec("F", name, [lab] * ys.size, ys, x_name="condition", y_name="coupling d", realisation=range(1, ys.size + 1),
            median=np.median(ys))
        a.plot([xx - 0.12, xx + 0.12], [np.median(ys)] * 2, color=col, lw=1.0)
    if all(not np.isfinite(edge[str((c, cr))]) for c in cs for cr in ("i", "ii")):
        for j, c in enumerate(cs):   # round-6 weak-edge runs that propagated at d >= 1, one column per realisation
            ds = [d for d, b in zip(G_LO, g6[str((*c, "lo"))]) if b and d >= 1]
            a.scatter(np.full(len(ds), x - 0.2 + 0.1 * j), ds,
                      label="front propagated" if (x == 3 and j == 0) else None, **PROP)
            rec("F", "front propagated (round 6)", [lab] * len(ds), ds, x_name="condition", y_name="coupling d",
                realisation=j + 1)
        a.text(x, 2.6, "upper edge\nnot resolved", rotation=90, ha="center", va="bottom", fontsize=5.5, linespacing=1.0)
a.set_yscale("log"); a.set_ylim(0.4, 25); a.minorticks_off()
a.set_yticks([0.5, 1, 2, 5, 10, 20]); a.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
a.set_xticks(range(len(LV))); a.set_xticklabels([l for l, _ in LV], rotation=40, ha="right")
a.set_xlim(-0.5, len(LV) - 0.5)
a.set_ylabel("coupling $d$")
a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, columnspacing=0.8, **LEG)

for i, x in enumerate(ax):
    panel_label(x, "ABCDEF"[i], dx=-0.2, dy=1.0, fontsize=9)
fig.tight_layout(h_pad=1.2, w_pad=1.0)
OUT = pathlib.Path(__file__).resolve().parent / "Figure_window"
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{OUT}.{ext}")               # dpi and bbox from the style file
cols = list(dict.fromkeys(k for r in ROWS for k in r))
with open(f"{OUT}_stats.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(ROWS)
