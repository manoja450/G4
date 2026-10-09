#!/usr/bin/env python3
"""
Combined loss surface for the 2-module Michel parameter scan.

DEFINITION
----------
Each module has its own loss

    L_M(R, W) = sqrt( 4 * (mu_sim - mu_data)^2 + (sigma_sim - sigma_data)^2 )

The two module losses are combined in quadrature:

    L_comb(R, W_M1, W_M2) = sqrt( L_M1(R, W_M1)^2 + L_M2(R, W_M2)^2 )

Since R is the only shared parameter, the combined minimum is separable:

    min_{W_M1,W_M2} L_comb(R, W_M1, W_M2)
        = sqrt( min_{W_M1} L_M1(R, W_M1)^2
              + min_{W_M2} L_M2(R, W_M2)^2 )

(R = reflectivity, W = water attenuation length coefficient.)

Outputs (into 3DScanPlots/):

  Non-combined (each module's own loss L_M, its own best fit):
    loss_heatmap_Module1.png
    loss_heatmap_Module2.png

  Combined (one shared R, a separate W for each module):
    loss_comb_vs_R_Module1_Module2.png   L_comb(R) with the best W_M1 and
                                         W_M2 at each R
    loss_comb_heatmap_Module1.png        L_comb on the (R, W_M1) plane, with
                                         W_M2 at its best value for each R
    loss_comb_heatmap_Module2.png        L_comb on the (R, W_M2) plane, with
                                         W_M1 at its best value for each R
    loss_comb_heatmap_W1_W2_Module1_Module2.png
                                         L_comb on the (W_M1, W_M2) plane at
                                         the best-fit shared R

  1D curves of the combined loss along each parameter:
    loss_comb_3profiles.png              three panels (R, W_M1, W_M2);
                                         W panels are slices at the best R

  Additional diagnostics:
    loss_modules_vs_R.png                L_M1, L_M2 and L_comb vs R, each
                                         module at its own best W for that R
    best_W_vs_R.png                      best W_M1 and best W_M2 vs R
    residuals_vs_R.png                   mu_sim - mu_data and
                                         sigma_sim - sigma_data vs R,
                                         per module, at the best W for each R
    loss_comb_vs_R_errorbars.png         L_comb vs R with error bars from the
                                         simulation statistics

Reads point_*.json files written by michel3Dscan.py.

Usage:
    python plot_combined_loss.py
    python plot_combined_loss.py --results results_3Dscan --outdir 3DScanPlots
    python plot_combined_loss.py --tag sep2M_41x34_R0.80-1.00_A0.20-1.00
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Plot text (kept in one place so all three figures stay consistent
# with the single-module heatmap)
# =====================================================================

TITLE_HEATMAP = "Michel Parameter Scan: Loss Surface"
TITLE_VS_R    = "Michel Parameter Scan: Loss vs Reflectivity"

XLABEL_R = "Reflectivity"
YLABEL_W = "Water Attenuation Length Coefficient"
BEST_LABEL = "Best-fit"

# combined loss: one shared R, a separate W for each module
CBAR_LABEL = (r"$L(R, W_1, W_2) = "
              r"\sqrt{L_1(R, W_1)^2 + L_2(R, W_2)^2}$")

# single-module (non-combined) loss, exactly as on the reference heatmap
CBAR_LABEL_SINGLE = (r"$L=\sqrt{4(\mu_{sim}-\mu_{data})^2"
                     r"+(\sigma_{sim}-\sigma_{data})^2}$")

ANNOT_BOX = dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="black", alpha=0.88)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_3Dscan",
                   help="folder containing point_*.json")
    p.add_argument("--outdir", default="3DScanPlots",
                   help="folder for output PNGs (created if missing)")
    p.add_argument("--tag", default=None,
                   help="only use JSONs whose 'scan' field equals this tag")
    p.add_argument("--cmap-combined", default="magma",
                   help="colormap for the combined heatmaps, so they "
                        "look different from the non-combined ones")
    return p.parse_args()


# =====================================================================
# I/O
# =====================================================================

def load_points(folder, tag):
    rows = []
    for path in glob.glob(os.path.join(folder, "point_*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("status") != "ok":
            continue
        if tag is not None and rec.get("scan") != tag:
            continue
        rows.append(rec)
    if not rows:
        raise SystemExit(f"No successful points found in {folder}")
    return rows


def build_grids(rows):
    nR = max(r["i"] for r in rows) + 1
    nA = max(r["j"] for r in rows) + 1
    Rg = np.full(nR, np.nan)
    Ag = np.full(nA, np.nan)
    L1 = np.full((nR, nA), np.nan)
    L2 = np.full((nR, nA), np.nan)
    for r in rows:
        i, j = r["i"], r["j"]
        Rg[i] = r["reflectivity"]
        Ag[j] = r["h2orefl"]
        L1[i, j] = r["loss_M1"]
        L2[i, j] = r["loss_M2"]
    return Rg, Ag, L1, L2


def build_stat_grids(rows, shape):
    """
    Per-module grids of the quantities behind the loss, for the residual
    and error-bar plots.

    For module m (1 or 2) the returned dict holds, on the (R, W) grid:

        dmu{m}   mu_sim - mu_data
        dsig{m}  sigma_sim - sigma_data
        err{m}   statistical error on L_m from the simulation sample size

    The error uses  delta(mu) = sigma_sim / sqrt(n)  and
    delta(sigma) = sigma_sim / sqrt(2n), propagated through
    L = sqrt(4 dmu^2 + dsig^2).  It covers the simulation statistics
    only: the data histogram's own uncertainty is not in the JSON files.
    Points whose JSON lacks these fields stay NaN.
    """
    out = {}
    for m in (1, 2):
        dmu = np.full(shape, np.nan)
        dsig = np.full(shape, np.nan)
        err = np.full(shape, np.nan)
        for r in rows:
            mu_s = r.get(f"mu_sim_M{m}")
            sg_s = r.get(f"sigma_sim_M{m}")
            mu_d = r.get(f"mu_data_M{m}")
            sg_d = r.get(f"sigma_data_M{m}")
            n = r.get(f"n_events_M{m}")
            if None in (mu_s, sg_s, mu_d, sg_d):
                continue
            i, j = r["i"], r["j"]
            a = mu_s - mu_d
            b = sg_s - sg_d
            dmu[i, j] = a
            dsig[i, j] = b
            if n:
                e_mu = sg_s / np.sqrt(n)
                e_sg = sg_s / np.sqrt(2.0 * n)
                L = np.sqrt(4.0 * a * a + b * b)
                if L > 0:
                    err[i, j] = np.sqrt((4.0 * a * e_mu) ** 2
                                        + (b * e_sg) ** 2) / L
        out[f"dmu{m}"] = dmu
        out[f"dsig{m}"] = dsig
        out[f"err{m}"] = err
    return out


# =====================================================================
# Combined loss
# =====================================================================

def combined_loss_per_module(L1, L2):
    """
    Combined loss shown on each module's own (R, W) plane.

    The other module's W is set to its best value at every R:

        Lc1[i, j] = sqrt( L_M1(R_i, W_j)^2 + min_W L_M2(R_i, W)^2 )
        Lc2[i, j] = sqrt( min_W L_M1(R_i, W)^2 + L_M2(R_i, W_j)^2 )

    Both surfaces have the same minimum value at the same R, which is
    the global minimum of L_comb(R, W_M1, W_M2).
    """
    with np.errstate(all="ignore"):
        best1 = np.nanmin(L1, axis=1)   # best M1 loss at each R
        best2 = np.nanmin(L2, axis=1)   # best M2 loss at each R
    Lc1 = np.sqrt(L1 ** 2 + best2[:, None] ** 2)
    Lc2 = np.sqrt(best1[:, None] ** 2 + L2 ** 2)
    return Lc1, Lc2


def separable_minimum(L1, L2):
    """
    For every R row, find the best W for each module independently,
    then compute the combined loss at that row.

    Returns
    -------
    curve : 1D array, length nR
        L_comb at the best (W_M1, W_M2) for each R row.
    j1_best, j2_best : 1D int arrays
        Index of the best attenuation for each module at each R row.
    """
    nR = L1.shape[0]
    curve = np.full(nR, np.nan)
    j1_best = np.full(nR, -1, dtype=int)
    j2_best = np.full(nR, -1, dtype=int)
    for i in range(nR):
        r1, r2 = L1[i], L2[i]
        if np.all(np.isnan(r1)) or np.all(np.isnan(r2)):
            continue
        j1 = int(np.nanargmin(r1))
        j2 = int(np.nanargmin(r2))
        j1_best[i] = j1
        j2_best[i] = j2
        curve[i] = np.sqrt(r1[j1] ** 2 + r2[j2] ** 2)
    return curve, j1_best, j2_best


def along_best_path(G, j_best):
    """G[i, j_best[i]] for every R row (NaN where the row has no best W)."""
    out = np.full(G.shape[0], np.nan)
    for i, j in enumerate(j_best):
        if j >= 0:
            out[i] = G[i, j]
    return out


# =====================================================================
# Plots
# =====================================================================

def plot_heatmap(Rg, Ag, L, out_path, title=None,
                         cmap="cividis", star_face="black", star_edge="white",
                         cbar_label=None):
    """Heatmap of a loss on a 2D (R, W) grid, best-fit = its minimum."""
    if title is None:
        title = TITLE_HEATMAP
    if cbar_label is None:
        cbar_label = CBAR_LABEL
    i_b, j_b = np.unravel_index(np.nanargmin(L), L.shape)
    bR, bA, bL = Rg[i_b], Ag[j_b], L[i_b, j_b]

    vmin = np.nanmin(L)
    vmax = np.nanpercentile(L, 95)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.pcolormesh(Rg, Ag, L.T, shading="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, pad=0.06, extend="max")
    cb.set_label(cbar_label, fontsize=16)
    cb.ax.tick_params(labelsize=14)

    ax.plot(bR, bA, marker="*", ms=28, color=star_face,
            markeredgecolor=star_edge, markeredgewidth=1.8,
            linestyle="none", zorder=5, label=BEST_LABEL)
    # keep the box inside the axes when the best-fit sits near an edge
    right = bR > Rg[0] + 0.8 * (Rg[-1] - Rg[0])
    top = bA > Ag[0] + 0.8 * (Ag[-1] - Ag[0])
    ax.annotate(f"{BEST_LABEL}\nW={bA:.4f}\nR={bR:.4f}\nL={bL:.2f}",
                xy=(bR, bA),
                xytext=(-22 if right else 22, -12 if top else 12),
                textcoords="offset points",
                ha="right" if right else "left",
                va="top" if top else "bottom",
                fontsize=14, zorder=6, bbox=ANNOT_BOX)

    ax.set_xlabel(XLABEL_R, fontsize=22)
    ax.set_ylabel(YLABEL_W, fontsize=22)
    ax.set_title(title, fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(True, color="white", alpha=0.35, linewidth=0.7)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return bR, bA, bL


def plot_combined_vs_R(Rg, Ag, L1, L2, out_path):
    """L_comb(R) at the separable best (W_M1, W_M2) for each R."""
    curve, j1_best, j2_best = separable_minimum(L1, L2)

    i_min = int(np.nanargmin(curve))
    bR = Rg[i_min]
    bA1 = Ag[j1_best[i_min]]
    bA2 = Ag[j2_best[i_min]]
    bL = curve[i_min]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(Rg, curve, "o-", color="C0", lw=2.2, ms=8)
    ax.axvline(bR, color="black", ls="--", alpha=0.6)
    ax.plot([bR], [bL], marker="*", ms=28, color="black",
            markeredgecolor="white", markeredgewidth=1.8, zorder=5,
            linestyle="none", label=BEST_LABEL)

    ax.annotate(f"{BEST_LABEL}\nW_M1={bA1:.4f}\nW_M2={bA2:.4f}\n"
                f"R={bR:.4f}\nL={bL:.2f}",
                xy=(bR, bL), xytext=(22, 25), textcoords="offset points",
                fontsize=14, zorder=6, bbox=ANNOT_BOX)

    ax.set_xlabel(XLABEL_R, fontsize=22)
    ax.set_ylabel(CBAR_LABEL, fontsize=16)
    ax.set_title(TITLE_VS_R, fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return curve, j1_best, j2_best, (bR, bA1, bA2, bL)


def plot_W1_W2_at_best_R(Rg, Ag, L1, L2, out_path, cmap="magma"):
    """
    Combined loss on the (W_M1, W_M2) plane at the best-fit shared R.

    The whole combined fit in one picture: the star gives both W values,
    and R is the shared best-fit reflectivity (shown in the title).
    """
    curve, _, _ = separable_minimum(L1, L2)
    i_best = int(np.nanargmin(curve))
    bR = Rg[i_best]

    # L[j1, j2] = sqrt( L_M1(R_best, W_j1)^2 + L_M2(R_best, W_j2)^2 )
    L = np.sqrt(L1[i_best][:, None] ** 2 + L2[i_best][None, :] ** 2)
    j1, j2 = np.unravel_index(np.nanargmin(L), L.shape)
    bW1, bW2, bL = Ag[j1], Ag[j2], L[j1, j2]

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.pcolormesh(Ag, Ag, L.T, shading="nearest", cmap=cmap,
                       vmin=np.nanmin(L), vmax=np.nanmax(L))
    cb = fig.colorbar(im, ax=ax, pad=0.06)
    cb.set_label(CBAR_LABEL, fontsize=16)
    cb.ax.tick_params(labelsize=14)

    ax.plot(bW1, bW2, marker="*", ms=28, color="white",
            markeredgecolor="black", markeredgewidth=1.8,
            linestyle="none", zorder=5, label=BEST_LABEL)
    right = bW1 > Ag[0] + 0.8 * (Ag[-1] - Ag[0])
    top = bW2 > Ag[0] + 0.8 * (Ag[-1] - Ag[0])
    ax.annotate(f"{BEST_LABEL}\nW_M1={bW1:.4f}\nW_M2={bW2:.4f}\n"
                f"R={bR:.4f}\nL={bL:.2f}",
                xy=(bW1, bW2),
                xytext=(-22 if right else 22, -12 if top else 12),
                textcoords="offset points",
                ha="right" if right else "left",
                va="top" if top else "bottom",
                fontsize=14, zorder=6, bbox=ANNOT_BOX)

    ax.set_xlabel(f"{YLABEL_W} (Module 1)", fontsize=20)
    ax.set_ylabel(f"{YLABEL_W} (Module 2)", fontsize=20)
    ax.set_title(f"{TITLE_HEATMAP} (Reflectivity = {bR:.4f})", fontsize=21)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=14)
    ax.grid(True, color="white", alpha=0.35, linewidth=0.7)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return bR, bW1, bW2, bL


# =====================================================================
# Three combined-loss profiles in the same style
# =====================================================================

def _annotate_above_star(fig, ax, sx, sy, text, x, y):
    """
    Put the ANNOT_BOX just above the star without covering the curve.

    The box is centred over the star and raised only as far as needed to
    clear the curve and the legend.  If the star is close to a side of
    the panel the box slides sideways to stay inside the axes, and if
    there is no free room at all the y-axis is extended upward.

    Call this after tight_layout(), so the axes have their final size.
    """
    ann = ax.annotate(text, xy=(sx, sy), xytext=(0, 20),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=14, zorder=6, bbox=ANNOT_BOX)

    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.asarray(x)[ok], np.asarray(y)[ok]
    pt = fig.dpi / 72.0                       # pixels per point
    pad = 12 * pt                             # box padding + clearance
    dys = np.arange(20, 500, 6)
    dxs = np.arange(-260, 261, 10)
    cands = sorted(((np.hypot(dx, dy - 20), dx, dy)
                    for dy in dys for dx in dxs))

    def curve_pixels():
        p = ax.transData.transform(np.column_stack([xs, ys]))
        t = np.linspace(0.0, 1.0, 25)[:, None, None]
        return (p[None, :-1] * (1 - t) + p[None, 1:] * t).reshape(-1, 2)

    def is_free(dx, dy, pix, renderer):
        ann.set_position((dx, dy))
        bb = ann.get_window_extent(renderer)
        x0, x1 = bb.x0 - pad, bb.x1 + pad
        y0, y1 = bb.y0 - pad, bb.y1 + pad
        axb = ax.get_window_extent(renderer)
        if x0 < axb.x0 or x1 > axb.x1 or y1 > axb.y1:
            return False
        if np.any((pix[:, 0] > x0) & (pix[:, 0] < x1)
                  & (pix[:, 1] > y0) & (pix[:, 1] < y1)):
            return False
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(renderer)
            if not (x1 < lb.x0 or x0 > lb.x1 or y1 < lb.y0 or y0 > lb.y1):
                return False
        return True

    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pix = curve_pixels()
        for _, dx, dy in cands:
            if is_free(dx, dy, pix, renderer):
                return ann
        lo, hi = ax.get_ylim()                # no room: add headroom on top
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))

    ann.set_position((0, 20))
    return ann


def plot_three_profiles(Rg, Ag, L1, L2, out_path):
    """
    Three panels, one per parameter, each with its own y-axis:

        Panel 1: L_comb vs R    (best W_M1 and W_M2 for each R)
        Panel 2: L_comb vs W_M1 (R and W_M2 fixed at the global best)
        Panel 3: L_comb vs W_M2 (R and W_M1 fixed at the global best)
    """
    # combined loss curve vs R (already have this)
    curve_R, j1_best, j2_best = separable_minimum(L1, L2)

    i_star = int(np.nanargmin(curve_R))
    R_star  = Rg[i_star]
    W1_star = Ag[j1_best[i_star]]
    W2_star = Ag[j2_best[i_star]]
    L_star  = curve_R[i_star]

    # profiles vs W1 and W2 at the global best (R_star, the other W fixed)
    L_vs_W1 = np.sqrt(L1[i_star, :] ** 2 + L2[i_star, j2_best[i_star]] ** 2)
    L_vs_W2 = np.sqrt(L1[i_star, j1_best[i_star]] ** 2 + L2[i_star, :] ** 2)

    textbox = (f"{BEST_LABEL}\n"
               f"W_M1={W1_star:.4f}\n"
               f"W_M2={W2_star:.4f}\n"
               f"R={R_star:.4f}\n"
               f"L={L_star:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))   # independent y-axes

    panels = [
        (Rg, curve_R, "Reflectivity",            "C0"),
        (Ag, L_vs_W1, "W (Module 1)",            "C2"),
        (Ag, L_vs_W2, "W (Module 2)",            "C3"),
    ]

    stars = []
    for ax, (x, y, xlabel, col) in zip(axes, panels):
        ax.plot(x, y, "o-", color=col, lw=2.2, ms=8)

        idx = int(np.nanargmin(y))
        sx, sy = x[idx], y[idx]

        ax.axvline(sx, color="black", ls="--", alpha=0.6)
        ax.plot([sx], [sy], marker="*", ms=28, color="black",
                markeredgecolor="white", markeredgewidth=1.8,
                linestyle="none", zorder=5, label=BEST_LABEL)

        stars.append((ax, sx, sy, x, y))

        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(CBAR_LABEL, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=14, framealpha=0.9)

    fig.suptitle("Michel Parameter Scan: Combined Loss vs Each Parameter",
                 fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # stat boxes go on last, once the axes have their final size
    for ax, sx, sy, x, y in stars:
        _annotate_above_star(fig, ax, sx, sy, textbox, x, y)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# NEW: each module's loss and the combined loss vs R
# =====================================================================

def plot_modules_vs_R(Rg, Ag, L1, L2, out_path):
    """
    L_M1, L_M2 and L_comb vs R, each module at its own best W for that R.

    Shows which module drives the combined loss across the good-fit
    range of R.
    """
    curve, j1_best, j2_best = separable_minimum(L1, L2)
    best1 = along_best_path(L1, j1_best)
    best2 = along_best_path(L2, j2_best)

    i_min = int(np.nanargmin(curve))
    bR = Rg[i_min]
    bA1 = Ag[j1_best[i_min]]
    bA2 = Ag[j2_best[i_min]]
    bL = curve[i_min]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(Rg, best1, "s-", color="C2", lw=2.0, ms=7,
            label=r"Module 1:  $L_1(R, W_1)$")
    ax.plot(Rg, best2, "^-", color="C3", lw=2.0, ms=7,
            label=r"Module 2:  $L_2(R, W_2)$")
    ax.plot(Rg, curve, "o-", color="C0", lw=2.2, ms=8,
            label=r"Combined:  $\sqrt{L_1^2 + L_2^2}$")
    ax.axvline(bR, color="black", ls="--", alpha=0.6)
    ax.plot([bR], [bL], marker="*", ms=28, color="black",
            markeredgecolor="white", markeredgewidth=1.8, zorder=5,
            linestyle="none", label=BEST_LABEL)

    ax.annotate(f"{BEST_LABEL}\nW_M1={bA1:.4f}\nW_M2={bA2:.4f}\n"
                f"R={bR:.4f}\nL={bL:.2f}\n"
                f"L_M1={best1[i_min]:.2f}\nL_M2={best2[i_min]:.2f}",
                xy=(bR, bL), xytext=(0.5, 0.97), textcoords="axes fraction",
                ha="center", va="top", fontsize=14, zorder=6, bbox=ANNOT_BOX)

    ax.set_xlabel(XLABEL_R, fontsize=22)
    ax.set_ylabel("Loss", fontsize=22)
    ax.set_title(TITLE_VS_R, fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# NEW: best W for each module vs R
# =====================================================================

def plot_best_W_vs_R(Rg, Ag, L1, L2, out_path):
    """Best W_M1 and best W_M2 as a function of the shared R."""
    curve, j1_best, j2_best = separable_minimum(L1, L2)
    W1 = np.where(j1_best >= 0, Ag[np.clip(j1_best, 0, None)], np.nan)
    W2 = np.where(j2_best >= 0, Ag[np.clip(j2_best, 0, None)], np.nan)

    i_min = int(np.nanargmin(curve))
    bR = Rg[i_min]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(Rg, W1, "s-", color="C2", lw=2.0, ms=7, label="Module 1")
    ax.plot(Rg, W2, "^-", color="C3", lw=2.0, ms=7, label="Module 2")
    ax.axvline(bR, color="black", ls="--", alpha=0.6)
    ax.plot([bR, bR], [W1[i_min], W2[i_min]], marker="*", ms=28,
            color="black", markeredgecolor="white", markeredgewidth=1.8,
            zorder=5, linestyle="none", label=BEST_LABEL)

    ax.set_xlabel(XLABEL_R, fontsize=22)
    ax.set_ylabel(f"Best {YLABEL_W}", fontsize=20)
    ax.set_title("Michel Parameter Scan: Best Attenuation vs Reflectivity",
                 fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# NEW: mean and width residuals vs R
# =====================================================================

def plot_residuals_vs_R(Rg, L1, L2, stats, out_path):
    """
    mu_sim - mu_data and sigma_sim - sigma_data vs R for both modules,
    each at its best W for that R.  Shows whether a module's loss comes
    from the peak position or from the width.
    """
    curve, j1_best, j2_best = separable_minimum(L1, L2)
    bR = Rg[int(np.nanargmin(curve))]

    dmu1 = along_best_path(stats["dmu1"], j1_best)
    dmu2 = along_best_path(stats["dmu2"], j2_best)
    dsg1 = along_best_path(stats["dsig1"], j1_best)
    dsg2 = along_best_path(stats["dsig2"], j2_best)

    if np.all(np.isnan(dmu1)) and np.all(np.isnan(dmu2)):
        print(f"Skipped: {out_path}  (no mu/sigma fields in the JSON files)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    panels = [
        (dmu1, dmu2, r"$\mu_{sim} - \mu_{data}$"),
        (dsg1, dsg2, r"$\sigma_{sim} - \sigma_{data}$"),
    ]

    for ax, (y1, y2, ylabel) in zip(axes, panels):
        ax.axhline(0.0, color="gray", lw=1.2)
        ax.plot(Rg, y1, "s-", color="C2", lw=2.0, ms=7, label="Module 1")
        ax.plot(Rg, y2, "^-", color="C3", lw=2.0, ms=7, label="Module 2")
        ax.axvline(bR, color="black", ls="--", alpha=0.6)

        ax.set_xlabel(XLABEL_R, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=14, framealpha=0.9)

    fig.suptitle("Michel Parameter Scan: Simulation - Data vs Reflectivity",
                 fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# NEW: combined loss vs R with statistical error bars
# =====================================================================

def plot_combined_vs_R_errorbars(Rg, Ag, L1, L2, stats, out_path):
    """
    L_comb(R) as in plot_combined_vs_R, with error bars from the
    simulation statistics (see build_stat_grids).
    """
    curve, j1_best, j2_best = separable_minimum(L1, L2)
    best1 = along_best_path(L1, j1_best)
    best2 = along_best_path(L2, j2_best)
    e1 = along_best_path(stats["err1"], j1_best)
    e2 = along_best_path(stats["err2"], j2_best)

    if np.all(np.isnan(e1)) or np.all(np.isnan(e2)):
        print(f"Skipped: {out_path}  (no n_events/sigma fields in the JSON files)")
        return

    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.sqrt((best1 * e1) ** 2 + (best2 * e2) ** 2) / curve

    i_min = int(np.nanargmin(curve))
    bR = Rg[i_min]
    bA1 = Ag[j1_best[i_min]]
    bA2 = Ag[j2_best[i_min]]
    bL = curve[i_min]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.errorbar(Rg, curve, yerr=err, fmt="o-", color="C0", lw=2.2, ms=8,
                ecolor="C0", elinewidth=1.6, capsize=4)
    ax.axvline(bR, color="black", ls="--", alpha=0.6)
    ax.plot([bR], [bL], marker="*", ms=28, color="black",
            markeredgecolor="white", markeredgewidth=1.8, zorder=5,
            linestyle="none", label=BEST_LABEL)

    ax.annotate(f"{BEST_LABEL}\nW_M1={bA1:.4f}\nW_M2={bA2:.4f}\n"
                f"R={bR:.4f}\nL={bL:.2f} ± {err[i_min]:.2f}",
                xy=(bR, bL), xytext=(22, 25), textcoords="offset points",
                fontsize=14, zorder=6, bbox=ANNOT_BOX)

    ax.set_xlabel(XLABEL_R, fontsize=22)
    ax.set_ylabel(CBAR_LABEL, fontsize=16)
    ax.set_title(TITLE_VS_R, fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# main
# =====================================================================

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_points(args.results, args.tag)
    Rg, Ag, L1, L2 = build_grids(rows)
    nR, nA = L1.shape
    stats = build_stat_grids(rows, L1.shape)

    print(f"Output folder: {os.path.abspath(args.outdir)}")
    print(f"Points used : {len(rows)}")
    print(f"Grid shape  : {nR} R x {nA} W")
    print()

    # ---------- 0) non-combined heatmaps: each module's own loss ----------
    single = {}
    for name, LM in (("Module1", L1), ("Module2", L2)):
        single[name] = plot_heatmap(
            Rg, Ag, LM,
            os.path.join(args.outdir, f"loss_heatmap_{name}.png"),
            title=f"{TITLE_HEATMAP} ({name.replace('Module', 'Module ')})",
            cbar_label=CBAR_LABEL_SINGLE,
        )

    # ---------- 1) L_comb vs R using separable best W per module ----------
    curve, j1b, j2b, (bR, bA1, bA2, bL) = plot_combined_vs_R(
        Rg, Ag, L1, L2,
        os.path.join(args.outdir, "loss_comb_vs_R_Module1_Module2.png"),
    )

    # ---------- 2,3) combined loss on each module's (R, W) plane ----------
    Lc1, Lc2 = combined_loss_per_module(L1, L2)
    bR_1, bW_1, bL_1 = plot_heatmap(
        Rg, Ag, Lc1,
        os.path.join(args.outdir, "loss_comb_heatmap_Module1.png"),
        title=f"{TITLE_HEATMAP} (Module 1)",
        cmap=args.cmap_combined,
        star_face="white", star_edge="black",   # visible on a dark minimum
    )
    bR_2, bW_2, bL_2 = plot_heatmap(
        Rg, Ag, Lc2,
        os.path.join(args.outdir, "loss_comb_heatmap_Module2.png"),
        title=f"{TITLE_HEATMAP} (Module 2)",
        cmap=args.cmap_combined,
        star_face="white", star_edge="black",   # visible on a dark minimum
    )

    # ---------- 4) combined loss on the (W_M1, W_M2) plane at best R ----------
    plot_W1_W2_at_best_R(
        Rg, Ag, L1, L2,
        os.path.join(args.outdir, "loss_comb_heatmap_W1_W2_Module1_Module2.png"),
        cmap=args.cmap_combined,
    )

    # ---------- 5) three 1D profiles (W panels at the best R) ----------
    plot_three_profiles(
        Rg, Ag, L1, L2,
        os.path.join(args.outdir, "loss_comb_3profiles.png"),
    )

    # ---------- 6) NEW: L_M1, L_M2 and L_comb vs R ----------
    plot_modules_vs_R(
        Rg, Ag, L1, L2,
        os.path.join(args.outdir, "loss_modules_vs_R.png"),
    )

    # ---------- 7) NEW: best W per module vs R ----------
    plot_best_W_vs_R(
        Rg, Ag, L1, L2,
        os.path.join(args.outdir, "best_W_vs_R.png"),
    )

    # ---------- 8) NEW: mean and width residuals vs R ----------
    plot_residuals_vs_R(
        Rg, L1, L2, stats,
        os.path.join(args.outdir, "residuals_vs_R.png"),
    )

    # ---------- 9) NEW: L_comb vs R with error bars ----------
    plot_combined_vs_R_errorbars(
        Rg, Ag, L1, L2, stats,
        os.path.join(args.outdir, "loss_comb_vs_R_errorbars.png"),
    )

    # ---------- summary ----------
    print()
    print("=" * 68)
    print("NON-COMBINED LOSS  (each module on its own)")
    print("=" * 68)
    for name, (r_, w_, l_) in single.items():
        print(f"{name:<17}: W={w_:.4f}  R={r_:.4f}  L={l_:.3f}")
    print("=" * 68)
    print("COMBINED LOSS  L = sqrt( L_M1^2 + L_M2^2 )")
    print("=" * 68)
    print(f"Best fit         : W_M1={bA1:.4f}  W_M2={bA2:.4f}  "
          f"R={bR:.4f}  L={bL:.3f}")
    print(f"Module 1 plot    : W={bW_1:.4f}  R={bR_1:.4f}  L={bL_1:.3f}")
    print(f"Module 2 plot    : W={bW_2:.4f}  R={bR_2:.4f}  L={bL_2:.3f}")
    print("=" * 68)
    print()
    print("Interpretation:")
    print(" - If the best fit sits at W_M1 != W_M2, the two modules")
    print("   prefer different attenuation coefficients.")
    print(" - If R is on the grid edge, extend the reflectivity range.")

    # ---------- edge warning ----------
    warn = []
    i_c = int(np.nanargmin(curve))
    if i_c in (0, nR - 1):
        warn.append("shared R on grid edge")
    if j1b[i_c] in (0, nA - 1):
        warn.append("W_M1 on grid edge")
    if j2b[i_c] in (0, nA - 1):
        warn.append("W_M2 on grid edge")
    for w in warn:
        print(f"WARNING: {w}. Consider extending the grid range.")


if __name__ == "__main__":
    main()
