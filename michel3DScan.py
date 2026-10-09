#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Separable 2-module Michel parameter scan (equivalent to a 3D scan).

Grid: 41 reflectivity values x 34 attenuation values = 1394 grid points.
Each grid point runs BOTH modules (M1 and M2) at the same (R, A):

    Module 1 (light water)  ->  buildm1/G4d2o   ->  data_m1/
    Module 2 (heavy water)  ->  build/G4d2o     ->  data/

So the total number of Geant4 jobs is:
    1394 M1 runs + 1394 M2 runs = 2788 jobs

Because each module is allowed its OWN attenuation coefficient A_M1, A_M2
but they must SHARE the Tyvek reflectivity R (Tyvek is always adjacent to
light water), the combined loss is separable:

    L(R, A_M1, A_M2) = L_M1(R, A_M1) + L_M2(R, A_M2)

    min_{A_M1,A_M2} L(R, A_M1, A_M2)
        = min_{A_M1} L_M1(R, A_M1) + min_{A_M2} L_M2(R, A_M2)

So an (nR x nA) scan that runs both modules at every point recovers the
exact 3D optimum at 1/nA of the cost of a brute-force (nR x nA x nA) grid.

Usage:
    python michel3Dscan.py --mode info
    python michel3Dscan.py --mode run --index N
    python michel3Dscan.py --mode collect
"""

import argparse
import glob
import json
import os
import subprocess
import time
import random

import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4H2O"

# ---------------- Module 1 : light water ----------------
GEANT4_EXEC_M1 = os.path.join(BASE_DIR, "buildm1", "G4d2o")
DATA_DIR_M1    = os.path.join(BASE_DIR, "data_m1")
REAL_DATA_M1   = os.path.join(BASE_DIR, "mac", "all_histogramsm1.root")
MIN_PE_THRESHOLD_M1 = 40.0     # <-- M1's own PE threshold
MIN_HITS_PER_PMT_M1 = 1
N_PMTS_M1           = 8
G4_OPTIONS_M1 = ["MichelElectronPGA", "NoVis", "NoTracks",
                 "PhysicsOn", "NeutronHP", "OutputDir", "./data_m1"]

# ---------------- Module 2 : heavy water ----------------
GEANT4_EXEC_M2 = os.path.join(BASE_DIR, "build", "G4d2o")
DATA_DIR_M2    = os.path.join(BASE_DIR, "data")
REAL_DATA_M2   = os.path.join(BASE_DIR, "mac", "all_histogramsgood.root")
MIN_PE_THRESHOLD_M2 = 60.0     # <-- M2's own PE threshold
MIN_HITS_PER_PMT_M2 = 2
N_PMTS_M2           = 12
G4_OPTIONS_M2 = ["MichelElectronPGA", "NoVis", "NoTracks",
                 "PhysicsOn", "NeutronHP", "OutputDir", "./data"]

# ---------------- shared run settings ----------------
N_PRIMARIES        = 20000
MU_WEIGHT          = 2.0
GEANT4_TIMEOUT_SEC = 3 * 3600

# ---------------- grids ----------------
REFLECTIVITY_GRID = np.round(np.linspace(0.80, 1.00, 41), 4)   # 41 pts, step 0.005
H2OREFL_GRID      = np.round(np.linspace(0.20, 1.00, 34), 4)   # 34 pts, step ~0.0242
# -> 41 x 34 = 1394 grid points; each runs BOTH modules
# -> 1394 M1 runs + 1394 M2 runs = 2788 Geant4 jobs total

SCAN_TAG    = "sep2M_41x34_R0.80-1.00_A0.20-1.00"
RESULTS_DIR = "results_3Dscan"
PLOT_OUT    = "parameter_scan_3Dscan_loss_surface.png"

RUN_NUMBER_OFFSET_M1 = 100000
RUN_NUMBER_OFFSET_M2 = 500

MIN_DELAY = 0.5
MAX_DELAY = 3.0

os.makedirs(DATA_DIR_M1, exist_ok=True)


# ============================================================================
# Grid indexing
# ============================================================================

def total_grid_points():
    return len(REFLECTIVITY_GRID) * len(H2OREFL_GRID)


def grid_point_for_index(index):
    nR = len(REFLECTIVITY_GRID)
    nA = len(H2OREFL_GRID)
    total = nR * nA
    if not (0 <= index < total):
        raise ValueError(f"index {index} out of range 0..{total - 1}")
    i, j = divmod(index, nA)     # i = reflectivity index, j = attenuation index
    return float(REFLECTIVITY_GRID[i]), float(H2OREFL_GRID[j]), i, j


# ============================================================================
# PMT quality mask
# ============================================================================

def build_pmt_quality_mask(pmt_num, min_hits_per_pmt, n_pmts):
    """
    Per-event mask: True only if every PMT 0..n_pmts-1 registered at least
    min_hits_per_pmt hits in that event.
    """
    mask = np.ones(len(pmt_num), dtype=bool)
    for p in range(n_pmts):
        hits_on_p = ak.to_numpy(ak.sum(pmt_num == p, axis=1))
        mask &= hits_on_p >= min_hits_per_pmt
    return mask


# ============================================================================
# Real-data mu / sigma (weighted mean of the thresholded histogram)
# ============================================================================

def get_real_data_mu_sigma(path, min_pe):
    with uproot.open(path) as f:
        h = f["michel_energy"]
        edges  = h.axis().edges()
        counts = h.values()
        centers = (edges[:-1] + edges[1:]) / 2

    first_bin = np.where(centers >= min_pe)[0][0]
    c = counts[first_bin:]
    x = centers[first_bin:]

    total = c.sum()
    mu = np.sum(x * c) / total
    var = np.sum(c * (x - mu) ** 2) / total
    sigma = np.sqrt(var)
    print(f"    real data [{os.path.basename(path)}]: "
          f"mu={mu:.2f} sigma={sigma:.2f} (from {total:.0f} events, cut {min_pe} PE)")
    return float(mu), float(sigma)


# ============================================================================
# Geant4 launch
# ============================================================================

def geant4_command(exe, run_number, reflectivity, h2orefl, extra_options):
    return ([exe, str(run_number),
             "Reflectivity", f"{reflectivity:.6f}",
             "H2oRefl",      f"{h2orefl:.6f}"]
            + ["EVENTS", str(N_PRIMARIES)] + extra_options)


def run_geant4(exe, sim_output_path, run_number, reflectivity, h2orefl, extra_options):
    if os.path.exists(sim_output_path):
        os.remove(sim_output_path)

    cmd = geant4_command(exe, run_number, reflectivity, h2orefl, extra_options)
    print("    [Geant4] " + " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR,
                                timeout=GEANT4_TIMEOUT_SEC,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    [Geant4] exit code {result.returncode}")
            if result.stderr:
                print(f"    [Geant4 stderr] {result.stderr[-2000:]}")
            return False
        if not os.path.exists(sim_output_path):
            print(f"    [Geant4] expected output missing: {sim_output_path}")
            if result.stdout:
                print(f"    [Geant4 stdout tail] {result.stdout[-2000:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    [Geant4] TIMED OUT")
        return False
    except Exception as e:
        print(f"    [Geant4] failed to launch: {e}")
        return False


def read_used_parameters(data_dir, run_number):
    """Read Reflectivity / H2oRefl from data_dir/beamOnFiles/beamOn<run>.dat."""
    path = os.path.join(data_dir, "beamOnFiles", f"beamOn{run_number:03d}.dat")
    refl = h2o = None
    try:
        with open(path) as f:
            for line in f:
                if "//" not in line:
                    continue
                value, desc = line.split("//", 1)
                if "ReflectivityOfTyvek" in desc:
                    refl = float(value.strip())
                elif "H2oAttenuationLengthCoefficient" in desc:
                    h2o = float(value.strip())
    except (OSError, ValueError):
        pass
    return refl, h2o


# ============================================================================
# Sim histogram -> mu, sigma with PMT quality cut
# ============================================================================

def load_simulation_statistics(sim_path, min_pe, min_hits_per_pmt, n_pmts):
    try:
        with uproot.open(sim_path) as sf:
            tree = sf["Sim_Tree"]
            num_hits = tree["eventData/numHits"].array().to_numpy()
            pmt_num  = tree["pmtHits/pmtHits.pmtNum"].array()

        n_all = len(num_hits)
        pmt_mask = build_pmt_quality_mask(pmt_num, min_hits_per_pmt, n_pmts)
        pe_mask  = num_hits >= min_pe
        final    = pe_mask & pmt_mask
        kept     = num_hits[final]

        print(f"    sim events: {n_all} total -> {pmt_mask.sum()} pass {min_hits_per_pmt} PE/PMT "
              f"-> {len(kept)} pass both cuts")

        if len(kept) == 0:
            return None, None, 0
        return float(np.mean(kept)), float(np.std(kept)), int(len(kept))
    except Exception as e:
        print(f"    [ROOT read] failed: {e}")
        return None, None, 0


def loss(mu_sim, sigma_sim, mu_data, sigma_data, mu_weight=MU_WEIGHT):
    return float(np.sqrt((mu_weight * (mu_sim - mu_data)) ** 2
                         + (sigma_sim - sigma_data) ** 2))


# ============================================================================
# Mode: info
# ============================================================================

def do_info():
    n = total_grid_points()
    print(f"Reflectivity grid ({len(REFLECTIVITY_GRID)}): {REFLECTIVITY_GRID.tolist()}")
    print(f"H2oRefl grid      ({len(H2OREFL_GRID)}): {H2OREFL_GRID.tolist()}")
    print(f"Grid points: {n}")
    print(f"Per-module points: {n}   (M1 jobs = {n}, M2 jobs = {n}, total = {2*n})")
    print(f"Run numbers M1: {RUN_NUMBER_OFFSET_M1}..{RUN_NUMBER_OFFSET_M1 + n - 1}")
    print(f"Run numbers M2: {RUN_NUMBER_OFFSET_M2}..{RUN_NUMBER_OFFSET_M2 + n - 1}")
    print(f"Results dir: {RESULTS_DIR}   (scan tag: {SCAN_TAG})")
    if n > 1000:
        print(f"  SLURM part 1: sbatch --array=0-999%8 3Dscan.slurm")
        print(f"  SLURM part 2: sbatch --array=0-{n-1001}%8 --export=ALL,INDEX_OFFSET=1000 3Dscan.slurm")
    else:
        print(f"  SLURM:        sbatch --array=0-{n-1}%8 3Dscan.slurm")


# ============================================================================
# Mode: run
# ============================================================================

def do_run(index):
    R, A, i, j = grid_point_for_index(index)
    run_m1 = RUN_NUMBER_OFFSET_M1 + index
    run_m2 = RUN_NUMBER_OFFSET_M2 + index

    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)

    print(f"\n=== Task {index}: grid[{i}][{j}]  R={R:.4f}  A={A:.4f}  "
          f"runM1={run_m1} runM2={run_m2}  (delay {delay:.2f}s) ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"point_{index}.json")

    record = {
        "scan": SCAN_TAG,
        "index": index, "i": i, "j": j,
        "reflectivity": R, "h2orefl": A,
        "n_primaries": N_PRIMARIES,
        "run_m1": run_m1, "run_m2": run_m2,
        "status": "failed",
    }

    # ---- real data targets for both modules ----
    try:
        muD1, sigD1 = get_real_data_mu_sigma(REAL_DATA_M1, MIN_PE_THRESHOLD_M1)
        muD2, sigD2 = get_real_data_mu_sigma(REAL_DATA_M2, MIN_PE_THRESHOLD_M2)
        record.update({"mu_data_M1": muD1, "sigma_data_M1": sigD1,
                       "mu_data_M2": muD2, "sigma_data_M2": sigD2})
    except Exception as e:
        record["error"] = f"real data read failed: {e}"
        print("FAILED:", record["error"])
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        return

    # ---- Module 1 ----
    sim_m1 = os.path.join(DATA_DIR_M1, f"Sim_D2ODetector{run_m1:03d}.root")
    print("  -- Module 1 (light water) --")
    t0 = time.time()
    ok1 = run_geant4(GEANT4_EXEC_M1, sim_m1, run_m1, R, A, G4_OPTIONS_M1)
    record["geant4_seconds_M1"] = time.time() - t0
    if not ok1:
        record["error"] = "Module 1 Geant4 run failed"
    else:
        usedR, usedA = read_used_parameters(DATA_DIR_M1, run_m1)
        record["used_reflectivity_M1"] = usedR
        record["used_h2orefl_M1"]      = usedA
        if usedR is None or usedA is None:
            print("    [check M1] could not read beamOn copy; params not verified")
        elif abs(usedR - R) > 1e-5 or abs(usedA - A) > 1e-5:
            ok1 = False
            record["error"] = (f"Module 1 used R={usedR}, A={usedA}, "
                               f"expected {R}, {A}")
        else:
            print(f"    [check M1] used R={usedR}, A={usedA}  OK")

        if ok1:
            mu1, sig1, n1 = load_simulation_statistics(
                sim_m1, MIN_PE_THRESHOLD_M1, MIN_HITS_PER_PMT_M1, N_PMTS_M1)
            if mu1 is None:
                ok1 = False
                record["error"] = "Module 1 produced zero events above threshold"
            else:
                L1 = loss(mu1, sig1, muD1, sigD1)
                record.update({"mu_sim_M1": mu1, "sigma_sim_M1": sig1,
                               "n_events_M1": n1, "loss_M1": L1})
                print(f"    M1: mu_sim={mu1:.2f} sigma_sim={sig1:.2f} "
                      f"loss_M1={L1:.3f}  (n={n1})")

    # ---- Module 2 ----
    sim_m2 = os.path.join(DATA_DIR_M2, f"Sim_D2ODetector{run_m2:03d}.root")
    print("  -- Module 2 (heavy water) --")
    t0 = time.time()
    ok2 = run_geant4(GEANT4_EXEC_M2, sim_m2, run_m2, R, A, G4_OPTIONS_M2)
    record["geant4_seconds_M2"] = time.time() - t0
    if not ok2:
        record["error"] = (record.get("error", "") + " | Module 2 Geant4 run failed").strip(" |")
    else:
        usedR, usedA = read_used_parameters(DATA_DIR_M2, run_m2)
        record["used_reflectivity_M2"] = usedR
        record["used_h2orefl_M2"]      = usedA
        if usedR is None or usedA is None:
            print("    [check M2] could not read beamOn copy; params not verified")
        elif abs(usedR - R) > 1e-5 or abs(usedA - A) > 1e-5:
            ok2 = False
            record["error"] = (record.get("error", "") +
                               f" | Module 2 used R={usedR}, A={usedA}, "
                               f"expected {R}, {A}").strip(" |")
        else:
            print(f"    [check M2] used R={usedR}, A={usedA}  OK")

        if ok2:
            mu2, sig2, n2 = load_simulation_statistics(
                sim_m2, MIN_PE_THRESHOLD_M2, MIN_HITS_PER_PMT_M2, N_PMTS_M2)
            if mu2 is None:
                ok2 = False
                record["error"] = (record.get("error", "") +
                                   " | Module 2 produced zero events above threshold").strip(" |")
            else:
                L2 = loss(mu2, sig2, muD2, sigD2)
                record.update({"mu_sim_M2": mu2, "sigma_sim_M2": sig2,
                               "n_events_M2": n2, "loss_M2": L2})
                print(f"    M2: mu_sim={mu2:.2f} sigma_sim={sig2:.2f} "
                      f"loss_M2={L2:.3f}  (n={n2})")

    # ---- combine ----
    if ok1 and ok2:
        record["loss_total"] = record["loss_M1"] + record["loss_M2"]
        record["status"] = "ok"
        print(f"  => total loss = {record['loss_total']:.4f}")

    # ---- cleanup ROOTs ----
    for p in (sim_m1, sim_m2):
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"    [deleted {os.path.basename(p)}]")
            except OSError as e:
                print(f"    [could not delete {p}: {e}]")

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {out_path}")


# ============================================================================
# Mode: collect  (2D scan of R x A, then exact 3D optimum by separability)
# ============================================================================

def do_collect():
    nR, nA = len(REFLECTIVITY_GRID), len(H2OREFL_GRID)
    loss1 = np.full((nR, nA), np.nan)
    loss2 = np.full((nR, nA), np.nan)
    mu1g  = np.full((nR, nA), np.nan)
    mu2g  = np.full((nR, nA), np.nan)
    sg1g  = np.full((nR, nA), np.nan)
    sg2g  = np.full((nR, nA), np.nan)

    muD1 = sigD1 = muD2 = sigD2 = None
    n_ok = n_fail = n_stale = 0

    for path in glob.glob(os.path.join(RESULTS_DIR, "point_*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("scan") != SCAN_TAG:
            n_stale += 1
            continue
        i, j = rec.get("i"), rec.get("j")
        if not (0 <= i < nR and 0 <= j < nA):
            n_stale += 1
            continue
        if rec.get("index") != i * nA + j:
            n_stale += 1
            continue
        if abs(rec.get("reflectivity", -1) - REFLECTIVITY_GRID[i]) > 1e-6:
            n_stale += 1
            continue
        if abs(rec.get("h2orefl", -1) - H2OREFL_GRID[j]) > 1e-6:
            n_stale += 1
            continue

        if muD1 is None and "mu_data_M1" in rec:
            muD1, sigD1 = rec["mu_data_M1"], rec["sigma_data_M1"]
            muD2, sigD2 = rec["mu_data_M2"], rec["sigma_data_M2"]

        if rec.get("status") == "ok":
            loss1[i, j] = rec["loss_M1"]
            loss2[i, j] = rec["loss_M2"]
            mu1g[i, j]  = rec["mu_sim_M1"]
            mu2g[i, j]  = rec["mu_sim_M2"]
            sg1g[i, j]  = rec["sigma_sim_M1"]
            sg2g[i, j]  = rec["sigma_sim_M2"]
            n_ok += 1
        else:
            n_fail += 1
            print(f"  point {rec.get('index')} FAILED: {rec.get('error')}")

    if n_stale:
        print(f"Skipped {n_stale} stale JSON files.")
    expected = nR * nA
    missing = expected - n_ok - n_fail
    print(f"Loaded {n_ok} ok + {n_fail} failed = {n_ok + n_fail} / {expected}"
          + (f"  ({missing} not run yet)" if missing else ""))

    if np.all(np.isnan(loss1)) or np.all(np.isnan(loss2)):
        raise RuntimeError("No successful grid points to plot")

    # ---- separable 3D minimum ----
    # For each R row i: bestA_M1(i) = argmin_j loss1[i,j], same for M2.
    # Combined loss at (R_i, A*_M1(i), A*_M2(i)) = min_j loss1 + min_j loss2.
    best_row_loss = np.full(nR, np.nan)
    best_j_M1     = np.full(nR, -1, dtype=int)
    best_j_M2     = np.full(nR, -1, dtype=int)
    for i in range(nR):
        r1 = loss1[i, :]
        r2 = loss2[i, :]
        if np.all(np.isnan(r1)) or np.all(np.isnan(r2)):
            continue
        j1 = int(np.nanargmin(r1))
        j2 = int(np.nanargmin(r2))
        best_j_M1[i] = j1
        best_j_M2[i] = j2
        best_row_loss[i] = r1[j1] + r2[j2]

    i_star = int(np.nanargmin(best_row_loss))
    best_R    = float(REFLECTIVITY_GRID[i_star])
    best_A_M1 = float(H2OREFL_GRID[best_j_M1[i_star]])
    best_A_M2 = float(H2OREFL_GRID[best_j_M2[i_star]])
    best_L    = float(best_row_loss[i_star])

    print("\n" + "=" * 64)
    print("SEPARABLE 3D SCAN - BEST FIT")
    print("=" * 64)
    print(f"Reflectivity (shared) : {best_R:.4f}")
    print(f"H2oRefl Module 1      : {best_A_M1:.4f}")
    print(f"H2oRefl Module 2      : {best_A_M2:.4f}")
    print(f"Loss M1               : {loss1[i_star, best_j_M1[i_star]]:.4f}")
    print(f"Loss M2               : {loss2[i_star, best_j_M2[i_star]]:.4f}")
    print(f"Total loss            : {best_L:.4f}")
    print("=" * 64)

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Panel 1: total loss as a function of R (best A per module)
    ax = axes[0]
    ax.plot(REFLECTIVITY_GRID, best_row_loss, "o-", color="C0", lw=2)
    ax.axvline(best_R, color="k", ls="--", alpha=0.6)
    ax.set_xlabel("Reflectivity of Tyvek")
    ax.set_ylabel(r"$\min_{A_{M1}} L_{M1} + \min_{A_{M2}} L_{M2}$")
    ax.set_title(f"Total loss vs R (best A per module)\n"
                 f"best R={best_R:.3f},  A_M1={best_A_M1:.3f},  A_M2={best_A_M2:.3f}")
    ax.grid(alpha=0.3)

    # Panel 2 & 3: per-module 2D loss surfaces, star at best
    for ax, grid, name, best_A in (
        (axes[1], loss1, "Module 1 (light water)", best_A_M1),
        (axes[2], loss2, "Module 2 (heavy water)", best_A_M2),
    ):
        im = ax.pcolormesh(REFLECTIVITY_GRID, H2OREFL_GRID, grid.T,
                           shading="nearest", cmap="cividis_r")
        fig.colorbar(im, ax=ax).set_label("loss")
        ax.plot(best_R, best_A, marker="*", ms=18,
                color="white", markeredgecolor="black", mew=1.2, zorder=5)
        ax.set_xlabel("Reflectivity of Tyvek")
        ax.set_ylabel("H2O attenuation coefficient")
        ax.set_title(f"{name} loss  (star = chosen point)")

    sup = "Separable 2-module scan"
    if muD1 is not None:
        sup += (f"   |   M1: mu_d={muD1:.1f} sig_d={sigD1:.1f} (cut {MIN_PE_THRESHOLD_M1} PE)"
                f"   |   M2: mu_d={muD2:.1f} sig_d={sigD2:.1f} (cut {MIN_PE_THRESHOLD_M2} PE)")
    fig.suptitle(sup, fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {PLOT_OUT}")

    # ---- edge warning ----
    warn = []
    if i_star in (0, nR - 1):
        warn.append("Reflectivity")
    if best_j_M1[i_star] in (0, nA - 1):
        warn.append("H2oRefl (M1)")
    if best_j_M2[i_star] in (0, nA - 1):
        warn.append("H2oRefl (M2)")
    if warn:
        print(f"WARNING: best fit on the edge of the {' & '.join(warn)} range - extend the grid.")

    # ---- dump machine-readable result ----
    summary = {
        "scan": SCAN_TAG,
        "reflectivity": best_R,
        "h2orefl_M1": best_A_M1,
        "h2orefl_M2": best_A_M2,
        "loss_M1": float(loss1[i_star, best_j_M1[i_star]]),
        "loss_M2": float(loss2[i_star, best_j_M2[i_star]]),
        "loss_total": best_L,
        "n_ok": n_ok, "n_failed": n_fail, "n_expected": expected,
        "mu_data_M1": muD1, "sigma_data_M1": sigD1,
        "mu_data_M2": muD2, "sigma_data_M2": sigD2,
    }
    with open(os.path.join(RESULTS_DIR, "best_fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {os.path.join(RESULTS_DIR, 'best_fit_summary.json')}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["info", "run", "collect"], required=True)
    parser.add_argument("--index", type=int, help="Grid index for --mode run")
    args = parser.parse_args()

    if args.mode == "info":
        do_info()
    elif args.mode == "run":
        if args.index is None:
            parser.error("--mode run requires --index")
        do_run(args.index)
    else:
        do_collect()
