#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Michel Spectrum Parameter Scan: Reflectivity x H2oAttenuationLengthCoefficient
- WIDE GRID: Reflectivity 0.80-1.00 (step 0.005, 41 points)
             H2oAttenuationLengthCoefficient 0.20-1.00 (step 0.025, 33 points)
             -> 1353 grid points
- WITH PMT QUALITY CUT (>= 2 PE per PMT), matching hasMultiplicityCut.py

Each grid point passes its Reflectivity / H2oRefl to G4d2o on the command
line, so parallel array tasks never share (or overwrite) beamOn.dat. The
remaining settings come from beamOn.dat, which is only read, never written.

Usage:
    python michel_parameter_scan.py --mode info            # grid size / SLURM array range
    python michel_parameter_scan.py --mode run --index N   # one grid point (SLURM array task)
    python michel_parameter_scan.py --mode collect         # loss surface + best fit
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
DATA_DIR = os.path.join(BASE_DIR, "data")
MAC_DIR = os.path.join(BASE_DIR, "mac")
BEAMON_FILE = os.path.join(BASE_DIR, "beamOn.dat")
REAL_DATA_PATH = os.path.join(MAC_DIR, "all_histogramsm1.root")
GEANT4_EXECUTABLE = os.path.join(BASE_DIR, "buildm1", "G4d2o")

MIN_PE_THRESHOLD = 40.0

# ================================================================
# PMT QUALITY CUT (matches hasMultiplicityCut.py)
# ================================================================
MIN_HITS_PER_PMT = 1
N_PMTS = 8

# ================================================================
# WIDE GRID: 41 x 33 = 1353 points
# ================================================================
REFLECTIVITY_GRID = np.round(np.linspace(0.80, 1.00, 41), 4)   # step 0.005
H2OREFL_GRID = np.round(np.linspace(0.20, 1.00, 33), 4)        # step 0.025

# Settings forced on the command line for every run (everything else from beamOn.dat)
N_PRIMARIES = 20000
FIXED_G4_OPTIONS = ["EVENTS", str(N_PRIMARIES), "MichelElectronPGA", "NoVis", "NoTracks",
                    "PhysicsOn", "NeutronHP", "OutputDir", "./data"]

MU_WEIGHT = 2.0

# Every result file is tagged with this; collect ignores files with another tag
# (e.g. leftovers of earlier scans in the same folder). Change it for a new scan.
SCAN_TAG = "wide_41x33_R0.80-1.00_A0.20-1.00"

RESULTS_DIR = "results_600m1"
PLOT_OUT = "parameter_scan_loss_surface_600m1.png"
RUN_NUMBER_OFFSET = 500
GEANT4_TIMEOUT_SECONDS = 3 * 3600  # R -> 1.0 means almost no loss at Tyvek: longer runs

# Small random start delay so array tasks don't all hit the filesystem at once
MIN_DELAY = 0.5
MAX_DELAY = 3.0


# ============================================================================
# Grid indexing
# ============================================================================

def grid_point_for_index(index):
    n_refl = len(REFLECTIVITY_GRID)
    n_h2o = len(H2OREFL_GRID)
    total = n_refl * n_h2o
    if not (0 <= index < total):
        raise ValueError(f"index {index} out of range 0..{total - 1}")
    i, j = divmod(index, n_refl)
    return float(REFLECTIVITY_GRID[j]), float(H2OREFL_GRID[i]), i, j


def total_grid_points():
    return len(REFLECTIVITY_GRID) * len(H2OREFL_GRID)


# ============================================================================
# PMT QUALITY MASK (matches hasMultiplicityCut.py)
# ============================================================================

def build_pmt_quality_mask(pmt_num, min_hits_per_pmt=MIN_HITS_PER_PMT, n_pmts=N_PMTS):
    """
    Per-event mask: True only if every PMT (0..n_pmts-1) registered at
    least min_hits_per_pmt hits in that event.
    pmt_num is a jagged awkward array (one list of PMT indices per event,
    from pmtHits/pmtHits.pmtNum).
    """
    mask = np.ones(len(pmt_num), dtype=bool)
    for p in range(n_pmts):
        hits_on_p = ak.to_numpy(ak.sum(pmt_num == p, axis=1))
        mask &= hits_on_p >= min_hits_per_pmt
    return mask


# ============================================================================
# beamOn.dat helpers (read-only)
# ============================================================================

def get_current_primaries():
    try:
        with open(BEAMON_FILE, "r") as f:
            for line in f:
                if "//primaries-to-generate" in line:
                    return line.split("//")[0].strip()
    except Exception:
        pass
    return "unknown"


def read_used_parameters(run_number):
    """
    G4d2o writes the settings it actually ran with to
    data/beamOnFiles/beamOn<run>.dat. Return (reflectivity, h2orefl) from it,
    or (None, None) if it can't be read.
    """
    path = os.path.join(DATA_DIR, "beamOnFiles", f"beamOn{run_number:03d}.dat")
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
# GET REAL DATA μ AND σ (DIRECT WEIGHTED MEAN)
# ============================================================================

def get_real_data_mu_sigma():
    """
    Read real data histogram and compute μ and σ directly.
    Uses all_histogramsgood.root (with quality cuts applied).
    """
    with uproot.open(REAL_DATA_PATH) as f:
        h = f["michel_energy"]
        edges = h.axis().edges()
        counts = h.values()
        centers = (edges[:-1] + edges[1:]) / 2

    # Apply threshold (60 PE cut)
    first_bin = np.where(centers >= MIN_PE_THRESHOLD)[0][0]
    filtered_counts = counts[first_bin:]
    filtered_centers = centers[first_bin:]

    # Compute weighted mean and std (NO FITTING!)
    total = filtered_counts.sum()
    mu_data = np.sum(filtered_centers * filtered_counts) / total
    var_data = np.sum(filtered_counts * (filtered_centers - mu_data)**2) / total
    sigma_data = np.sqrt(var_data)

    print(f"Real data: mu={mu_data:.2f} sigma={sigma_data:.2f} (from {total:.0f} events)")
    return mu_data, sigma_data


# ============================================================================
# RUN GEANT4 WITH ALL GRID PARAMETERS ON THE COMMAND LINE
# ============================================================================

def geant4_command(run_number, reflectivity, h2orefl):
    return ([GEANT4_EXECUTABLE, str(run_number),
             "Reflectivity", f"{reflectivity:.6f}",
             "H2oRefl", f"{h2orefl:.6f}"]
            + FIXED_G4_OPTIONS)


def run_geant4(sim_output_path, run_number, reflectivity, h2orefl):
    """
    Run Geant4 with run number, Reflectivity and H2oRefl passed as
    command-line arguments (no shared beamOn.dat writes -> no race).
    """
    if os.path.exists(sim_output_path):
        os.remove(sim_output_path)

    cmd = geant4_command(run_number, reflectivity, h2orefl)
    print("    [Geant4] " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            timeout=GEANT4_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    [Geant4] non-zero exit code {result.returncode}")
            if result.stderr:
                print(f"    [Geant4 stderr] {result.stderr[-2000:]}")
            return False
        if not os.path.exists(sim_output_path):
            print(f"    [Geant4] finished but expected output not found: {sim_output_path}")
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


# ============================================================================
# LOAD SIMULATION STATISTICS WITH PMT QUALITY CUT
# ============================================================================

def load_simulation_statistics(sim_output_path):
    """
    Load simulation and compute mean/std.
    APPLIES PMT QUALITY CUT (>= 2 PE per PMT) to match the real data.
    """
    try:
        with uproot.open(sim_output_path) as sim_file:
            tree = sim_file["Sim_Tree"]

            # Get numHits (total PE per event)
            num_hits = tree["eventData/numHits"].array().to_numpy()

            # Get PMT hit information for quality cut
            pmt_num = tree["pmtHits/pmtHits.pmtNum"].array()

        n_events_all = len(num_hits)

        # Apply PMT quality cut: every PMT must have >= MIN_HITS_PER_PMT hits
        pmt_quality_mask = build_pmt_quality_mask(pmt_num)

        # Apply threshold (60 PE cut) AND PMT quality cut
        pe_mask = num_hits >= MIN_PE_THRESHOLD
        final_mask = pe_mask & pmt_quality_mask
        num_hits_filtered = num_hits[final_mask]

        print(f"    Simulation events: {n_events_all} total -> "
              f"{pmt_quality_mask.sum()} pass {MIN_HITS_PER_PMT} PE/PMT -> "
              f"{len(num_hits_filtered)} pass both cuts")

        if len(num_hits_filtered) == 0:
            return None, None, 0

        mu_sim = np.mean(num_hits_filtered)
        sigma_sim = np.std(num_hits_filtered)
        return mu_sim, sigma_sim, len(num_hits_filtered)

    except Exception as e:
        print(f"    [ROOT read] failed: {e}")
        return None, None, 0


def loss(mu_sim, sigma_sim, mu_data, sigma_data, mu_weight=MU_WEIGHT):
    return float(np.sqrt((mu_weight * (mu_sim - mu_data)) ** 2 + (sigma_sim - sigma_data) ** 2))


# ============================================================================
# Mode: info
# ============================================================================

def do_info():
    n = total_grid_points()
    print(f"Reflectivity grid ({len(REFLECTIVITY_GRID)}): {REFLECTIVITY_GRID.tolist()}")
    print(f"H2oRefl grid      ({len(H2OREFL_GRID)}): {H2OREFL_GRID.tolist()}")
    print(f"Total grid points: {n}")
    if n > 1000:
        print(f"  SLURM part 1:  sbatch --array=0-999%8 scan_array.slurm")
        print(f"  SLURM part 2:  sbatch --array=0-{n - 1001}%8 --export=ALL,INDEX_OFFSET=1000 scan_array.slurm")
    else:
        print(f"  SLURM:         sbatch --array=0-{n - 1}%8 scan_array.slurm")
    print(f"Run numbers:       {RUN_NUMBER_OFFSET} .. {RUN_NUMBER_OFFSET + n - 1}")
    print(f"Results folder:    {RESULTS_DIR}   (scan tag: {SCAN_TAG})")
    print("Example command:   " + " ".join(geant4_command(RUN_NUMBER_OFFSET, *grid_point_for_index(0)[:2])))


# ============================================================================
# Mode: run
# ============================================================================

def do_run(index):
    reflectivity, h2orefl, i, j = grid_point_for_index(index)
    run_number = RUN_NUMBER_OFFSET + index
    sim_output_path = os.path.join(DATA_DIR, f"Sim_D2ODetector{run_number:03d}.root")

    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)

    print(f"Task {index}: grid[{i}][{j}] -> Reflectivity={reflectivity:.4f}, "
          f"H2oRefl={h2orefl:.4f}, run_number={run_number}")
    print(f"    primaries-to-generate: {N_PRIMARIES} (command line)")
    print(f"    PMT quality cut: >= {MIN_HITS_PER_PMT} PE on each of {N_PMTS} PMTs")
    print(f"    Random delay: {delay:.2f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"point_{index}.json")

    mu_data, sigma_data = get_real_data_mu_sigma()

    record = {
        "scan": SCAN_TAG,
        "index": index, "i": i, "j": j, "run_number": run_number,
        "reflectivity": reflectivity, "h2orefl": h2orefl,
        "mu_data": mu_data, "sigma_data": sigma_data,
        "n_primaries": N_PRIMARIES,
        "command": " ".join(geant4_command(run_number, reflectivity, h2orefl)),
        "pmt_quality_cut": f">= {MIN_HITS_PER_PMT} PE on each of {N_PMTS} PMTs",
    }

    try:
        t0 = time.time()
        ok = run_geant4(sim_output_path, run_number, reflectivity, h2orefl)
        elapsed = time.time() - t0
        print(f"    [Geant4] finished in {elapsed:.1f}s, success={ok}")
        record["geant4_seconds"] = elapsed

        if not ok:
            raise RuntimeError("Geant4 run failed")

        # Confirm the run really used this grid point's parameters
        used_refl, used_h2o = read_used_parameters(run_number)
        record.update({"used_reflectivity": used_refl, "used_h2orefl": used_h2o})
        if used_refl is None or used_h2o is None:
            print("    [check] could not read beamOnFiles copy; parameters not verified")
        elif abs(used_refl - reflectivity) > 1e-5 or abs(used_h2o - h2orefl) > 1e-5:
            raise RuntimeError(f"run used Reflectivity={used_refl}, H2oRefl={used_h2o}, "
                               f"expected {reflectivity}, {h2orefl}")
        else:
            print(f"    [check] run used Reflectivity={used_refl}, H2oRefl={used_h2o}  OK")

        mu_sim, sigma_sim, n_events_passed = load_simulation_statistics(sim_output_path)

        if mu_sim is None:
            raise RuntimeError("Simulation produced zero events above threshold")

        L = loss(mu_sim, sigma_sim, mu_data, sigma_data)
        record.update({
            "mu_sim": mu_sim,
            "sigma_sim": sigma_sim,
            "n_events_passed": n_events_passed,
            "loss": L,
            "status": "ok"
        })
        print(f"mu_sim={mu_sim:.2f} sigma_sim={sigma_sim:.2f} loss={L:.3f} (n_events={n_events_passed})")

    except Exception as e:
        record.update({"error": str(e), "status": "failed"})
        print(f"FAILED: {e}")

    # Delete ROOT file to save disk space
    if os.path.exists(sim_output_path):
        try:
            os.remove(sim_output_path)
            print(f"    [Deleted trial file: {os.path.basename(sim_output_path)}]")
        except OSError as e:
            print(f"    [Could not delete trial file: {e}]")

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {out_path}")


# ============================================================================
# Mode: collect
# ============================================================================

def do_collect():
    shape = (len(H2OREFL_GRID), len(REFLECTIVITY_GRID))
    loss_grid = np.full(shape, np.nan)
    mu_grid = np.full(shape, np.nan)
    sigma_grid = np.full(shape, np.nan)
    mu_data = sigma_data = None
    n_found = n_failed = n_stale = 0

    for path in glob.glob(os.path.join(RESULTS_DIR, "point_*.json")):
        with open(path) as f:
            rec = json.load(f)
        i, j = rec["i"], rec["j"]
        # Skip files left over from an earlier scan
        if (rec.get("scan") != SCAN_TAG
                or not (0 <= i < shape[0] and 0 <= j < shape[1])
                or rec.get("index") != i * shape[1] + j
                or abs(rec.get("reflectivity", -1) - REFLECTIVITY_GRID[j]) > 1e-6
                or abs(rec.get("h2orefl", -1) - H2OREFL_GRID[i]) > 1e-6):
            n_stale += 1
            continue
        if rec.get("status") == "ok":
            loss_grid[i, j] = rec["loss"]
            mu_grid[i, j] = rec["mu_sim"]
            sigma_grid[i, j] = rec["sigma_sim"]
            n_found += 1
        else:
            n_failed += 1
            print(f"  point {rec.get('index')} FAILED: {rec.get('error')}")
        if mu_data is None and "mu_data" in rec:
            mu_data, sigma_data = rec["mu_data"], rec["sigma_data"]

    if n_stale:
        print(f"Skipped {n_stale} result files from an older scan (scan tag or grid values don't match)")
    expected = total_grid_points()
    missing = expected - n_found - n_failed
    print(f"Loaded {n_found} ok + {n_failed} failed = {n_found + n_failed} / {expected}"
          + (f"  ({missing} not run yet)" if missing else ""))

    if np.all(np.isnan(loss_grid)):
        raise RuntimeError("No successful grid points to plot")

    best_flat_idx = np.nanargmin(loss_grid)
    bi, bj = np.unravel_index(best_flat_idx, loss_grid.shape)
    best_refl = REFLECTIVITY_GRID[bj]
    best_h2orefl = H2OREFL_GRID[bi]
    best_loss = loss_grid[bi, bj]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # --- loss surface
    ax = axes[0]
    im = ax.pcolormesh(REFLECTIVITY_GRID, H2OREFL_GRID, loss_grid, shading="nearest", cmap="cividis_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$L = \sqrt{%.0f(\mu_{sim}-\mu_{data})^2 + (\sigma_{sim}-\sigma_{data})^2}$" % (MU_WEIGHT ** 2))
    ax.plot(best_refl, best_h2orefl, marker="*", markersize=18,
            color="white", markeredgecolor="black", markeredgewidth=1.2, zorder=5)
    ax.annotate(
        f"Best-fit\nReflectivity={best_refl:.3f}\nH2oRefl={best_h2orefl:.3f}\nL={best_loss:.2f}",
        xy=(best_refl, best_h2orefl), xytext=(10, 10), textcoords="offset points",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
    )
    ax.set_title(f"Loss ({n_found}/{expected} points)")

    # --- simulated mean and width, with the data value as a contour
    for ax, grid, target, name in ((axes[1], mu_grid, mu_data, "μ_sim"),
                                   (axes[2], sigma_grid, sigma_data, "σ_sim")):
        im = ax.pcolormesh(REFLECTIVITY_GRID, H2OREFL_GRID, grid, shading="nearest", cmap="viridis")
        fig.colorbar(im, ax=ax).set_label(f"{name} (PE)")
        if target is not None and np.sum(np.isfinite(grid)) > 3:
            try:
                cs = ax.contour(REFLECTIVITY_GRID, H2OREFL_GRID, np.ma.masked_invalid(grid),
                                levels=[target], colors="white", linewidths=2)
                ax.clabel(cs, fmt={target: f"data {target:.1f}"}, fontsize=9)
            except Exception:
                pass
        ax.plot(best_refl, best_h2orefl, marker="*", markersize=14,
                color="white", markeredgecolor="black", zorder=5)
        ax.set_title(f"{name} (white line = data value)")

    for ax in axes:
        ax.set_xlabel("Reflectivity of Tyvek")
        ax.set_ylabel("H2O Attenuation Length Coefficient")

    title = "Michel Parameter Scan (wide grid)"
    if mu_data is not None:
        title += f"  —  PMT cut >= {MIN_HITS_PER_PMT} PE/PMT, μ_data={mu_data:.1f}, σ_data={sigma_data:.1f} PE"
    fig.suptitle(title, fontsize=13)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to: {PLOT_OUT}")
    print(f"Best fit: Reflectivity={best_refl:.4f}, H2oRefl={best_h2orefl:.4f}, Loss={best_loss:.4f}")

    edge = []
    if bj in (0, len(REFLECTIVITY_GRID) - 1):
        edge.append("Reflectivity")
    if bi in (0, len(H2OREFL_GRID) - 1):
        edge.append("H2oRefl")

    print("\n" + "=" * 60)
    print(f"SUMMARY - WIDE GRID SCAN ({len(REFLECTIVITY_GRID)} x {len(H2OREFL_GRID)})")
    print("=" * 60)
    print(f"Reflectivity range:       {REFLECTIVITY_GRID[0]:.2f} - {REFLECTIVITY_GRID[-1]:.2f}")
    print(f"H2oRefl range:            {H2OREFL_GRID[0]:.2f} - {H2OREFL_GRID[-1]:.2f}")
    print(f"PMT quality cut:          >= {MIN_HITS_PER_PMT} PE on each of {N_PMTS} PMTs")
    print(f"Total grid points:        {expected}")
    print(f"Successful:               {n_found}")
    print(f"Failed:                   {n_failed}")
    print(f"Best Reflectivity:        {best_refl:.4f}")
    print(f"Best H2oRefl:             {best_h2orefl:.4f}")
    print(f"Best Loss:                {best_loss:.4f}")
    if mu_data is not None:
        print(f"Data μ:                   {mu_data:.2f} PE   (sim at best: {mu_grid[bi, bj]:.2f})")
        print(f"Data σ:                   {sigma_data:.2f} PE   (sim at best: {sigma_grid[bi, bj]:.2f})")
    print(f"primaries per run:        {N_PRIMARIES}")
    if edge:
        print(f"WARNING: best fit is on the edge of the {' and '.join(edge)} range - extend the grid.")
    print("=" * 60)


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
