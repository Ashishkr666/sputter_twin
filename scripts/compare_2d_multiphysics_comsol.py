"""Comprehensive Benchmark: SputterTwin 2D Multiphysics vs. COMSOL Multiphysics.

Evaluates the new 2D Axisymmetric Multiphysics field engine against:
1. COMSOL Multiphysics Reference Benchmark (Ground Truth)
2. SputterTwin 0D Analytical Baseline (plasma.py)
3. SputterTwin 2D Multiphysics Engine (physics2d/plasma2d.py)

Shows the exact quantitative improvement and error reduction across 7 operating regimes.
"""

from __future__ import annotations

import os
import sys

import torch  # Ensure torch is imported before matplotlib on Windows
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sputtertwin.physics.plasma import calculate_discharge_state
from sputtertwin.physics2d import compute_magnetron_magnetic_field, simulate_plasma_2d


# Standard COMSOL Multiphysics DC Magnetron benchmark cases
COMSOL_BENCHMARK_CASES = [
    {
        "id": "Case 1",
        "desc": "Low P / Low W",
        "power_w": 100.0,
        "pressure_mtorr": 2.0,
        "ar_flow_sccm": 20.0,
        "voltage_v": 384.5,
        "current_a": 0.260,
        "ion_flux": 9.40e20,
        "electron_temp_ev": 3.75,
        "plasma_density_m3": 5.10e17,
    },
    {
        "id": "Case 2",
        "desc": "Low-Mid Power",
        "power_w": 150.0,
        "pressure_mtorr": 3.5,
        "ar_flow_sccm": 20.0,
        "voltage_v": 388.2,
        "current_a": 0.386,
        "ion_flux": 1.40e21,
        "electron_temp_ev": 3.28,
        "plasma_density_m3": 8.20e17,
    },
    {
        "id": "Case 3",
        "desc": "Nominal Baseline (Slide 7)",
        "power_w": 220.0,
        "pressure_mtorr": 5.0,
        "ar_flow_sccm": 20.0,
        "voltage_v": 395.0,
        "current_a": 0.557,
        "ion_flux": 1.98e21,
        "electron_temp_ev": 3.10,
        "plasma_density_m3": 1.21e18,
    },
    {
        "id": "Case 4",
        "desc": "Mid-High Power",
        "power_w": 300.0,
        "pressure_mtorr": 5.0,
        "ar_flow_sccm": 20.0,
        "voltage_v": 412.0,
        "current_a": 0.728,
        "ion_flux": 2.58e21,
        "electron_temp_ev": 3.05,
        "plasma_density_m3": 1.59e18,
    },
    {
        "id": "Case 5",
        "desc": "High P / High W",
        "power_w": 350.0,
        "pressure_mtorr": 8.0,
        "ar_flow_sccm": 20.0,
        "voltage_v": 382.0,
        "current_a": 0.916,
        "ion_flux": 3.25e21,
        "electron_temp_ev": 2.65,
        "plasma_density_m3": 2.10e18,
    },
    {
        "id": "Case 6",
        "desc": "Extreme Power",
        "power_w": 450.0,
        "pressure_mtorr": 10.0,
        "ar_flow_sccm": 20.0,
        "voltage_v": 376.5,
        "current_a": 1.195,
        "ion_flux": 4.25e21,
        "electron_temp_ev": 2.50,
        "plasma_density_m3": 2.82e18,
    },
    {
        "id": "Case 7",
        "desc": "Transition Regime",
        "power_w": 250.0,
        "pressure_mtorr": 15.0,
        "ar_flow_sccm": 20.0,
        "voltage_v": 328.0,
        "current_a": 0.762,
        "ion_flux": 2.70e21,
        "electron_temp_ev": 2.25,
        "plasma_density_m3": 1.85e18,
    },
]


def run_comparison() -> None:
    print("=" * 95)
    print("      SPUTTERTWIN 2D MULTIPHYSICS vs. COMSOL MULTIPHYSICS COMPARISON")
    print("=" * 95)

    print("\n[1/3] Computing 2D Axisymmetric Magnetic Field Baseline...")
    b2d = compute_magnetron_magnetic_field(grid_r_points=60, grid_z_points=50)
    print(f"  [OK] Magnetic field ready (Racetrack radius = {b2d.racetrack_radius_m*1000:.1f} mm, B_parallel = {b2d.b_parallel_peak_tesla*10000:.1f} G)")

    print("\n[2/3] Simulating 7 cases: COMSOL vs. 0D Analytical vs. 2D Multiphysics...")
    print("-" * 95)
    print(f"{'Case & Conditions':<22} | {'Metric':<8} | {'COMSOL':<10} | {'0D Analytical':<14} | {'2D Multiphysics':<16} | {'Improvement'}")
    print("-" * 95)

    v_err_0d, v_err_2d = [], []
    i_err_0d, i_err_2d = [], []

    results_data = []

    for c in COMSOL_BENCHMARK_CASES:
        w = c["power_w"]
        p = c["pressure_mtorr"]
        flow = c["ar_flow_sccm"]

        # 0D Analytical
        st_0d = calculate_discharge_state(power_w=w, pressure_mtorr=p, ar_flow_sccm=flow)

        # 2D Multiphysics
        st_2d = simulate_plasma_2d(
            power_w=w,
            pressure_mtorr=p,
            ar_flow_sccm=flow,
            mag_field=b2d,
        )

        # Error metrics
        e_v_0d = abs(st_0d.voltage_v - c["voltage_v"]) / c["voltage_v"] * 100.0
        e_v_2d = abs(st_2d.voltage_v - c["voltage_v"]) / c["voltage_v"] * 100.0
        v_err_0d.append(e_v_0d)
        v_err_2d.append(e_v_2d)

        e_i_0d = abs(st_0d.current_a - c["current_a"]) / c["current_a"] * 100.0
        e_i_2d = abs(st_2d.current_a - c["current_a"]) / c["current_a"] * 100.0
        i_err_0d.append(e_i_0d)
        i_err_2d.append(e_i_2d)

        case_label = f"{c['id']} ({w:.0f}W, {p:.1f}mT)"
        v_imp = f"{e_v_0d:.1f}% -> {e_v_2d:.1f}%"
        i_imp = f"{e_i_0d:.1f}% -> {e_i_2d:.1f}%"

        print(f"{case_label:<22} | {'V_d (V)':<8} | {c['voltage_v']:<10.1f} | {st_0d.voltage_v:<8.1f} ({e_v_0d:4.1f}%) | {st_2d.voltage_v:<8.1f} ({e_v_2d:4.1f}%)  | {v_imp}")
        print(f"{'':<22} | {'I_d (A)':<8} | {c['current_a']:<10.3f} | {st_0d.current_a:<8.3f} ({e_i_0d:4.1f}%) | {st_2d.current_a:<8.3f} ({e_i_2d:4.1f}%)  | {i_imp}")
        print("-" * 95)

        results_data.append({
            "case": c,
            "st_0d": st_0d,
            "st_2d": st_2d,
            "e_v_0d": e_v_0d,
            "e_v_2d": e_v_2d,
            "e_i_0d": e_i_0d,
            "e_i_2d": e_i_2d,
        })

    mean_v_0d, mean_v_2d = np.mean(v_err_0d), np.mean(v_err_2d)
    mean_i_0d, mean_i_2d = np.mean(i_err_0d), np.mean(i_err_2d)

    print("\nOVERALL ACCURACY IMPROVEMENT SUMMARY:")
    print(f"  * Discharge Voltage (V_d):  0D Error = {mean_v_0d:.2f}%  -->  2D Multiphysics Error = {mean_v_2d:.2f}%  (Reduced by {((mean_v_0d - mean_v_2d)/mean_v_0d)*100:.1f}%)")
    print(f"  * Discharge Current (I_d):  0D Error = {mean_i_0d:.2f}%  -->  2D Multiphysics Error = {mean_i_2d:.2f}%  (Reduced by {((mean_i_0d - mean_i_2d)/mean_i_0d)*100:.1f}%)")
    print(f"  * Max Voltage Outlier:      0D Max = {np.max(v_err_0d):.2f}%  -->  2D Max = {np.max(v_err_2d):.2f}%")
    print("=" * 95)

    # 3. Generate 4-panel comparison figure
    print("\n[3/3] Generating visual comparison plots...")
    out_dir = "plots/07_multiphysics_2d"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "2d_multiphysics_vs_comsol_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: I-V Curves
    powers_dense = np.linspace(80.0, 480.0, 40)
    p_fixed = 5.0
    v_0d = [calculate_discharge_state(w, p_fixed).voltage_v for w in powers_dense]
    i_0d = [calculate_discharge_state(w, p_fixed).current_a for w in powers_dense]
    v_2d = [simulate_plasma_2d(w, p_fixed, mag_field=b2d).voltage_v for w in powers_dense]
    i_2d = [simulate_plasma_2d(w, p_fixed, mag_field=b2d).current_a for w in powers_dense]

    axes[0, 0].plot(i_0d, v_0d, "b--", lw=2, label="0D Analytical Baseline")
    axes[0, 0].plot(i_2d, v_2d, "g-", lw=2.5, label="2D Multiphysics Engine")

    c_i_5mT = [c["current_a"] for c in COMSOL_BENCHMARK_CASES if c["pressure_mtorr"] == 5.0]
    c_v_5mT = [c["voltage_v"] for c in COMSOL_BENCHMARK_CASES if c["pressure_mtorr"] == 5.0]
    axes[0, 0].plot(c_i_5mT, c_v_5mT, "ro", markersize=9, label="COMSOL Ground Truth")
    axes[0, 0].set_xlabel("Discharge Current $I_d$ (A)", fontsize=11)
    axes[0, 0].set_ylabel("Discharge Voltage $V_d$ (V)", fontsize=11)
    axes[0, 0].set_title("1. Cathode I-V Curve ($P = 5.0$ mTorr)", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Voltage vs Pressure at 220 W
    pressures_dense = np.linspace(1.5, 18.0, 40)
    w_fixed = 220.0
    v_0d_p = [calculate_discharge_state(w_fixed, p).voltage_v for p in pressures_dense]
    v_2d_p = [simulate_plasma_2d(w_fixed, p, mag_field=b2d).voltage_v for p in pressures_dense]

    axes[0, 1].plot(pressures_dense, v_0d_p, "b--", lw=2, label="0D Analytical")
    axes[0, 1].plot(pressures_dense, v_2d_p, "g-", lw=2.5, label="2D Multiphysics")

    c_p = [c["pressure_mtorr"] for c in COMSOL_BENCHMARK_CASES]
    c_v_scaled = [c["voltage_v"] * ((220.0 / c["power_w"]) ** (1.0 / 7.0)) for c in COMSOL_BENCHMARK_CASES]
    axes[0, 1].plot(c_p, c_v_scaled, "ro", markersize=8, label="COMSOL (Scaled to 220W)")
    axes[0, 1].set_xlabel("Argon Pressure (mTorr)", fontsize=11)
    axes[0, 1].set_ylabel("Discharge Voltage $V_d$ (V)", fontsize=11)
    axes[0, 1].set_title("2. Voltage vs. Pressure ($W = 220$ W)", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Radial Cathode Ion Current Density Profile J_i(r)
    r_mm = b2d.r_grid_m * 1000.0
    sol_220w = simulate_plasma_2d(220.0, 5.0, mag_field=b2d)
    j_2d = sol_220w.current_density_1d

    # Gaussian 0D approximation for comparison
    j_0d = np.max(j_2d) * np.exp(-0.5 * ((r_mm - 25.0) / 5.0) ** 2)

    axes[1, 0].plot(r_mm, j_0d, "b--", lw=2, label="0D Gaussian Approximation")
    axes[1, 0].plot(r_mm, j_2d, "g-", lw=2.5, label="2D Self-Consistent Drift-Diffusion Profile")
    axes[1, 0].axvline(sol_220w.racetrack_radius_m * 1000.0, color="red", linestyle=":", lw=1.5, label=f"Racetrack Peak ({sol_220w.racetrack_radius_m*1000:.1f} mm)")
    axes[1, 0].set_xlabel("Cathode Radial Coordinate $r$ (mm)", fontsize=11)
    axes[1, 0].set_ylabel("Ion Current Density $J_i(r)$ (A/m$^2$)", fontsize=11)
    axes[1, 0].set_title("3. Self-Consistent Cathode Racetrack Erosion", fontsize=12, fontweight="bold")
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Error Reduction Bar Chart across all 7 cases
    case_names = [f"C{i}" for i in range(1, 8)]
    x_idx = np.arange(len(case_names))
    width = 0.35

    axes[1, 1].bar(x_idx - width/2, v_err_0d, width, label="0D Analytical Error (%)", color="royalblue", alpha=0.8)
    axes[1, 1].bar(x_idx + width/2, v_err_2d, width, label="2D Multiphysics Error (%)", color="forestgreen", alpha=0.8)
    axes[1, 1].set_xlabel("Benchmark Operating Case", fontsize=11)
    axes[1, 1].set_ylabel("Percentage Error vs. COMSOL (%)", fontsize=11)
    axes[1, 1].set_title("4. Error Reduction (0D Analytical vs. 2D Multiphysics)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x_idx)
    axes[1, 1].set_xticklabels(case_names)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("SputterTwin 2D Multiphysics Upgrade vs. COMSOL Multiphysics Benchmark", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"  [OK] Comparison figure saved to: {out_path}")
    print("=" * 95)


if __name__ == "__main__":
    run_comparison()
