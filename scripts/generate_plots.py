"""Comprehensive publication-quality plot generator for SputterTwin.

Generates full diagnostic visualization suites for all four physics modules:
1. Plasma Discharge Physics (plasma.py)
2. Sputter Yield Physics (sputter_yield.py)
3. Gas Kinetic Transport (transport.py)
4. Thin-Film Deposition & Wafer Profiles (deposition.py)
5. Integrated System Executive Dashboard
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sputtertwin.physics.deposition import simulate_deposition
from sputtertwin.physics.plasma import calculate_discharge_state
from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    calculate_sputter_yield,
    calculate_sputter_yield_array,
)
from sputtertwin.physics.transport import (
    calculate_knudsen_number,
    calculate_mean_free_path,
    calculate_scattering_broadening,
    calculate_transmission_probability,
    calculate_transport_summary,
)

# Professional visual style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def generate_plasma_plots(base_dir: str):
    """Generate diagnostic plots for plasma.py."""
    out_dir = os.path.join(base_dir, "01_plasma")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating 01_plasma plots...")

    # Plot 1: I-V Characteristics across pressures
    fig, ax = plt.subplots(figsize=(8, 6))
    voltages = np.linspace(250, 550, 200)
    pressures = [2.0, 5.0, 10.0, 20.0]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    # Empirical magnetron constant used in plasma.py
    n_exp = 6.0
    m_exp = 0.4
    k_const = 0.75 / ((5.0 ** m_exp) * (400.0 ** n_exp))

    for p, c in zip(pressures, colors):
        currents = k_const * (p ** m_exp) * (voltages ** n_exp)
        ax.plot(voltages, currents, label=f"P = {p:.1f} mTorr", color=c, lw=2.2)

    # Power hyperbolas (W = V * I)
    for w in [100, 200, 300, 400]:
        i_w = w / voltages
        ax.plot(voltages, i_w, "--", color="gray", alpha=0.5, lw=1)
        ax.text(voltages[-20], (w / voltages[-20]) + 0.03, f"{w} W", color="gray", fontsize=9)

    ax.set_xlim(280, 520)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("Cathode Discharge Voltage $V_d$ (V)")
    ax.set_ylabel("Discharge Current $I_d$ (A)")
    ax.set_title("DC Magnetron I–V Characteristics: $I_d = k \\cdot P^m \\cdot V_d^n$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plasma_01_iv_characteristics.png"), dpi=300)
    plt.close(fig)

    # Plot 2: Voltage and Current scaling vs Power
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    powers = np.linspace(50, 500, 50)
    for p, c in zip([3.0, 5.0, 10.0], ["#1f77b4", "#2ca02c", "#d62728"]):
        v_list = [calculate_discharge_state(w, p).voltage_v for w in powers]
        i_list = [calculate_discharge_state(w, p).current_a for w in powers]
        ax1.plot(powers, v_list, label=f"{p:.0f} mTorr", color=c, lw=2)
        ax2.plot(powers, i_list, label=f"{p:.0f} mTorr", color=c, lw=2)

    ax1.set_xlabel("Cathode Power (W)")
    ax1.set_ylabel("Discharge Voltage (V)")
    ax1.set_title("Discharge Voltage vs. Power")
    ax1.legend()

    ax2.set_xlabel("Cathode Power (W)")
    ax2.set_ylabel("Discharge Current (A)")
    ax2.set_title("Discharge Current vs. Power")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plasma_02_voltage_current_scaling.png"), dpi=300)
    plt.close(fig)

    # Plot 3: Plasma Density & Electron Temperature vs Pressure
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    pressures_dense = np.linspace(1.0, 25.0, 60)
    states = [calculate_discharge_state(220.0, p) for p in pressures_dense]
    densities = [s.plasma_density_m3 for s in states]
    temperatures = [s.electron_temp_ev for s in states]

    color1 = "#1f77b4"
    ax1.set_xlabel("Argon Set Pressure (mTorr)")
    ax1.set_ylabel("Plasma Density $n_e$ ($m^{-3}$)", color=color1)
    line1 = ax1.plot(pressures_dense, densities, color=color1, lw=2.2, label="$n_e$ (Density)")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Electron Temperature $T_e$ (eV)", color=color2)
    line2 = ax2.plot(pressures_dense, temperatures, color=color2, lw=2.2, linestyle="--", label="$T_e$ (Temp)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(1.5, 4.5)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    ax1.set_title("Plasma Sheath Parameters vs. Pressure (220 W)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plasma_03_density_and_temperature.png"), dpi=300)
    plt.close(fig)

    # Plot 4: Ar Flow Rate Coupling (Option B model)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    flows = np.linspace(5, 50, 40)
    p_effs = [calculate_discharge_state(220.0, 5.0, ar_flow_sccm=f).effective_pressure_mtorr for f in flows]
    v_flows = [calculate_discharge_state(220.0, 5.0, ar_flow_sccm=f).voltage_v for f in flows]

    ax1.plot(flows, p_effs, "g.-", lw=2)
    ax1.axvline(20, color="gray", linestyle="--", alpha=0.7, label="Baseline (20 sccm)")
    ax1.set_xlabel("Ar Gas Flow (sccm)")
    ax1.set_ylabel("Effective Pressure $P_{eff}$ (mTorr)")
    ax1.set_title("Effective Chamber Pressure vs. Ar Flow")
    ax1.legend()

    ax2.plot(flows, v_flows, "b.-", lw=2)
    ax2.axvline(20, color="gray", linestyle="--", alpha=0.7, label="Baseline (20 sccm)")
    ax2.set_xlabel("Ar Gas Flow (sccm)")
    ax2.set_ylabel("Discharge Voltage (V)")
    ax2.set_title("Discharge Voltage Shift via Flow Coupling")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plasma_04_argon_flow_coupling.png"), dpi=300)
    plt.close(fig)


def generate_sputter_yield_plots(base_dir: str):
    """Generate diagnostic plots for sputter_yield.py."""
    out_dir = os.path.join(base_dir, "02_sputter_yield")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating 02_sputter_yield plots...")

    # Plot 1: Energy dependence with native vectorization
    fig, ax = plt.subplots(figsize=(8.5, 6))
    energies = np.linspace(5, 1000, 300)
    mat_colors = {"Cu": "#d95f02", "Al": "#7570b3", "Ti": "#1b9e77"}

    for mat_name in ["Cu", "Al", "Ti"]:
        y_vals = calculate_sputter_yield_array(energies, material=mat_name, angle_rad=0.0)
        eth = MATERIALS[mat_name].threshold_energy
        ax.plot(energies, y_vals, label=f"{mat_name} (Eth = {eth:.0f} eV)", color=mat_colors[mat_name], lw=2.2)
        ax.axvline(eth, color=mat_colors[mat_name], linestyle=":", alpha=0.6)

    # Shaded literature benchmark zones
    ax.axvspan(380, 420, alpha=0.15, color="gray", label="Nominal Sheath Energy (400 eV)")
    ax.text(450, 2.1, "Cu: Highest yield\n(Low binding energy $U_s = 3.49$ eV)", color="#d95f02", fontsize=10)
    ax.text(450, 0.45, "Ti: Lowest yield\n(High binding energy $U_s = 4.89$ eV)", color="#1b9e77", fontsize=10)

    ax.set_xlabel("Incident $Ar^+$ Kinetic Energy (eV)")
    ax.set_ylabel("Sputter Yield $Y(E)$ (atoms / incident ion)")
    ax.set_title("Energy-Dependent Sputter Yield: Yamamura-Tawara Formulation")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yield_01_energy_dependence_materials.png"), dpi=300)
    plt.close(fig)

    # Plot 2: Angular distribution Y(theta)/Y(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    angles_deg = np.linspace(0, 89.9, 180)
    angles_rad = np.radians(angles_deg)

    for energy in [200.0, 400.0, 800.0]:
        y_normal = calculate_sputter_yield(energy, "Cu", angle_rad=0.0)
        y_angular = [calculate_sputter_yield(energy, "Cu", angle_rad=th) / y_normal for th in angles_rad]
        ax.plot(angles_deg, y_angular, label=f"E = {energy:.0f} eV", lw=2.2)

    ax.axvline(65.0, color="red", linestyle="--", alpha=0.7, label="Optimum Angle $\\theta_{opt} = 65^\\circ$")
    ax.axvline(85.0, color="black", linestyle=":", alpha=0.7, label="Grazing Cutoff ($85^\\circ$)")
    ax.set_xlabel("Incidence Angle $\\theta$ from Surface Normal (degrees)")
    ax.set_ylabel("Normalized Sputter Yield $Y(\\theta) / Y(0)$")
    ax.set_title("Angular Enhancement in Sputter Yield (Copper Target)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yield_02_angular_distribution_yamamura.png"), dpi=300)
    plt.close(fig)

    # Plot 3: Material properties & yield correlation bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    materials = ["Ti", "Al", "Cu"]
    yields_400 = [calculate_sputter_yield(400.0, m) for m in materials]
    us_vals = [MATERIALS[m].sublimation_energy for m in materials]

    bars1 = ax1.bar(materials, yields_400, color=["#1b9e77", "#7570b3", "#d95f02"], width=0.55)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Yield at 400 eV (atoms / ion)")
    ax1.set_title("Comparative Yield at 400 eV")
    ax1.set_ylim(0, 2.8)

    bars2 = ax2.bar(materials, us_vals, color=["#1b9e77", "#7570b3", "#d95f02"], width=0.55)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f} eV", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Sublimation / Binding Energy $U_s$ (eV)")
    ax2.set_title("Surface Binding Energy $U_s$ (Inverse Sputterability)")
    ax2.set_ylim(0, 6.0)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yield_03_yield_vs_binding_energy.png"), dpi=300)
    plt.close(fig)


def generate_transport_plots(base_dir: str):
    """Generate diagnostic plots for transport.py."""
    out_dir = os.path.join(base_dir, "03_transport")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating 03_transport plots...")

    # Plot 1: Mean Free Path & Transport Regimes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    pressures = np.linspace(1, 50, 100)
    mfps_mm = [calculate_mean_free_path(p) * 1e3 for p in pressures]
    kns = [calculate_knudsen_number(calculate_mean_free_path(p), 0.08) for p in pressures]

    ax1.plot(pressures, mfps_mm, color="#1f77b4", lw=2.2)
    ax1.set_xlabel("Argon Gas Pressure (mTorr)")
    ax1.set_ylabel("Mean Free Path $\\lambda_{mfp}$ (mm)")
    ax1.set_title("Particle Mean Free Path ($P \\cdot \\lambda = \\mathrm{const}$)")

    ax2.plot(pressures, kns, color="black", lw=2.2)
    ax2.set_yscale("log")
    ax2.axhspan(1.0, 10.0, alpha=0.2, color="green", label="Ballistic ($Kn > 1.0$)")
    ax2.axhspan(0.1, 1.0, alpha=0.25, color="orange", label="Transition ($0.1 \\leq Kn \\leq 1.0$)")
    ax2.axhspan(0.01, 0.1, alpha=0.2, color="red", label="Continuum / Diffusive ($Kn < 0.1$)")
    ax2.axvline(5.0, color="blue", linestyle="--", label="Nominal (5 mTorr)")
    ax2.set_xlabel("Argon Gas Pressure (mTorr)")
    ax2.set_ylabel("Knudsen Number $Kn = \\lambda / d$ (d = 80 mm)")
    ax2.set_title("Collisional Transport Regimes")
    ax2.set_ylim(0.02, 5.0)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transport_01_mfp_and_knudsen_regimes.png"), dpi=300)
    plt.close(fig)

    # Plot 2: 2D Transmission Contour Map (Distance vs Pressure)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    p_grid = np.linspace(1, 20, 60)
    d_grid = np.linspace(40, 140, 60)
    P_mesh, D_mesh = np.meshgrid(p_grid, d_grid)
    trans_prob = np.zeros_like(P_mesh)

    for i in range(P_mesh.shape[0]):
        for j in range(P_mesh.shape[1]):
            mfp = calculate_mean_free_path(P_mesh[i, j])
            trans_prob[i, j] = calculate_transmission_probability(D_mesh[i, j] * 1e-3, mfp) * 100.0

    cp = ax.contourf(P_mesh, D_mesh, trans_prob, levels=20, cmap="viridis")
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label("Ballistic Transmission Probability (%)")
    ax.contour(P_mesh, D_mesh, trans_prob, levels=[1, 5, 10, 20, 50], colors="white", alpha=0.6)
    ax.plot(5.0, 80.0, "r*", markersize=14, label="Nominal Point (5 mTorr, 80 mm)")
    ax.set_xlabel("Pressure (mTorr)")
    ax.set_ylabel("Target-to-Substrate Distance (mm)")
    ax.set_title("Ballistic Particle Transmission: $P_{\\mathrm{bal}} = \\exp(-d / \\lambda)$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transport_02_transmission_contour.png"), dpi=300)
    plt.close(fig)

    # Plot 3: Scattering Broadening & Thermalization Efficiency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    p_range = np.linspace(1, 25, 50)
    d_m = 0.08
    broadenings = [calculate_scattering_broadening(p, d_m) for p in p_range]
    n_therms = [1.0 / (1.0 + (d_m / calculate_mean_free_path(p)) / 10.0) for p in p_range]

    ax1.plot(p_range, broadenings, "m.-", lw=2)
    ax1.set_xlabel("Pressure (mTorr)")
    ax1.set_ylabel("Scattering Broadening Factor $BF$")
    ax1.set_title("Angular/Spatial Broadening Factor: $\\sqrt{1 + d/\\lambda}$")

    ax2.plot(p_range, n_therms, "c.-", lw=2)
    ax2.set_xlabel("Pressure (mTorr)")
    ax2.set_ylabel("Thermalization Transport Efficiency $\\eta_{\\mathrm{transport}}$")
    ax2.set_title("Gas Phase Transport Efficiency to Substrate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transport_03_broadening_and_thermalization.png"), dpi=300)
    plt.close(fig)


def generate_deposition_plots(base_dir: str):
    """Generate diagnostic plots for deposition.py."""
    out_dir = os.path.join(base_dir, "04_deposition")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating 04_deposition plots...")

    # Plot 1: 2D Wafer Thickness Heatmap (Slide 7 Benchmark)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    res = simulate_deposition(
        power_w=220, pressure_mtorr=5.0, deposition_time_s=720,
        distance_mm=80, grid_size=61, material="Ti"
    )
    cmap = plt.get_cmap("RdYlBu_r")
    norm = TwoSlopeNorm(
        vmin=np.nanmin(res.thickness_map),
        vcenter=res.mean_thickness_nm,
        vmax=np.nanmax(res.thickness_map)
    )
    X, Y = np.meshgrid(res.x_grid_mm, res.y_grid_mm)
    im = ax.pcolormesh(X, Y, res.thickness_map, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(im, ax=ax, label="Film Thickness (nm)")
    ax.contour(X, Y, res.thickness_map, colors="k", alpha=0.35, levels=8)

    # Draw wafer perimeter
    circle = plt.Circle((0, 0), 75.0, color="black", fill=False, lw=2, linestyle="--", label="150 mm Wafer Boundary")
    ax.add_patch(circle)

    ax.set_title(
        f"2D Film Thickness Distribution (Ti Benchmark)\n"
        f"Mean: {res.mean_thickness_nm:.1f} nm | Rate: {res.deposition_rate_nm_min:.2f} nm/min | NU: {res.uniformity_percent:.2f}%"
    )
    ax.set_xlabel("Wafer X Position (mm)")
    ax.set_ylabel("Wafer Y Position (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deposition_01_wafer_thickness_2d_nominal.png"), dpi=300)
    plt.close(fig)

    # Plot 2: Radial Profile Slices (Racetrack & Target Erosion Effect)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    r_wafer = np.linspace(0, 75, 50)
    w_rad = 75.0

    # Variation of racetrack radius
    for r_race in [15.0, 25.0, 35.0]:
        r_res = simulate_deposition(220, 5.0, distance_mm=80, racetrack_radius_mm=r_race, material="Ti", grid_size=15)
        # Profile calculation
        bf = calculate_scattering_broadening(5.0, 0.08)
        c_eff = min(1.4922 * (r_race / 80.0) ** 2 / (bf ** 2), 0.95)
        mean_r2 = 0.5
        t0 = r_res.mean_thickness_nm / (1.0 - c_eff * mean_r2)
        h_slice = t0 * (1.0 - c_eff * (r_wafer / w_rad) ** 2)
        ax1.plot(r_wafer, h_slice, lw=2.2, label=f"$r_{{race}} = {r_race:.0f}$ mm (NU: {r_res.uniformity_percent:.2f}%)")

    ax1.set_xlabel("Radial Distance from Wafer Center $r$ (mm)")
    ax1.set_ylabel("Film Thickness $h(r)$ (nm)")
    ax1.set_title("Radial Thickness: Racetrack Radius Effect")
    ax1.legend()

    # Variation of target erosion
    for erosion in [0.0, 2.0, 5.0]:
        r_res = simulate_deposition(220, 5.0, distance_mm=80, target_erosion_mm=erosion, material="Ti", grid_size=15)
        bf = calculate_scattering_broadening(5.0, 0.08)
        c_eff = min(1.4922 * (25.0 / 80.0) ** 2 / (bf ** 2) * (1.0 + 0.10 * erosion), 0.95)
        mean_r2 = 0.5
        t0 = r_res.mean_thickness_nm / (1.0 - c_eff * mean_r2)
        h_slice = t0 * (1.0 - c_eff * (r_wafer / w_rad) ** 2)
        ax2.plot(r_wafer, h_slice, lw=2.2, label=f"Erosion = {erosion:.0f} mm (NU: {r_res.uniformity_percent:.2f}%)")

    ax2.set_xlabel("Radial Distance from Wafer Center $r$ (mm)")
    ax2.set_ylabel("Film Thickness $h(r)$ (nm)")
    ax2.set_title("Radial Thickness: Target Erosion Collimation")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deposition_02_radial_profile_racetrack_erosion.png"), dpi=300)
    plt.close(fig)

    # Plot 3: Pressure Sweep vs Rate and Uniformity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    pressures = np.linspace(1, 25, 30)
    rates_ti, nus_ti = [], []
    rates_cu, nus_cu = [], []

    for p in pressures:
        r_ti = simulate_deposition(220, p, distance_mm=80, material="Ti", grid_size=11)
        r_cu = simulate_deposition(220, p, distance_mm=80, material="Cu", grid_size=11)
        rates_ti.append(r_ti.deposition_rate_nm_min)
        nus_ti.append(r_ti.uniformity_percent)
        rates_cu.append(r_cu.deposition_rate_nm_min)
        nus_cu.append(r_cu.uniformity_percent)

    ax1.plot(pressures, rates_cu, "o-", color="#d95f02", lw=2, label="Copper (Cu)")
    ax1.plot(pressures, rates_ti, "s-", color="#1b9e77", lw=2, label="Titanium (Ti)")
    ax1.set_xlabel("Pressure (mTorr)")
    ax1.set_ylabel("Deposition Rate (nm/min)")
    ax1.set_title("Deposition Rate vs. Pressure (Transport Attenuation)")
    ax1.legend()

    ax2.plot(pressures, nus_ti, "s-", color="#2ca02c", lw=2, label="Non-Uniformity %")
    ax2.set_xlabel("Pressure (mTorr)")
    ax2.set_ylabel("Thickness Non-Uniformity (%)")
    ax2.set_title("Uniformity vs. Pressure (Collisional Broadening)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deposition_03_rate_and_uniformity_vs_pressure.png"), dpi=300)
    plt.close(fig)

    # Plot 4: Operating Space Contours (Power vs Distance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    powers = np.linspace(100, 450, 16)
    distances = np.linspace(50, 130, 16)
    P_grid, D_grid = np.meshgrid(powers, distances)
    R_map = np.zeros_like(P_grid)
    U_map = np.zeros_like(P_grid)

    for i in range(P_grid.shape[0]):
        for j in range(P_grid.shape[1]):
            res = simulate_deposition(P_grid[i, j], 5.0, distance_mm=D_grid[i, j], material="Cu", grid_size=7)
            R_map[i, j] = res.deposition_rate_nm_min
            U_map[i, j] = res.uniformity_percent

    cp1 = ax1.contourf(P_grid, D_grid, R_map, levels=20, cmap="viridis")
    fig.colorbar(cp1, ax=ax1, label="Growth Rate (nm/min)")
    ax1.set_xlabel("Cathode Power (W)")
    ax1.set_ylabel("Throw Distance (mm)")
    ax1.set_title("Cu Deposition Rate Operating Window")

    cp2 = ax2.contourf(P_grid, D_grid, U_map, levels=20, cmap="plasma")
    fig.colorbar(cp2, ax=ax2, label="Non-Uniformity (%)")
    ax2.set_xlabel("Cathode Power (W)")
    ax2.set_ylabel("Throw Distance (mm)")
    ax2.set_title("Cu Film Uniformity Operating Window")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deposition_04_operating_window_contours.png"), dpi=300)
    plt.close(fig)

    # Plot 5: Time scaling linearity
    fig, ax = plt.subplots(figsize=(8, 5.5))
    times_s = np.linspace(60, 1800, 20)
    thick_cu = [simulate_deposition(220, 5.0, deposition_time_s=t, material="Cu", grid_size=5).mean_thickness_nm for t in times_s]
    thick_ti = [simulate_deposition(220, 5.0, deposition_time_s=t, material="Ti", grid_size=5).mean_thickness_nm for t in times_s]

    ax.plot(times_s / 60.0, thick_cu, "o-", color="#d95f02", lw=2.2, label="Copper (Cu)")
    ax.plot(times_s / 60.0, thick_ti, "s-", color="#1b9e77", lw=2.2, label="Titanium (Ti)")
    ax.set_xlabel("Deposition Time (minutes)")
    ax.set_ylabel("Mean Deposited Thickness (nm)")
    ax.set_title("Film Thickness Evolution vs. Deposition Time (Strict Linear Scaling)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deposition_05_time_scaling_linearity.png"), dpi=300)
    plt.close(fig)

    # Plot 6: Direct Comparison of Cu vs Ti Wafer Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    res_cu = simulate_deposition(220, 5.0, deposition_time_s=720, material="Cu", grid_size=45)
    res_ti = simulate_deposition(220, 5.0, deposition_time_s=720, material="Ti", grid_size=45)

    im1 = ax1.pcolormesh(X_grid := np.meshgrid(res_cu.x_grid_mm, res_cu.y_grid_mm)[0],
                         Y_grid := np.meshgrid(res_cu.x_grid_mm, res_cu.y_grid_mm)[1],
                         res_cu.thickness_map, cmap="inferno", shading="auto")
    fig.colorbar(im1, ax=ax1, label="Thickness (nm)")
    ax1.set_title(f"Copper (Cu) Target\nMean: {res_cu.mean_thickness_nm:.1f} nm, Rate: {res_cu.deposition_rate_nm_min:.1f} nm/min")
    ax1.set_aspect("equal")

    im2 = ax2.pcolormesh(X_grid, Y_grid, res_ti.thickness_map, cmap="inferno", shading="auto")
    fig.colorbar(im2, ax=ax2, label="Thickness (nm)")
    ax2.set_title(f"Titanium (Ti) Target\nMean: {res_ti.mean_thickness_nm:.1f} nm, Rate: {res_ti.deposition_rate_nm_min:.1f} nm/min")
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deposition_06_cu_vs_ti_comparison.png"), dpi=300)
    plt.close(fig)


def generate_system_dashboard(base_dir: str):
    """Generate integrated 4-panel executive dashboard connecting all physics stages."""
    out_dir = os.path.join(base_dir, "05_system_dashboard")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating 05_system_dashboard...")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SputterTwin Physics-Informed Digital Twin: End-to-End Simulation Pipeline", fontsize=16, fontweight="bold")

    # Panel 1: Plasma Discharge
    voltages = np.linspace(250, 550, 100)
    for p, c in zip([2.0, 5.0, 10.0], ["#1f77b4", "#2ca02c", "#d62728"]):
        k = 0.75 / ((5.0 ** 0.4) * (400.0 ** 6.0))
        axs[0, 0].plot(voltages, k * (p ** 0.4) * (voltages ** 6.0), color=c, lw=2, label=f"P = {p:.0f} mTorr")
    axs[0, 0].plot(398.0, 0.553, "k*", markersize=14, label="Nominal Discharge (220 W, 5 mTorr)")
    axs[0, 0].set_ylim(0, 1.2)
    axs[0, 0].set_xlabel("Discharge Voltage $V_d$ (V)")
    axs[0, 0].set_ylabel("Discharge Current $I_d$ (A)")
    axs[0, 0].set_title("1. Plasma Discharge ($I_d = k P^m V_d^n$)")
    axs[0, 0].legend()

    # Panel 2: Sputter Yield
    energies = np.linspace(10, 800, 100)
    axs[0, 1].plot(energies, calculate_sputter_yield_array(energies, "Cu"), color="#d95f02", lw=2.2, label="Copper (Cu)")
    axs[0, 1].plot(energies, calculate_sputter_yield_array(energies, "Ti"), color="#1b9e77", lw=2.2, label="Titanium (Ti)")
    axs[0, 1].plot(398.0, calculate_sputter_yield(398.0, "Cu"), "o", color="#d95f02", markersize=8, label="Cu @ 398 eV")
    axs[0, 1].plot(398.0, calculate_sputter_yield(398.0, "Ti"), "s", color="#1b9e77", markersize=8, label="Ti @ 398 eV")
    axs[0, 1].set_xlabel("Ion Kinetic Energy (eV)")
    axs[0, 1].set_ylabel("Sputter Yield (atoms / ion)")
    axs[0, 1].set_title("2. Sputter Yield (Yamamura-Tawara)")
    axs[0, 1].legend()

    # Panel 3: Gas Kinetic Transport
    pressures = np.linspace(1, 20, 50)
    trans = [calculate_transmission_probability(0.08, calculate_mean_free_path(p)) * 100.0 for p in pressures]
    axs[1, 0].plot(pressures, trans, "b.-", lw=2, label="Ballistic Transmission %")
    axs[1, 0].plot(5.0, calculate_transmission_probability(0.08, calculate_mean_free_path(5.0)) * 100.0, "ro", markersize=10, label="Nominal (5 mTorr)")
    axs[1, 0].set_xlabel("Argon Pressure (mTorr)")
    axs[1, 0].set_ylabel("Ballistic Transmission Probability (%)")
    axs[1, 0].set_title("3. Gas Transport ($P_{\\mathrm{bal}} = \\exp(-d/\\lambda)$)")
    axs[1, 0].legend()

    # Panel 4: Wafer Thickness Map
    res = simulate_deposition(220, 5.0, deposition_time_s=720, material="Cu", grid_size=41)
    X, Y = np.meshgrid(res.x_grid_mm, res.y_grid_mm)
    im = axs[1, 1].pcolormesh(X, Y, res.thickness_map, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=axs[1, 1], label="Thickness (nm)")
    circle = plt.Circle((0, 0), 75.0, color="red", fill=False, lw=1.5, linestyle="--")
    axs[1, 1].add_patch(circle)
    axs[1, 1].set_aspect("equal")
    axs[1, 1].set_xlabel("Wafer X (mm)")
    axs[1, 1].set_ylabel("Wafer Y (mm)")
    axs[1, 1].set_title(f"4. Deposition Profile (Cu: {res.mean_thickness_nm:.1f} nm, NU: {res.uniformity_percent:.2f}%)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sputtertwin_complete_physics_dashboard.png"), dpi=300)
    plt.close(fig)


def main():
    plots_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plots_root, exist_ok=True)
    print(f"Starting comprehensive diagnostic plot generation in: {plots_root}")

    generate_plasma_plots(plots_root)
    generate_sputter_yield_plots(plots_root)
    generate_transport_plots(plots_root)
    generate_deposition_plots(plots_root)
    generate_system_dashboard(plots_root)

    print(f"All 17 comprehensive diagnostic plots generated successfully in: {plots_root}")


if __name__ == "__main__":
    main()
