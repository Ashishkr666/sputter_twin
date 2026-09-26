"""Academic Publication Figure: Stage 1 & Stage 2 Full Integration Dashboard.

Generates a 6-panel, journal-grade dashboard (300 DPI, pure white background)
demonstrating the complete physical and neural coupling between:
- Stage 1: Plasma discharge, magnetic confinement, cathode sheath, and gas rarefaction.
- Stage 2: SOTA sputter yield, Thomson cascade atom emission, and dynamic racetrack erosion.
"""

import os
import sys
import math

# Windows DLL safety: torch first
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sputtertwin import (
    simulate_integrated_discharge_and_erosion,
    calculate_sputter_yield_sota,
    calculate_sputter_yield,
    calculate_thomson_energy_spectrum,
    MATERIALS,
)
from sputtertwin.data import get_eckstein_data


def set_academic_style():
    """Apply rigorous IEEE/AIP peer-reviewed academic white journal styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12.5,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "grid.color": "#e0e0e0",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
    })


def main():
    set_academic_style()
    print("Executing End-to-End Integrated Multiphysics Simulation (Cu Target, 220W, 5 mTorr, 10h)...")

    res = simulate_integrated_discharge_and_erosion(
        power_w=220.0,
        pressure_mtorr=5.0,
        material="Cu",
        target_radius_mm=25.0,
        initial_thickness_mm=6.0,
        total_hours=10.0,
        time_step_hours=0.5,
        roughness_factor=0.15,
        grid_r=50,
        grid_z=40,
        num_ejected_samples=15000,
        seed=42,
    )

    print(res.summary())

    # Create 6-panel academic dashboard
    fig = plt.figure(figsize=(14.5, 9.5), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    # -------------------------------------------------------------------------
    # Panel (a): Stage 1 2D Plasma Potential & Cathode Sheath Drop V(r, z)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    R_mm = res.plasma.r_grid_m * 1e3
    Z_mm = res.plasma.z_grid_m * 1e3
    RR, ZZ = np.meshgrid(R_mm, Z_mm)
    
    # In plasma2d, V_2d shape is (grid_z, grid_r)
    v_levels = np.linspace(np.min(res.plasma.V_2d), np.max(res.plasma.V_2d), 20)
    cf1 = ax1.contourf(RR, ZZ, res.plasma.V_2d, levels=v_levels, cmap="viridis", alpha=0.9)
    cs1 = ax1.contour(RR, ZZ, res.plasma.V_2d, levels=10, colors="#222222", linewidths=0.6, alpha=0.6)
    ax1.clabel(cs1, inline=True, fontsize=7.5, fmt="%d V")
    
    cb1 = fig.colorbar(cf1, ax=ax1, pad=0.03, aspect=20)
    cb1.set_label(r"Plasma Potential $V(r, z)$ [V]", fontsize=9.5)
    
    # Draw racetrack marker on target cathode (z = 0)
    r_race_mm = res.plasma.racetrack_radius_m * 1e3
    ax1.scatter([r_race_mm], [0.0], color="#d62728", s=45, marker="^", zorder=5, label=f"Racetrack ({r_race_mm:.1f} mm)")
    ax1.axhline(0.0, color="#000000", lw=2.0)
    ax1.text(2.0, 1.5, "Cathode Target ($z=0$)", fontsize=8, color="#000000", weight="bold")
    ax1.set_xlabel(r"Radial Position $r$ [mm]")
    ax1.set_ylabel(r"Axial Distance $z$ [mm]")
    ax1.set_title(r"(a) Stage 1: 2D Plasma Potential & Sheath", loc="left", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)

    # -------------------------------------------------------------------------
    # Panel (b): Stage 1 Cathode Ion Current Density Bombardment J_i(r)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    j_i = res.plasma.current_density_1d
    ax2.plot(R_mm, j_i, color="#1f77b4", lw=2.2, label=r"Cathode Ion Flux $J_i(r)$")
    ax2.fill_between(R_mm, 0, j_i, color="#1f77b4", alpha=0.15)
    
    # Secondary axis: Gas Rarefaction T_gas(r, 0)
    if res.rarefaction is not None:
        ax2_twin = ax2.twinx()
        t_gas_cathode = res.rarefaction.T_gas_2d_K[0, :]
        ax2_twin.plot(R_mm, t_gas_cathode, color="#e6550d", lw=1.8, linestyle="--", label=r"Gas Temp $T_{\mathrm{gas}}$")
        ax2_twin.set_ylabel(r"Gas Temperature $T_{\mathrm{gas}}$ [K]", color="#e6550d", fontsize=9.5)
        ax2_twin.tick_params(axis="y", labelcolor="#e6550d")
        ax2_twin.set_ylim(290, 450)

    ax2.axvline(r_race_mm, color="#d62728", linestyle=":", lw=1.5, label=f"Peak $r = {r_race_mm:.1f}$ mm")
    ax2.set_xlabel(r"Target Radius $r$ [mm]")
    ax2.set_ylabel(r"Ion Current Density $J_i$ [$\mathrm{A/m^2}$]")
    ax2.set_title(r"(b) Stage 1: Ion Bombardment & Rarefaction", loc="left", fontweight="bold")
    ax2.set_ylim(0, np.max(j_i) * 1.15)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True)

    # -------------------------------------------------------------------------
    # Panel (c): Stage 2 SOTA Sputter Yield vs Angle Y(theta) & Ruzic Roughness
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    v_d = res.plasma.voltage_v
    angles_deg = np.linspace(0, 85, 150)
    angles_rad = np.radians(angles_deg)

    y_yamamura = [calculate_sputter_yield(v_d, "Cu", th) for th in angles_rad]
    y_sota = [calculate_sputter_yield_sota(v_d, "Cu", th, roughness_factor=0.15) for th in angles_rad]
    y_rough_high = [calculate_sputter_yield_sota(v_d, "Cu", th, roughness_factor=0.30) for th in angles_rad]

    ax3.plot(angles_deg, y_yamamura, "k--", lw=1.6, label="Ideal Smooth (Yamamura)")
    ax3.plot(angles_deg, y_sota, color="#2ca02c", lw=2.2, label=r"SOTA Ruzic ($r=0.15$)")
    ax3.plot(angles_deg, y_rough_high, color="#98df8a", lw=1.6, linestyle="-.", label=r"High Roughness ($r=0.30$)")

    # Overlay Eckstein experimental angular data for Cu at ~400 eV
    eck_data = get_eckstein_data("Cu")
    mask_400 = np.isclose(eck_data["energy_ev"], 400.0, atol=80.0)
    if np.any(mask_400):
        ax3.scatter(
            eck_data["angle_deg"][mask_400],
            eck_data["yield_atoms_per_ion"][mask_400],
            color="#d62728",
            edgecolor="#000000",
            s=45,
            zorder=6,
            label="Eckstein Experiment (400 eV)",
        )

    ax3.set_xlabel(r"Incidence Angle $\theta$ [deg]")
    ax3.set_ylabel(r"Sputter Yield $Y$ [atoms/ion]")
    ax3.set_title(r"(c) Stage 2: SOTA Sputter Yield vs Angle", loc="left", fontweight="bold")
    ax3.set_xlim(0, 85)
    ax3.legend(loc="upper left", framealpha=0.9)
    ax3.grid(True)

    # -------------------------------------------------------------------------
    # Panel (d): Stage 2 Nascent Sputtered Atom Thomson Kinetic Energy Cascade
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ejected_e = res.ejected_particles.energies
    
    # Histogram of Monte Carlo sampled nascent energies
    counts, bin_edges, _ = ax4.hist(
        ejected_e[ejected_e <= 30.0],
        bins=60,
        density=True,
        color="#3182bd",
        alpha=0.65,
        edgecolor="#08519c",
        lw=0.7,
        label=r"Sampled Nascent Atoms ($N=15{,}000$)",
    )

    # Theoretical Thomson spectrum curve f(E)
    e_curve = np.linspace(0.01, 30.0, 300)
    u_s = MATERIALS["Cu"].sublimation_energy
    f_thomson = calculate_thomson_energy_spectrum(e_curve, "Cu", v_d)
    ax4.plot(e_curve, f_thomson, "r-", lw=2.0, label=r"Analytical Thomson $f(E)$")
    
    # Mark theoretical peak at Us / 2
    e_peak_theory = u_s / 2.0
    ax4.axvline(e_peak_theory, color="#000000", linestyle=":", lw=1.5, label=f"Peak $E = U_s/2 = {e_peak_theory:.2f}$ eV")
    ax4.set_xlabel(r"Nascent Atom Kinetic Energy $E$ [eV]")
    ax4.set_ylabel(r"Probability Density $f(E)$ [$\mathrm{eV^{-1}}$]")
    ax4.set_title(r"(d) Stage 2: Thomson Cascade Energy Spectrum", loc="left", fontweight="bold")
    ax4.set_xlim(0, 30.0)
    ax4.legend(loc="upper right", framealpha=0.9)
    ax4.grid(True)

    # -------------------------------------------------------------------------
    # Panel (e): Stage 1 -> Stage 2 Dynamic Racetrack Groove Deepening d(r, t)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    time_points = res.erosion.time_hours
    depth_hist = res.erosion.depth_history_mm
    
    # Plot snapshots at t = 2, 4, 6, 8, 10 hours
    selected_hours = [2.0, 4.0, 6.0, 8.0, 10.0]
    colors = plt.cm.copper(np.linspace(0.8, 0.2, len(selected_hours)))
    
    for h_target, col in zip(selected_hours, colors):
        idx = int(np.argmin(np.abs(time_points - h_target)))
        ax5.plot(
            res.erosion.r_grid_mm,
            depth_hist[idx],
            lw=1.8,
            color=col,
            label=f"$t = {time_points[idx]:.1f}$ h",
        )

    ax5.axhline(6.0, color="#d62728", linestyle="--", lw=1.4, label="Target Thickness (6.0 mm)")
    ax5.set_xlabel(r"Target Radius $r$ [mm]")
    ax5.set_ylabel(r"Erosion Groove Depth $d(r, t)$ [mm]")
    ax5.set_title(r"(e) Stage 2: Dynamic Racetrack Groove Deepening", loc="left", fontweight="bold")
    ax5.set_ylim(-0.2, 6.5)
    ax5.invert_yaxis()  # Invert so 0 mm is top surface and deeper is downward
    ax5.legend(loc="lower right", framealpha=0.9, fontsize=8.0)
    ax5.grid(True)

    # -------------------------------------------------------------------------
    # Panel (f): Target Performance Metrics (Peak Depth & Utilization vs Time)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Peak depth vs hours
    peak_depth_history = [np.max(d) for d in depth_hist]
    line1 = ax6.plot(time_points, peak_depth_history, color="#d62728", lw=2.2, label=r"Peak Depth $d_{\max}(t)$")
    ax6.set_xlabel(r"Operating Duration [hours]")
    ax6.set_ylabel(r"Peak Erosion Depth [mm]", color="#d62728")
    ax6.tick_params(axis="y", labelcolor="#d62728")
    ax6.set_ylim(0, 6.5)

    # Secondary axis: Target utilization efficiency (%)
    ax6_twin = ax6.twinx()
    util_history = [
        res.erosion.target_utilization_efficiency_pct * (d_max / max(1e-6, res.erosion.peak_erosion_depth_mm))
        for d_max in peak_depth_history
    ]
    line2 = ax6_twin.plot(time_points, util_history, color="#1f77b4", lw=2.0, linestyle="--", label=r"Utilization $\eta(t)$")
    ax6_twin.set_ylabel(r"Target Utilization Efficiency $\eta$ [\%]", color="#1f77b4")
    ax6_twin.tick_params(axis="y", labelcolor="#1f77b4")
    ax6_twin.set_ylim(0, 40.0)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc="center right", framealpha=0.9)
    ax6.set_title(r"(f) Target Utilization & Lifetime Metrics", loc="left", fontweight="bold")
    ax6.grid(True)

    # Supertitle
    fig.suptitle(
        r"$\mathbf{SputterTwin\colon\ Integrated\ Multiphysics\ Pipeline\ (Stage\ 1\ Plasma\ \to\ Stage\ 2\ Target\ Erosion)}$"
        f"\nPlanar DC Magnetron Sputtering of Cu Target ($P = {res.power_w:.0f}\\,\\mathrm{{W}}$, $p = {res.pressure_mtorr:.1f}\\,\\mathrm{{mTorr}}$, $t = {res.total_hours:.0f}\\,\\mathrm{{h}}$)",
        fontsize=12.5,
        y=0.98,
        weight="bold",
    )

    # Save to local plots directory
    os.makedirs("plots/09_integrated_stages", exist_ok=True)
    out_local = "plots/09_integrated_stages/integrated_stage1_stage2_dashboard.png"
    plt.savefig(out_local, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    print(f"Saved figure to: {out_local}")

    # Also save to user artifacts directory
    artifact_dir = r"C:\Users\TESTUSER\.gemini\antigravity\brain\abde6399-e280-40dd-ac58-893247689231"
    out_artifact = os.path.join(artifact_dir, "integrated_stage1_stage2_dashboard.png")
    plt.savefig(out_artifact, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    print(f"Saved artifact to: {out_artifact}")
    plt.close()


if __name__ == "__main__":
    main()
