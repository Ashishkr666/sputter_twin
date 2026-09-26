"""Generate 4-Panel 2D Multiphysics Field Contour Visualizations.

Generates:
1. 2D Magnetic Field |B(r, z)| with magnetic flux streamlines and racetrack null-point.
2. 2D Electrostatic Potential V(r, z) resolving the cathode sheath potential drop.
3. 2D Electron Density n_e(r, z) showing magnetic trap confinement.
4. 2D Gas Temperature T_gas(r, z) demonstrating the high-power sputter wind rarefaction.
"""

from __future__ import annotations

import os
import sys

import torch  # Ensure torch is imported before matplotlib on Windows
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sputtertwin.physics2d import (
    compute_magnetron_magnetic_field,
    simulate_plasma_2d,
    compute_gas_rarefaction_2d,
)


def generate_multiphysics_2d_dashboard(
    power_w: float = 220.0,
    pressure_mtorr: float = 5.0,
    save_path: str = "plots/07_multiphysics_2d/multiphysics_2d_fields_dashboard.png",
) -> str:
    """Generate and save 4-panel publication-grade 2D spatial field contours."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Solving 2D Multiphysics fields at W = {power_w:.0f} W, P = {pressure_mtorr:.1f} mTorr...")
    b2d = compute_magnetron_magnetic_field(grid_r_points=80, grid_z_points=60)
    p2d = simulate_plasma_2d(power_w=power_w, pressure_mtorr=pressure_mtorr, mag_field=b2d)
    g2d = compute_gas_rarefaction_2d(
        power_w=power_w,
        pressure_mtorr=pressure_mtorr,
        racetrack_radius_m=p2d.racetrack_radius_m,
        grid_r=len(b2d.r_grid_m),
        grid_z=len(b2d.z_grid_m),
    )

    R_mm = b2d.r_grid_m * 1000.0
    Z_mm = b2d.z_grid_m * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Magnetic Field |B(r, z)| with Streamlines
    ax1 = axes[0, 0]
    b_gauss = b2d.B_mag * 10000.0
    cf1 = ax1.contourf(R_mm, Z_mm, b_gauss, levels=40, cmap="viridis")
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label("|B| (Gauss)", fontsize=11)

    # Add magnetic flux streamlines
    ax1.contour(R_mm, Z_mm, b2d.psi_flux, levels=15, colors="white", alpha=0.6, linewidths=0.8)
    ax1.axvline(b2d.racetrack_radius_m * 1000.0, color="red", linestyle="--", lw=1.8, label=f"Racetrack ({b2d.racetrack_radius_m*1000:.1f} mm)")
    ax1.set_xlabel("Cathode Radial Coordinate $r$ (mm)", fontsize=11)
    ax1.set_ylabel("Axial Distance from Cathode $z$ (mm)", fontsize=11)
    ax1.set_title("1. Magnetic Field $|\\mathbf{B}(r, z)|$ & Flux Lines", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)

    # Panel 2: Electrostatic Potential V(r, z)
    ax2 = axes[0, 1]
    cf2 = ax2.contourf(R_mm, Z_mm, p2d.V_2d, levels=40, cmap="coolwarm")
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label("Potential $V$ (Volts)", fontsize=11)
    ax2.contour(R_mm, Z_mm, p2d.V_2d, levels=[-300, -200, -100, -50, 0, 10], colors="black", alpha=0.5, linewidths=0.7)
    ax2.set_xlabel("Cathode Radial Coordinate $r$ (mm)", fontsize=11)
    ax2.set_ylabel("Axial Distance from Cathode $z$ (mm)", fontsize=11)
    ax2.set_title(f"2. Electrostatic Potential $V(r, z)$ ($V_d = {p2d.voltage_v:.1f}$ V)", fontsize=12, fontweight="bold")

    # Panel 3: Electron Density n_e(r, z) (Magnetic Trap)
    ax3 = axes[1, 0]
    ne_cm3 = p2d.ne_2d / 1e6  # Convert to cm^-3
    cf3 = ax3.contourf(R_mm, Z_mm, ne_cm3, levels=40, cmap="inferno")
    cbar3 = plt.colorbar(cf3, ax=ax3)
    cbar3.set_label("Electron Density $n_e$ (cm$^{-3}$)", fontsize=11)
    ax3.axvline(b2d.racetrack_radius_m * 1000.0, color="cyan", linestyle="--", lw=1.5, label="Magnetic Trap Center")
    ax3.set_xlabel("Cathode Radial Coordinate $r$ (mm)", fontsize=11)
    ax3.set_ylabel("Axial Distance from Cathode $z$ (mm)", fontsize=11)
    ax3.set_title(f"3. Electron Density $n_e(r, z)$ (Peak = {np.max(p2d.ne_2d):.2e} m$^{{-3}}$)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)

    # Panel 4: Gas Temperature T_gas(r, z) & Rarefaction
    ax4 = axes[1, 1]
    cf4 = ax4.contourf(R_mm, Z_mm, g2d.T_gas_2d_K, levels=40, cmap="plasma")
    cbar4 = plt.colorbar(cf4, ax=ax4)
    cbar4.set_label("Gas Temperature $T_g$ (K)", fontsize=11)
    ax4.contour(R_mm, Z_mm, g2d.rarefaction_factor_2d, levels=[0.70, 0.80, 0.90, 0.95], colors="white", alpha=0.7, linewidths=0.8)
    ax4.set_xlabel("Cathode Radial Coordinate $r$ (mm)", fontsize=11)
    ax4.set_ylabel("Axial Distance from Cathode $z$ (mm)", fontsize=11)
    ax4.set_title(f"4. Sputter Wind Gas Heating (Peak $= {g2d.peak_gas_temp_k:.1f}$ K, Rarefaction $= {g2d.effective_pressure_reduction_pct:.1f}\\%$)", fontsize=12, fontweight="bold")

    plt.suptitle("SputterTwin 2D Multiphysics Simulation: Coupled Field Solutions", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"  [OK] Saved 2D Multiphysics dashboard to: {save_path}")
    return save_path


if __name__ == "__main__":
    generate_multiphysics_2d_dashboard()
