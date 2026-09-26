"""
Compare SputterTwin Yield PINN against Industry-Standard BCA Codes (SRIM & SDTrimSP)
and Laboratory Ground-Truth Experiments (Wolfgang Eckstein, Max Planck IPP).

Generates:
1. Comprehensive 4-way numerical benchmark table.
2. Academic journal publication plot (300 DPI, white background) comparing:
   - SRIM (Stopping and Range of Ions in Matter, J.F. Ziegler)
   - SDTrimSP (Max-Planck-Institut für Plasmaphysik, W. Eckstein & W. Möller)
   - Wolfgang Eckstein Experimental Ground Truth
   - SputterTwin Yield PINN Surrogate
"""

import os
import sys
import numpy as np

# Ensure torch is imported first to avoid Windows c10.dll issues
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sputtertwin.data.srim_sdtrimsp_data import get_bca_benchmark
from sputtertwin.data.eckstein_sputter_yield_data import get_eckstein_data
from sputtertwin.pinn.yield_pinn import load_yield_pinn

ARTIFACT_DIR = r"C:\Users\TESTUSER\.gemini\antigravity\brain\abde6399-e280-40dd-ac58-893247689231"
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots", "08_yield_sota"))
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_bca_comparison():
    print("=" * 95)
    print("  4-WAY INDUSTRY SOTA BENCHMARK: SRIM vs. SDTrimSP vs. ECKSTEIN EXPERIMENT vs. SPUTTERTWIN")
    print("=" * 95)

    model = load_yield_pinn()
    model.eval()

    # Academic styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.edgecolor": "#1e293b",
        "axes.linewidth": 1.1,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "xtick.color": "#0f172a",
        "ytick.color": "#0f172a",
        "text.color": "#0f172a",
        "axes.labelcolor": "#0f172a",
    })

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("#ffffff")
    for ax in axs:
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#0f172a", direction="in", length=4, width=1.1)
        ax.grid(True)

    materials = [("Cu", "Copper", axs[0]), ("Ti", "Titanium", axs[1]), ("Al", "Aluminum", axs[2])]

    dense_e = np.linspace(50.0, 1000.0, 200)

    for mat, mat_name, ax in materials:
        srim = get_bca_benchmark("SRIM", mat)
        sdtrimsp = get_bca_benchmark("SDTrimSP", mat)
        exp = get_eckstein_data(mat, normal_incidence_only=True)

        with torch.no_grad():
            twin_dense = np.array([model.predict_yield(float(e), 0.0, mat) for e in dense_e])
            twin_at_bca = np.array([model.predict_yield(float(e), 0.0, mat) for e in srim["energy_ev"]])

        # Print table
        print(f"\nTarget Material: {mat_name} ({mat}) - Normal Incidence Ar+ Bombardment:")
        print(f"{'Energy (eV)':<12} | {'SRIM (BCA)':<12} | {'SDTrimSP (BCA)':<15} | {'Eckstein Exp':<15} | {'SputterTwin PINN':<18} | {'Dev vs Exp (%)':<15}")
        print("-" * 95)
        for i, e in enumerate(srim["energy_ev"]):
            srim_y = srim["yield_atoms_per_ion"][i]
            sd_y = sdtrimsp["yield_atoms_per_ion"][i]
            tw_y = twin_at_bca[i]
            # Find closest experimental point
            exp_match = exp["yield_atoms_per_ion"][np.abs(exp["energy_ev"] - e) < 1.0]
            exp_str = f"{exp_match[0]:.3f}" if len(exp_match) > 0 else "N/A"
            dev_str = f"{abs(tw_y - exp_match[0])/exp_match[0]*100:.2f}%" if len(exp_match) > 0 else "N/A"
            print(f"{e:<12.0f} | {srim_y:<12.3f} | {sd_y:<15.3f} | {exp_str:<15} | {tw_y:<18.3f} | {dev_str:<15}")

        # Plot curves
        ax.plot(dense_e, twin_dense, label="SputterTwin Yield PINN", color="#b91c1c", lw=2.4, zorder=4)
        ax.plot(srim["energy_ev"], srim["yield_atoms_per_ion"], "o--", label="SRIM (ZBL Potential)", color="#1d4ed8", lw=1.8, markersize=5)
        ax.plot(sdtrimsp["energy_ev"], sdtrimsp["yield_atoms_per_ion"], "s-.", label="SDTrimSP (Max Planck IPP)", color="#047857", lw=1.8, markersize=5)
        ax.scatter(exp["energy_ev"], exp["yield_atoms_per_ion"], label="Eckstein Exp. Ground Truth", color="#d97706", s=40, zorder=5, edgecolor="black", linewidth=0.8)

        ax.set_title(r"$\mathrm{Ar}^+$ on " + f"{mat_name} ({mat})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(r"Incident $\mathrm{Ar}^+$ Kinetic Energy (eV)", fontsize=10, fontweight="bold")
        if mat == "Cu":
            ax.set_ylabel("Sputter Yield Y (atoms / ion)", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1020)
        ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8, loc="upper left")

    plt.suptitle("Cross-Benchmark of Sputter Yield: Industry BCA Codes vs. Experimental Data vs. SputterTwin\n"
                 "SRIM (ZBL) | SDTrimSP (Kr-C Dynamic) | Wolfgang Eckstein IPP Experiments | SputterTwin PINN",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    plot_path_local = os.path.join(PLOTS_DIR, "srim_sdtrimsp_vs_sputtertwin.png")
    plot_path_artifact = os.path.join(ARTIFACT_DIR, "srim_sdtrimsp_vs_sputtertwin.png")

    plt.savefig(plot_path_local, dpi=300, facecolor="white", edgecolor="none")
    plt.savefig(plot_path_artifact, dpi=300, facecolor="white", edgecolor="none")
    plt.close()

    print("\n" + "=" * 95)
    print(f"Academic white comparison plot saved to: {plot_path_local}")
    print(f"Artifact plot saved to: {plot_path_artifact}")
    print("=" * 95)


if __name__ == "__main__":
    run_bca_comparison()
