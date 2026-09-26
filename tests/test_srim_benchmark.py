"""Unit tests for SRIM and SDTrimSP simulation benchmark validation."""

import pytest
import numpy as np

# Ensure torch is imported first to avoid Windows c10.dll issues
import torch

from sputtertwin.data.srim_sdtrimsp_data import (
    SRIM_SDTRIMSP_BENCHMARK_DATA,
    get_bca_benchmark,
)
from sputtertwin.pinn.yield_pinn import load_yield_pinn


@pytest.fixture(scope="module")
def yield_pinn_model():
    model = load_yield_pinn()
    model.eval()
    return model


def test_bca_dataset_integrity():
    """Verify that both SRIM and SDTrimSP benchmark data exist for Cu, Ti, and Al."""
    for mat in ["Cu", "Ti", "Al"]:
        for code in ["SRIM", "SDTrimSP"]:
            data = get_bca_benchmark(code, mat)
            assert len(data["energy_ev"]) == 8, f"Expected 8 energy points for {code}-{mat}"
            assert np.all(data["yield_atoms_per_ion"] > 0.0), f"Yields must be positive for {code}-{mat}"
            assert np.all(np.diff(data["yield_atoms_per_ion"]) > 0.0), f"Yields must monotonically increase for {code}-{mat}"


def test_sputtertwin_vs_sdtrimsp_cu_agreement(yield_pinn_model):
    """Verify SputterTwin tracks SDTrimSP (Max Planck IPP) within standard BCA code variance (< 6%)."""
    sdtrimsp_cu = get_bca_benchmark("SDTrimSP", "Cu")
    energies = sdtrimsp_cu["energy_ev"]
    sdtrimsp_yields = sdtrimsp_cu["yield_atoms_per_ion"]

    twin_yields = np.array([
        yield_pinn_model.predict_yield(float(e), 0.0, "Cu")
        for e in energies
    ])

    rel_diffs = np.abs(twin_yields - sdtrimsp_yields) / sdtrimsp_yields * 100.0
    mean_diff = float(np.mean(rel_diffs))
    max_diff = float(np.max(rel_diffs))

    print(f"\nSputterTwin vs SDTrimSP (Cu): Mean Dev = {mean_diff:.2f}%, Max Dev = {max_diff:.2f}%")
    assert mean_diff < 6.0, f"Mean deviation from SDTrimSP ({mean_diff:.2f}%) exceeds 6.0%"


def test_sputtertwin_vs_srim_cu_agreement(yield_pinn_model):
    """Verify SputterTwin tracks SRIM (Ziegler BCA) within standard BCA code variance (< 8%)."""
    srim_cu = get_bca_benchmark("SRIM", "Cu")
    energies = srim_cu["energy_ev"]
    srim_yields = srim_cu["yield_atoms_per_ion"]

    twin_yields = np.array([
        yield_pinn_model.predict_yield(float(e), 0.0, "Cu")
        for e in energies
    ])

    rel_diffs = np.abs(twin_yields - srim_yields) / srim_yields * 100.0
    mean_diff = float(np.mean(rel_diffs))

    print(f"\nSputterTwin vs SRIM (Cu): Mean Dev = {mean_diff:.2f}%")
    assert mean_diff < 8.0, f"Mean deviation from SRIM ({mean_diff:.2f}%) exceeds 8.0%"


def test_sputtertwin_vs_srim_al_agreement(yield_pinn_model):
    """Verify SputterTwin matches SRIM for Aluminum within < 3.5% mean deviation."""
    srim_al = get_bca_benchmark("SRIM", "Al")
    energies = srim_al["energy_ev"]
    srim_yields = srim_al["yield_atoms_per_ion"]

    twin_yields = np.array([
        yield_pinn_model.predict_yield(float(e), 0.0, "Al")
        for e in energies
    ])

    rel_diffs = np.abs(twin_yields - srim_yields) / srim_yields * 100.0
    mean_diff = float(np.mean(rel_diffs))

    print(f"\nSputterTwin vs SRIM (Al): Mean Dev = {mean_diff:.2f}%")
    assert mean_diff < 3.5, f"Mean deviation from SRIM Al ({mean_diff:.2f}%) exceeds 3.5%"


def test_sputtertwin_vs_sdtrimsp_all_materials(yield_pinn_model):
    """Verify SputterTwin tracks SDTrimSP across standard targets (Cu < 6%, Al < 12%, Ti < 20%)."""
    thresholds = {"Cu": 6.0, "Al": 12.0, "Ti": 20.0}
    for mat in ["Cu", "Ti", "Al"]:
        data = get_bca_benchmark("SDTrimSP", mat)
        energies = data["energy_ev"]
        bca_yields = data["yield_atoms_per_ion"]

        twin_yields = np.array([
            yield_pinn_model.predict_yield(float(e), 0.0, mat)
            for e in energies
        ])

        rel_diffs = np.abs(twin_yields - bca_yields) / bca_yields * 100.0
        mean_diff = float(np.mean(rel_diffs))
        thresh = thresholds[mat]
        assert mean_diff < thresh, f"Mean deviation for {mat} ({mean_diff:.2f}%) exceeds {thresh}%"
