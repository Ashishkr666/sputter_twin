"""
SputterTwin - Physics-Informed Digital Twin for DC Magnetron Sputtering.

An end-to-end multiphysics and deep learning simulation platform for
semiconductor and thin-film PVD processing.
"""

__version__ = "0.2.0"

# Unified High-Level Physics Exports
from sputtertwin.physics import (
    # Stage 1: Plasma & Sheath Physics
    DischargeState,
    calculate_discharge_state,
    format_discharge_state,
    simulate_plasma_2d,
    compute_magnetron_magnetic_field,
    compute_gas_rarefaction_2d,
    Plasma2DResult,
    MagneticField2D,
    GasRarefaction2D,
    # Stage 2: Sputter Yield & Target Erosion
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
    calculate_sputter_yield_sota,
    calculate_thomson_energy_spectrum,
    sample_ejected_energy_and_angle,
    TargetErosionModel,
    TargetErosionResult,
    # Stage 3: Gas-Phase Transport
    calculate_mean_free_path,
    calculate_knudsen_number,
    calculate_transmission_probability,
    calculate_transport_summary,
    TransportSummary,
    # Stage 4: Film Deposition
    simulate_deposition,
    DepositionResult,
    # End-to-End Integrated Multiphysics Pipeline
    simulate_integrated_discharge_and_erosion,
    IntegratedDischargeErosionResult,
    IntegratedPINNSurrogate,
)

__all__ = [
    "__version__",
    "DischargeState",
    "calculate_discharge_state",
    "format_discharge_state",
    "simulate_plasma_2d",
    "compute_magnetron_magnetic_field",
    "compute_gas_rarefaction_2d",
    "Plasma2DResult",
    "MagneticField2D",
    "GasRarefaction2D",
    "MATERIALS",
    "TargetMaterial",
    "calculate_sputter_yield",
    "calculate_sputter_yield_sota",
    "calculate_thomson_energy_spectrum",
    "sample_ejected_energy_and_angle",
    "TargetErosionModel",
    "TargetErosionResult",
    "calculate_mean_free_path",
    "calculate_knudsen_number",
    "calculate_transmission_probability",
    "calculate_transport_summary",
    "TransportSummary",
    "simulate_deposition",
    "DepositionResult",
    "simulate_integrated_discharge_and_erosion",
    "IntegratedDischargeErosionResult",
    "IntegratedPINNSurrogate",
]
