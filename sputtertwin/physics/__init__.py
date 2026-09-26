"""Physics package for SputterTwin.

Exposes core plasma discharge, sputter yield, gas transport, and thin-film
deposition simulation models for DC magnetron sputtering systems.
"""

from sputtertwin.physics.deposition import (
    DepositionResult,
    simulate_deposition,
)
from sputtertwin.physics.plasma import (
    DischargeState,
    calculate_discharge_state,
    format_discharge_state,
)
from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
    calculate_sputter_yield_array,
    calculate_kinematic_factor,
    calculate_thomson_energy_spectrum,
    sample_ejected_energy_and_angle,
    calculate_sputter_yield_sota,
    EjectedParticles,
)
from sputtertwin.physics.target_erosion import (
    TargetErosionModel,
    TargetErosionResult,
)
from sputtertwin.physics.transport import (
    TransportSummary,
    calculate_knudsen_number,
    calculate_mean_free_path,
    calculate_scattering_broadening,
    calculate_transmission_probability,
    calculate_transport_summary,
)
from sputtertwin.physics2d import (
    compute_magnetron_magnetic_field,
    simulate_plasma_2d,
    compute_gas_rarefaction_2d,
    Plasma2DResult,
    MagneticField2D,
    GasRarefaction2D,
)

from sputtertwin.physics.integrated import (
    IntegratedDischargeErosionResult,
    IntegratedPINNSurrogate,
    simulate_integrated_discharge_and_erosion,
)

__all__ = [
    "TargetMaterial",
    "MATERIALS",
    "calculate_sputter_yield",
    "calculate_sputter_yield_array",
    "calculate_kinematic_factor",
    "calculate_thomson_energy_spectrum",
    "sample_ejected_energy_and_angle",
    "calculate_sputter_yield_sota",
    "EjectedParticles",
    "DischargeState",
    "calculate_discharge_state",
    "format_discharge_state",
    "calculate_mean_free_path",
    "calculate_knudsen_number",
    "calculate_transmission_probability",
    "calculate_scattering_broadening",
    "calculate_transport_summary",
    "TransportSummary",
    "DepositionResult",
    "simulate_deposition",
    "TargetErosionModel",
    "TargetErosionResult",
    "compute_magnetron_magnetic_field",
    "simulate_plasma_2d",
    "compute_gas_rarefaction_2d",
    "Plasma2DResult",
    "MagneticField2D",
    "GasRarefaction2D",
    "IntegratedDischargeErosionResult",
    "IntegratedPINNSurrogate",
    "simulate_integrated_discharge_and_erosion",
]

