"""PlasmaPy Benchmark Test Suite for SputterTwin DC Magnetron Plasma Physics.

Benchmarks plasma parameters computed in sputtertwin.physics.plasma (Bohm velocity,
sheath dynamics, Debye length, plasma frequency, and thermal velocity hierarchies)
against PlasmaPy's formal theoretical formulation.
"""

import math
import unittest

import numpy as np
import pytest

from sputtertwin.physics.plasma import (
    _ELEMENTARY_CHARGE_C,
    _M_AR_KG,
    calculate_discharge_state,
)

# Conditionally import plasmapy and astropy units
try:
    import astropy.units as u
    from plasmapy.formulary.frequencies import plasma_frequency
    from plasmapy.formulary.lengths import Debye_length
    from plasmapy.formulary.speeds import ion_sound_speed, thermal_speed
    from plasmapy.particles import Particle

    HAS_PLASMAPY = True
except ImportError:
    HAS_PLASMAPY = False


@pytest.mark.skipif(not HAS_PLASMAPY, reason="PlasmaPy is required for this benchmark suite")
class TestPlasmaPyBenchmark(unittest.TestCase):
    """Benchmark SputterTwin discharge physics against PlasmaPy formulary."""

    def setUp(self):
        """Set up nominal DC magnetron operating point (Cu/Ti, 220W, 5 mTorr)."""
        self.power_w = 220.0
        self.pressure_mtorr = 5.0
        self.ar_flow_sccm = 20.0
        self.ds = calculate_discharge_state(
            power_w=self.power_w,
            pressure_mtorr=self.pressure_mtorr,
            ar_flow_sccm=self.ar_flow_sccm,
        )

    def test_bohm_velocity_agreement(self):
        """Benchmark Bohm velocity u_bohm against PlasmaPy's ion_sound_speed.

        In SputterTwin:
            u_bohm = sqrt(e * Te / M_Ar)
        In PlasmaPy:
            ion_sound_speed(T_e, ion='Ar-40 1+', T_i=0*K, gamma_e=1, gamma_i=1)
        """
        # SputterTwin calculation
        u_bohm_sputtertwin = math.sqrt(
            (_ELEMENTARY_CHARGE_C * self.ds.electron_temp_ev) / _M_AR_KG
        )

        # PlasmaPy calculation with cold ions (Ti << Te) standard for DC magnetrons
        u_bohm_plasmapy = ion_sound_speed(
            T_e=self.ds.electron_temp_ev * u.eV,
            ion="Ar-40 1+",
            gamma_e=1.0,
            gamma_i=1.0,
            T_i=0.0 * u.K,
        ).to(u.m / u.s).value

        rel_error = abs(u_bohm_sputtertwin - u_bohm_plasmapy) / u_bohm_plasmapy
        # Must agree to within 0.1% (governed by minor isotope atomic mass precision difference)
        self.assertLess(
            rel_error,
            1e-3,
            f"Bohm velocity discrepancy: SputterTwin={u_bohm_sputtertwin:.2f} m/s, PlasmaPy={u_bohm_plasmapy:.2f} m/s (rel error={rel_error:.2e})",
        )

    def test_debye_length_physical_scale(self):
        """Verify that the plasma Debye length lambda_D matches typical magnetron conditions.

        For n_e ~ 10^16 - 10^17 m^-3 and T_e ~ 3 eV, lambda_D should be ~ 30-100 um.
        """
        lambda_d = Debye_length(
            T_e=self.ds.electron_temp_ev * u.eV,
            n_e=self.ds.plasma_density_m3 * (u.m ** -3),
        ).to(u.um).value

        # Physically expected Debye length in DC magnetron racetrack trap: 10 um to 300 um
        self.assertGreater(lambda_d, 10.0, "Debye length too small for DC magnetron")
        self.assertLess(lambda_d, 300.0, "Debye length too large for DC magnetron")

        # Child-Langmuir sheath thickness s ~ lambda_D * (2*Vd / Te)^(3/4)
        # For Vd ~ 400V, Te ~ 3eV, (2*400/3)^0.75 ~ 66. Sheath ~ 66 * lambda_D ~ 1-5 mm.
        sheath_thickness_est_mm = (lambda_d * 1e-3) * ((2.0 * self.ds.voltage_v / self.ds.electron_temp_ev) ** 0.75)
        self.assertGreater(sheath_thickness_est_mm, 0.2, "Estimated sheath thickness under 0.2 mm")
        self.assertLess(sheath_thickness_est_mm, 10.0, "Estimated sheath thickness over 10 mm")

    def test_plasma_frequency_and_quasineutrality(self):
        """Verify electron plasma frequency omega_pe >> ion plasma frequency omega_pi."""
        omega_pe = plasma_frequency(
            n=self.ds.plasma_density_m3 * (u.m ** -3),
            particle="e-",
        ).to(u.rad / u.s).value

        omega_pi = plasma_frequency(
            n=self.ds.plasma_density_m3 * (u.m ** -3),
            particle="Ar+",
        ).to(u.rad / u.s).value

        # High frequency electron response (GHz) vs massive ion response (MHz)
        self.assertGreater(omega_pe, 1e9, "Electron plasma frequency should be > 1 GHz")
        self.assertGreater(omega_pe / omega_pi, 100.0, "omega_pe / omega_pi must exceed 100")

    def test_velocity_hierarchy_criterion(self):
        """Verify the fundamental sheath criterion hierarchy: v_th,e >> u_Bohm >> v_th,i.

        This hierarchy is the physical foundation that permits the Bohm sheath approximation.
        """
        # Electron thermal speed
        v_th_e = thermal_speed(
            T=self.ds.electron_temp_ev * u.eV,
            particle="e-",
            method="most_probable",
        ).to(u.m / u.s).value

        # Argon ion thermal speed assuming room temp neutral/ion background (300 K)
        v_th_i = thermal_speed(
            T=300.0 * u.K,
            particle="Ar+",
            method="most_probable",
        ).to(u.m / u.s).value

        u_bohm = math.sqrt((_ELEMENTARY_CHARGE_C * self.ds.electron_temp_ev) / _M_AR_KG)

        self.assertGreater(v_th_e, 50.0 * u_bohm, "Electrons must be much faster than Bohm speed")
        self.assertGreater(u_bohm, 2.0 * v_th_i, "Bohm speed must exceed room-temperature ion thermal speed")

    def test_bohm_flux_reconstruction(self):
        """Verify that Gamma_i = exp(-0.5) * n_e * Bohm_velocity reconstructs SputterTwin ion_flux."""
        u_bohm_plasmapy = ion_sound_speed(
            T_e=self.ds.electron_temp_ev * u.eV,
            ion="Ar-40 1+",
            gamma_e=1.0,
            gamma_i=1.0,
            T_i=0.0 * u.K,
        ).to(u.m / u.s).value

        reconstructed_flux = math.exp(-0.5) * self.ds.plasma_density_m3 * u_bohm_plasmapy
        rel_flux_err = abs(reconstructed_flux - self.ds.ion_flux) / self.ds.ion_flux
        self.assertLess(rel_flux_err, 1e-3, "Reconstructed ion flux deviates from SputterTwin state")


if __name__ == "__main__":
    unittest.main()
