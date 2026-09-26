"""2D Dynamic Target Racetrack Erosion Engine for DC Magnetron Sputtering.

Simulates the time-dependent mechanical erosion of circular planar magnetron
targets over operational hours.

Physics Model:
1. Coupling: Couples cathode ion current density J_i(r) [A/m^2] or ion flux
   Gamma_i(r) [m^-2 s^-1] from the 2D plasma solver with the energy and
   angular-dependent Yamamura-Tawara sputter yield Y(E_eff, theta_local).
2. Local groove tilt enhancement:
   theta_local(r) = arctan(|d(depth)/dr|)
   As racetrack erosion deepens, groove sidewalls steepen, increasing the local
   incidence angle and boosting the local sputter yield Y(theta) via Yamamura
   oblique incidence enhancement.
3. Target atomic volume:
   n_target = rho * N_A / M  [atoms / m^3]
   where rho is mass density (kg/m^3), N_A is Avogadro's constant, and M is molar mass.
4. Dynamic erosion rate:
   d(depth)/dt = J_i(r) * Y(E_eff, theta_local) / (e * n_target)  [m / s]
5. Key output metrics:
   - Peak erosion depth (mm)
   - Racetrack groove width at FWHM (mm)
   - Target utilization efficiency (%) = (volume sputtered) / (total volume) * 100
   - Target lifetime (hours until breakthrough through target thickness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

from sputtertwin.physics.sputter_yield import (
    MATERIALS,
    TargetMaterial,
    calculate_sputter_yield,
    calculate_sputter_yield_sota,
)

# Fundamental physical constants
_AVOGADRO: float = 6.02214076e23        # Avogadro constant in atoms / mol
_E_CHARGE: float = 1.602176634e-19     # Elementary charge in C


@dataclass
class TargetErosionResult:
    """Comprehensive results of a dynamic target racetrack erosion simulation.

    Attributes:
        r_grid_mm: 1D radial grid coordinates across the target disk in mm.
        r_grid_m: 1D radial grid coordinates in meters.
        depth_profile_mm: 1D final groove depth profile d(r) in mm.
        depth_profile_m: 1D final groove depth profile d(r) in meters.
        depth_history_mm: 2D array of groove depth profiles at each time step (shape: [steps, n_r]) in mm.
        time_hours: 1D array of simulation time points in hours.
        erosion_rate_mm_hr: 1D final erosion rate profile across radius in mm/h.
        erosion_rate_m_s: 1D final erosion rate profile across radius in m/s.
        peak_erosion_depth_mm: Maximum groove erosion depth reached in mm.
        racetrack_fwhm_mm: Full width at half maximum of the racetrack groove in mm.
        target_utilization_efficiency_pct: Target utilization efficiency percentage (0 - 100%).
        target_lifetime_hours: Operational hours until breakthrough through initial thickness.
        breakthrough_occurred: True if breakthrough occurred during the simulation.
        breakthrough_time_hours: Exact time of breakthrough in hours (or None if not reached).
        material: Chemical symbol of the target material ('Cu', 'Ti', 'Al').
        discharge_voltage_v: Cathode discharge voltage in Volts.
        initial_thickness_mm: Initial target thickness in mm.
        target_radius_mm: Target disk radius in mm.
    """

    r_grid_mm: np.ndarray
    r_grid_m: np.ndarray
    depth_profile_mm: np.ndarray
    depth_profile_m: np.ndarray
    depth_history_mm: np.ndarray
    time_hours: np.ndarray
    erosion_rate_mm_hr: np.ndarray
    erosion_rate_m_s: np.ndarray
    peak_erosion_depth_mm: float
    racetrack_fwhm_mm: float
    target_utilization_efficiency_pct: float
    target_lifetime_hours: float
    breakthrough_occurred: bool = False
    breakthrough_time_hours: Optional[float] = None
    material: str = "Cu"
    discharge_voltage_v: float = 400.0
    initial_thickness_mm: float = 6.0
    target_radius_mm: float = 25.0
    roughness_factor: float = 0.0

    # User-friendly property aliases
    @property
    def peak_depth_mm(self) -> float:
        """Alias for peak_erosion_depth_mm."""
        return self.peak_erosion_depth_mm

    @property
    def groove_width_fwhm_mm(self) -> float:
        """Alias for racetrack_fwhm_mm."""
        return self.racetrack_fwhm_mm

    @property
    def fwhm_mm(self) -> float:
        """Alias for racetrack_fwhm_mm."""
        return self.racetrack_fwhm_mm

    @property
    def target_utilization_pct(self) -> float:
        """Alias for target_utilization_efficiency_pct."""
        return self.target_utilization_efficiency_pct

    @property
    def utilization_pct(self) -> float:
        """Alias for target_utilization_efficiency_pct."""
        return self.target_utilization_efficiency_pct

    @property
    def lifetime_hours(self) -> float:
        """Alias for target_lifetime_hours."""
        return self.target_lifetime_hours

    def reconstruct_2d_depth_map(
        self, grid_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct 2D (x, y) Cartesian surface map of target erosion depth.

        Args:
            grid_size: Number of points along x and y axes.

        Returns:
            Tuple of (X_mm, Y_mm, Depth_mm) 2D arrays. Points outside the
            circular target disk are set to NaN.
        """
        r_max = self.target_radius_mm
        x = np.linspace(-r_max, r_max, grid_size)
        y = np.linspace(-r_max, r_max, grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        # Interpolate radial depth profile onto 2D Cartesian grid
        depth_2d = np.interp(R, self.r_grid_mm, self.depth_profile_mm, right=0.0)
        # Mask off-target points
        depth_2d[R > r_max] = np.nan
        return X, Y, depth_2d


class TargetErosionModel:
    """2D Dynamic Target Racetrack Erosion Engine for planar circular magnetrons.

    Simulates the time evolution of target erosion d(r, t) under non-uniform
    ion bombardment, incorporating local groove wall tilt enhancement.

    Attributes:
        material: TargetMaterial instance for target.
        target_radius_mm: Target disk radius in mm.
        target_radius_m: Target disk radius in meters.
        initial_thickness_mm: Initial target plate thickness in mm.
        initial_thickness_m: Initial target plate thickness in meters.
        n_target: Target atomic density in atoms / m^3 (rho * N_A / M).
        roughness_factor: Surface micro-roughness damping parameter (Ruzic model).
    """

    def __init__(
        self,
        target_material: Union[str, TargetMaterial] = "Cu",
        target_radius: float = 25.0,
        initial_target_thickness: float = 6.0,
        *,
        material: Optional[Union[str, TargetMaterial]] = None,
        target_radius_mm: Optional[float] = None,
        target_radius_m: Optional[float] = None,
        initial_thickness: Optional[float] = None,
        initial_thickness_mm: Optional[float] = None,
        initial_thickness_m: Optional[float] = None,
        roughness_factor: float = 0.0,
    ) -> None:
        """Initialize TargetErosionModel.

        Args:
            target_material: Target material symbol ('Cu', 'Ti', 'Al') or TargetMaterial.
            target_radius: Target radius in mm (or meters if < 0.2). Default: 25.0 mm.
            initial_target_thickness: Target thickness in mm (or meters if < 0.05). Default: 6.0 mm.
            material: Alias for target_material.
            target_radius_mm: Explicit target radius in mm.
            target_radius_m: Explicit target radius in meters.
            initial_thickness: Alias for initial_target_thickness.
            initial_thickness_mm: Explicit initial thickness in mm.
            initial_thickness_m: Explicit initial thickness in meters.
        """
        # 1. Resolve material
        mat_input = material if material is not None else target_material
        if isinstance(mat_input, str):
            if mat_input not in MATERIALS:
                raise KeyError(
                    f"Target material '{mat_input}' not found. Supported: {list(MATERIALS.keys())}"
                )
            self._material_obj = MATERIALS[mat_input]
            self._material_name = mat_input
        elif isinstance(mat_input, TargetMaterial):
            self._material_obj = mat_input
            self._material_name = mat_input.name
        else:
            raise TypeError(
                f"material must be str or TargetMaterial, got {type(mat_input).__name__}"
            )

        # 2. Resolve target radius
        if target_radius_m is not None:
            r_m = float(target_radius_m)
        elif target_radius_mm is not None:
            r_m = float(target_radius_mm) * 1e-3
        else:
            r_val = float(target_radius)
            # Auto-detect units: if < 0.2, assume SI meters (e.g., 0.025 m)
            r_m = r_val if r_val < 0.2 else r_val * 1e-3

        if not math.isfinite(r_m) or r_m <= 0.0:
            raise ValueError(f"target_radius must be positive and finite, got {r_m}")

        self.target_radius_m: float = r_m
        self.target_radius_mm: float = r_m * 1e3
        self.target_radius: float = self.target_radius_mm

        # 3. Resolve initial thickness
        th_val = (
            initial_thickness_m
            if initial_thickness_m is not None
            else (
                initial_thickness_mm * 1e-3
                if initial_thickness_mm is not None
                else (
                    initial_thickness
                    if initial_thickness is not None
                    else initial_target_thickness
                )
            )
        )
        th_val = float(th_val)
        # Auto-detect units: if < 0.05, assume SI meters (e.g., 0.006 m)
        th_m = th_val if th_val < 0.05 else th_val * 1e-3

        if not math.isfinite(th_m) or th_m <= 0.0:
            raise ValueError(
                f"initial_target_thickness must be positive and finite, got {th_m}"
            )

        self.initial_target_thickness_m: float = th_m
        self.initial_target_thickness_mm: float = th_m * 1e3
        self.initial_thickness_m: float = self.initial_target_thickness_m
        self.initial_thickness_mm: float = self.initial_target_thickness_mm
        self.initial_target_thickness: float = self.initial_target_thickness_mm

        # 4. Target atomic density n_target = rho * N_A / M [atoms / m^3]
        # In MATERIALS: density is in g/cm^3, atomic_mass is in g/mol.
        # In SI units:
        # rho_SI = density * 1e3 kg/m^3
        # M_SI = atomic_mass * 1e-3 kg/mol
        # n_target = (rho_SI * N_A) / M_SI = (density * 1e6 * N_A) / atomic_mass
        rho_g_cm3 = self._material_obj.density
        m_g_mol = self._material_obj.atomic_mass
        self.n_target: float = (rho_g_cm3 * 1e6 * _AVOGADRO) / m_g_mol
        self.target_density_m3: float = self.n_target
        self.roughness_factor: float = max(0.0, float(roughness_factor))

        # State tracking for last simulation
        self.last_result: Optional[TargetErosionResult] = None

    @property
    def target_material(self) -> Union[str, TargetMaterial]:
        """Return the target material name."""
        return self._material_name

    @property
    def material_object(self) -> TargetMaterial:
        """Return the underlying TargetMaterial data object."""
        return self._material_obj

    # Key metric proxy properties for convenience
    @property
    def peak_erosion_depth_mm(self) -> float:
        """Peak erosion depth from the latest simulation run (mm)."""
        return self.last_result.peak_erosion_depth_mm if self.last_result else 0.0

    @property
    def racetrack_fwhm_mm(self) -> float:
        """Racetrack FWHM groove width from the latest simulation run (mm)."""
        return self.last_result.racetrack_fwhm_mm if self.last_result else 0.0

    @property
    def target_utilization_efficiency_pct(self) -> float:
        """Target utilization efficiency (%) from the latest simulation run."""
        return self.last_result.target_utilization_efficiency_pct if self.last_result else 0.0

    @property
    def target_lifetime_hours(self) -> float:
        """Target lifetime (hours) from the latest simulation run."""
        return self.last_result.target_lifetime_hours if self.last_result else 0.0

    def calculate_erosion_rate(
        self,
        current_density_a_m2: Union[float, np.ndarray],
        discharge_voltage_v: float,
        theta_local_rad: Union[float, np.ndarray] = 0.0,
        unit: str = "m/s",
        roughness_factor: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Calculate local instantaneous target erosion rate.

        Formula:
            d(depth)/dt = J_i(r) * Y(E_eff, theta_local) / (e * n_target)

        Args:
            current_density_a_m2: Ion current density J_i in A/m^2 (or ion flux if > 1e12).
            discharge_voltage_v: Cathode discharge voltage in Volts (E_eff = |V_d| in eV).
            theta_local_rad: Local surface groove inclination angle in radians (0 = normal).
            unit: Output rate unit: 'm/s' or 'mm/h'.
            roughness_factor: Optional surface micro-roughness damping parameter (Ruzic).

        Returns:
            Erosion rate in specified units.
        """
        e_eff = abs(float(discharge_voltage_v))
        if e_eff <= self._material_obj.threshold_energy:
            return 0.0 if np.isscalar(current_density_a_m2) else np.zeros_like(current_density_a_m2, dtype=float)

        rf = self.roughness_factor if roughness_factor is None else max(0.0, float(roughness_factor))
        j_arr = np.asarray(current_density_a_m2, dtype=float)
        theta_arr = np.asarray(theta_local_rad, dtype=float)

        # Detect particle flux Gamma_i vs current density J_i
        # If > 1e12, input is particle flux Gamma_i (ions / m^2 s), so Gamma_i = input
        # If <= 1e12, input is current density J_i (A / m^2), so Gamma_i = J_i / e
        is_flux = np.any(j_arr > 1e12)
        gamma_i = j_arr if is_flux else (j_arr / _E_CHARGE)

        # Calculate SOTA sputter yield Y(E_eff, theta)
        if theta_arr.ndim == 0:
            y_val = calculate_sputter_yield_sota(e_eff, self._material_obj, float(theta_arr), roughness_factor=rf)
            rate_m_s = gamma_i * y_val / self.n_target
        else:
            y_vals = np.array([
                calculate_sputter_yield_sota(e_eff, self._material_obj, float(th), roughness_factor=rf)
                for th in theta_arr
            ])
            rate_m_s = gamma_i * y_vals / self.n_target

        if unit == "mm/h":
            return rate_m_s * 3.6e6
        return rate_m_s

    def calculate_fwhm(
        self,
        profile: np.ndarray,
        r_grid: np.ndarray,
    ) -> float:
        """Calculate the Full Width at Half Maximum (FWHM) of a radial profile.

        Uses sub-grid linear interpolation around the peak.

        Args:
            profile: 1D array of radial values (e.g. depth in mm or flux).
            r_grid: 1D array of radial coordinates (in mm or m).

        Returns:
            FWHM in the same units as r_grid. Returns 0.0 if profile is flat/zero.
        """
        prof = np.asarray(profile, dtype=float)
        r = np.asarray(r_grid, dtype=float)
        peak_val = float(np.max(prof))
        if peak_val <= 1e-12:
            return 0.0

        half_val = 0.5 * peak_val
        ipeak = int(np.argmax(prof))

        # 1. Search left of peak for crossing
        idx_left = np.where(prof[:ipeak] < half_val)[0]
        if len(idx_left) > 0:
            i = idx_left[-1]
            denom = prof[i + 1] - prof[i]
            frac = (half_val - prof[i]) / denom if abs(denom) > 1e-15 else 0.0
            r_left = r[i] + frac * (r[i + 1] - r[i])
        else:
            r_left = r[0]

        # 2. Search right of peak for crossing
        idx_right = np.where(prof[ipeak:] < half_val)[0]
        if len(idx_right) > 0:
            k = ipeak + idx_right[0]
            denom = prof[k] - prof[k - 1]
            frac = (half_val - prof[k - 1]) / denom if abs(denom) > 1e-15 else 0.0
            r_right = r[k - 1] + frac * (r[k] - r[k - 1])
        else:
            r_right = r[-1]

        return float(max(0.0, r_right - r_left))

    def calculate_target_utilization(
        self,
        depth_profile: np.ndarray,
        r_grid: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate target utilization efficiency percentage.

        Formula:
            utilization (%) = (volume of sputtered target material) / (total target volume) * 100
            V_sputtered = 2 * pi * int_0^R r * d(r) dr
            V_total = pi * R^2 * initial_thickness

        Args:
            depth_profile: 1D array of groove depths. If values > 0.05, assumed mm; else m.
            r_grid: 1D radial coordinates. If values > 0.5, assumed mm; else m.

        Returns:
            Target utilization efficiency in percent (0.0 to 100.0%).
        """
        d_arr = np.asarray(depth_profile, dtype=float)
        # Unit normalization: convert mm to m if needed
        d_m = d_arr * 1e-3 if np.max(d_arr) > 0.05 else d_arr.copy()
        # Clamp to initial thickness so sputtered volume never exceeds target plate
        d_m = np.clip(d_m, 0.0, self.initial_target_thickness_m)

        if r_grid is not None:
            r_arr = np.asarray(r_grid, dtype=float)
            r_m = r_arr * 1e-3 if np.max(r_arr) > 0.5 else r_arr.copy()
        else:
            r_m = np.linspace(0.0, self.target_radius_m, len(d_m))

        v_sputtered = 2.0 * math.pi * float(np.trapz(d_m * r_m, r_m))
        v_total = math.pi * (self.target_radius_m**2) * self.initial_target_thickness_m
        utilization_pct = (v_sputtered / v_total) * 100.0
        return float(np.clip(utilization_pct, 0.0, 100.0))

    def calculate_target_lifetime(
        self,
        peak_depth_mm: float,
        elapsed_hours: float,
        peak_erosion_rate_mm_hr: Optional[float] = None,
    ) -> float:
        """Calculate target operational lifetime in hours until breakthrough.

        Args:
            peak_depth_mm: Current maximum groove depth in mm.
            elapsed_hours: Total simulated time in hours.
            peak_erosion_rate_mm_hr: Optional instantaneous peak erosion rate (mm/h).

        Returns:
            Lifetime in operational hours until complete breakthrough through target thickness.
        """
        if peak_depth_mm >= self.initial_target_thickness_mm:
            return float(elapsed_hours)

        if peak_erosion_rate_mm_hr is not None and peak_erosion_rate_mm_hr > 0.0:
            return float(self.initial_target_thickness_mm / peak_erosion_rate_mm_hr)

        if peak_depth_mm > 0.0 and elapsed_hours > 0.0:
            avg_rate = peak_depth_mm / elapsed_hours
            return float(self.initial_target_thickness_mm / avg_rate)

        return float("inf")

    # Method aliases for convenience
    calculate_lifetime = calculate_target_lifetime
    calculate_utilization = calculate_target_utilization

    def simulate_erosion(
        self,
        ion_flux_profile: Union[np.ndarray, list, Any],
        discharge_voltage_v: Optional[float] = None,
        total_hours: float = 10.0,
        time_step_hours: float = 0.5,
        *,
        r_grid: Optional[np.ndarray] = None,
        clamp_at_thickness: bool = True,
        roughness_factor: Optional[float] = None,
    ) -> TargetErosionResult:
        """Simulate dynamic 2D racetrack target groove erosion over time.

        Evolves groove depth d(r, t) accounting for local surface tilt angle:
            theta_local(r) = arctan(|d(depth)/dr|)
            d(depth)/dt = J_i(r) * Y(E_eff, theta_local) / (e * n_target)

        Args:
            ion_flux_profile: Cathode ion current density J_i(r) [A/m^2], particle flux
                Gamma_i(r) [m^-2 s^-1], or Plasma2DResult instance.
            discharge_voltage_v: Cathode discharge voltage V_d in Volts. If None and
                ion_flux_profile is a Plasma2DResult, uses its voltage_v attribute.
            total_hours: Total operational sputtering duration in hours (e.g. 10.0).
            time_step_hours: Numerical time integration step in hours (default: 0.5).
            r_grid: Optional radial coordinate array (m or mm). If None, inferred
                from Plasma2DResult or constructed linearly from target_radius.
            clamp_at_thickness: If True, restricts groove depth to initial target thickness.
            roughness_factor: Surface micro-roughness damping parameter (Ruzic model).

        Returns:
            TargetErosionResult with depth profiles, histories, and key performance metrics.
        """
        # 1. Unpack ion_flux_profile and extract radial grid and voltage
        if hasattr(ion_flux_profile, "current_density_1d") and hasattr(ion_flux_profile, "r_grid_m"):
            # Plasma2DResult instance
            j_profile = np.asarray(ion_flux_profile.current_density_1d, dtype=float)
            r_grid_m = np.asarray(ion_flux_profile.r_grid_m, dtype=float)
            if discharge_voltage_v is None:
                discharge_voltage_v = float(ion_flux_profile.voltage_v)
        else:
            j_profile = np.asarray(ion_flux_profile, dtype=float)
            if r_grid is not None:
                r_arr = np.asarray(r_grid, dtype=float)
                r_grid_m = r_arr * 1e-3 if np.max(r_arr) > 0.5 else r_arr.copy()
            else:
                r_grid_m = np.linspace(0.0, self.target_radius_m, len(j_profile))

        if discharge_voltage_v is None:
            discharge_voltage_v = 400.0  # standard DC magnetron operating voltage

        rf = self.roughness_factor if roughness_factor is None else max(0.0, float(roughness_factor))
        v_d = abs(float(discharge_voltage_v))
        total_h = max(0.0, float(total_hours))
        dt_h = max(1e-4, float(time_step_hours))

        # Detect particle flux Gamma_i vs current density J_i
        is_flux = np.any(j_profile > 1e12)
        gamma_i = j_profile if is_flux else (j_profile / _E_CHARGE)
        j_i_a_m2 = (j_profile * _E_CHARGE) if is_flux else j_profile

        n_r = len(r_grid_m)
        depth_m = np.zeros(n_r, dtype=float)

        # Handle zero total hours or sub-threshold voltage
        if total_h <= 0.0 or v_d <= self._material_obj.threshold_energy:
            zero_arr = np.zeros(n_r, dtype=float)
            res = TargetErosionResult(
                r_grid_mm=r_grid_m * 1e3,
                r_grid_m=r_grid_m,
                depth_profile_mm=zero_arr,
                depth_profile_m=zero_arr,
                depth_history_mm=np.zeros((1, n_r), dtype=float),
                time_hours=np.array([0.0]),
                erosion_rate_mm_hr=zero_arr,
                erosion_rate_m_s=zero_arr,
                peak_erosion_depth_mm=0.0,
                racetrack_fwhm_mm=0.0,
                target_utilization_efficiency_pct=0.0,
                target_lifetime_hours=float("inf"),
                breakthrough_occurred=False,
                breakthrough_time_hours=None,
                material=self._material_name,
                discharge_voltage_v=v_d,
                initial_thickness_mm=self.initial_target_thickness_mm,
                target_radius_mm=self.target_radius_mm,
                roughness_factor=rf,
            )
            self.last_result = res
            return res

        # 2. Time-stepping schedule
        n_steps = max(1, int(math.ceil(total_h / dt_h)))
        time_points = [0.0]
        depth_history = [depth_m.copy()]

        breakthrough_occurred = False
        breakthrough_time_h: Optional[float] = None
        t_curr = 0.0
        final_rate_m_s = np.zeros(n_r, dtype=float)

        for _ in range(n_steps):
            dt_step = min(dt_h, total_h - t_curr)
            if dt_step <= 0.0:
                break
            dt_sec = dt_step * 3600.0

            # Compute local surface inclination angle theta_local(r)
            # theta_local = arctan(|d(depth)/dr|)
            if np.max(depth_m) > 1e-12:
                grad = np.gradient(depth_m, r_grid_m)
                grad[0] = 0.0  # Axisymmetric center symmetry
                abs_slope = np.abs(grad)
                theta_local = np.arctan(abs_slope)
                # Clip to physical grazing incidence cutoff (85 deg)
                theta_local = np.clip(theta_local, 0.0, math.radians(85.0))
            else:
                theta_local = np.zeros(n_r, dtype=float)

            # Sputter yield with SOTA Yamamura-Ruzic angular enhancement and roughness damping
            y_arr = np.array([
                calculate_sputter_yield_sota(v_d, self._material_obj, float(th), roughness_factor=rf)
                for th in theta_local
            ])

            # Local erosion rate: d(depth)/dt = J_i * Y / (e * n_target)
            rate_m_s = gamma_i * y_arr / self.n_target
            final_rate_m_s = rate_m_s

            depth_old = depth_m.copy()
            depth_m = depth_m + rate_m_s * dt_sec

            # Breakthrough check
            peak_d = float(np.max(depth_m))
            if peak_d >= self.initial_target_thickness_m and not breakthrough_occurred:
                breakthrough_occurred = True
                old_peak = float(np.max(depth_old))
                delta_peak = peak_d - old_peak
                frac = (
                    (self.initial_target_thickness_m - old_peak) / delta_peak
                    if delta_peak > 1e-15
                    else 1.0
                )
                breakthrough_time_h = t_curr + frac * dt_step

            if clamp_at_thickness:
                depth_m = np.minimum(depth_m, self.initial_target_thickness_m)

            t_curr += dt_step
            time_points.append(t_curr)
            depth_history.append(depth_m.copy())

        # 3. Calculate key summary metrics
        peak_depth_mm = float(np.max(depth_m) * 1e3)
        r_grid_mm = r_grid_m * 1e3

        # Groove width at FWHM
        fwhm_mm = self.calculate_fwhm(depth_m * 1e3, r_grid_mm)
        if fwhm_mm <= 0.0 and np.max(final_rate_m_s) > 0.0:
            fwhm_mm = self.calculate_fwhm(final_rate_m_s * 3.6e6, r_grid_mm)

        # Target utilization efficiency (%)
        utilization_pct = self.calculate_target_utilization(depth_m, r_grid_m)

        # Target lifetime (hours)
        if breakthrough_occurred and breakthrough_time_h is not None:
            lifetime_hours = breakthrough_time_h
        else:
            lifetime_hours = self.calculate_lifetime(
                peak_depth_mm=peak_depth_mm,
                elapsed_hours=total_h,
            )

        res = TargetErosionResult(
            r_grid_mm=r_grid_mm,
            r_grid_m=r_grid_m,
            depth_profile_mm=depth_m * 1e3,
            depth_profile_m=depth_m,
            depth_history_mm=np.array(depth_history) * 1e3,
            time_hours=np.array(time_points),
            erosion_rate_mm_hr=final_rate_m_s * 3.6e6,
            erosion_rate_m_s=final_rate_m_s,
            peak_erosion_depth_mm=peak_depth_mm,
            racetrack_fwhm_mm=fwhm_mm,
            target_utilization_efficiency_pct=utilization_pct,
            target_lifetime_hours=lifetime_hours,
            breakthrough_occurred=breakthrough_occurred,
            breakthrough_time_hours=breakthrough_time_h,
            material=self._material_name,
            discharge_voltage_v=v_d,
            initial_thickness_mm=self.initial_target_thickness_mm,
            target_radius_mm=self.target_radius_mm,
            roughness_factor=rf,
        )
        self.last_result = res
        return res
