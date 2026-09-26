# SputterTwin

**Physics-Informed AI Digital Twin for DC Magnetron Sputtering**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

SputterTwin is a scientific simulation and physics-informed machine learning platform for thin-film DC magnetron sputtering systems used in semiconductor and advanced materials manufacturing.

---

## Architecture Overview

SputterTwin couples analytical kinetic/plasma physics with deep surrogate models (Physics-Informed Neural Networks):

```
Operating Parameters (W, P, flow)
              │
              ▼
   ┌───────────────────────┐
   │ 1. Plasma Discharge   │ ──► Sheath Voltage V_d, Current I_d, Ion Flux Γ_i, Bohm Sheath
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │ 2. Sputter Yield      │ ──► Yamamura-Tawara Yield Y(E, θ) (Cu, Ti, Al)
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │ 3. Gas Transport      │ ──► Mean Free Path λ, Knudsen Kn, Thermalization & Scattering
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │ 4. Film Deposition    │ ──► 2D Wafer Thickness Map, Growth Rate (nm/min), Non-Uniformity (%)
   └───────────────────────┘
```

### Physics Modules (`sputtertwin/physics/`)
- **`plasma.py`**: Non-linear magnetron power-law discharge ($I_d = k P_{\text{eff}}^m V_d^n$), sheath dynamics, Bohm criterion for plasma density, secondary electron emission.
- **`sputter_yield.py`**: Yamamura-Tawara & Bohdansky formulation for $\text{Ar}^+$ bombardment on planar targets ($\text{Cu}$, $\text{Ti}$, $\text{Al}$), Kr-C nuclear stopping, threshold cutoff, and grazing angular distribution.
- **`transport.py`**: Gas kinetic theory, mean free path, Knudsen regime classification (ballistic / transition / continuum), Beer-Lambert unscattered transmission, and random-walk angular broadening.
- **`deposition.py`**: 2D circular wafer film profile simulation, racetrack geometry non-uniformity modeling, target groove erosion collimation, and thermalization transport efficiency. Calibrated to semiconductor fab benchmark conditions.

### PINN Modules (`sputtertwin/pinn/`)
- **`plasma_pinn.py`**: Deep surrogate neural network for the plasma discharge stage, trained under 4 physics conservation losses (Power Conservation, Magnetron I-V Scaling, Ion Current Continuity, Bohm Sheath Criterion).

---

## Installation

```bash
# Clone or navigate to the repository
cd sputter_twin

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### 1. Analytical Physics Simulation

```python
from sputtertwin.physics.deposition import simulate_deposition

# Run nominal deposition benchmark (220 W, 5.0 mTorr, Cu target, 80 mm distance)
result = simulate_deposition(
    power_w=220.0,
    pressure_mtorr=5.0,
    ar_flow_sccm=20.0,
    distance_mm=80.0,
    deposition_time_s=720.0,
    material="Cu",
    grid_size=25,
)

print(result.summary())
print(f"Mean Thickness: {result.mean_thickness_nm:.2f} nm")
print(f"Deposition Rate: {result.deposition_rate_nm_min:.2f} nm/min")
print(f"Non-Uniformity: {result.uniformity_percent:.2f} %")
```

### 2. Physics-Informed Neural Network (PINN) Surrogate

```python
from sputtertwin.pinn.plasma_pinn import PlasmaPINN, generate_plasma_dataset, train_plasma_pinn

# Train or evaluate the Plasma PINN surrogate
dataset = generate_plasma_dataset(n_power=20, n_pressure=20, n_flow=3)
model, history = train_plasma_pinn(dataset=dataset, verbose=True)

# Instant physical prediction (millisecond inference)
pred = model.predict(power_w=300.0, pressure_mtorr=5.0, ar_flow_sccm=20.0)
print(f"Predicted Voltage: {pred['voltage_v']:.1f} V")
print(f"Predicted Current: {pred['current_a']:.3f} A")
print(f"Power Conservation: {pred['power_calc_w']:.1f} W (target: 300 W)")
```

---

## Testing

Run the full automated test suite covering both physics models and PINN neural components:

```bash
pytest tests/ -v
```

---

## Generating Diagnostic Visualizations

Generate the complete 17-figure publication diagnostic plot suite:

```bash
python scripts/generate_plots.py
```
Output figures are saved to `plots/`.
