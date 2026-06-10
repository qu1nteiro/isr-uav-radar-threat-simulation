# ISR UAV Radar Threat Simulation — Run Guide

WARNING: The full simulation takes approximately 2 hours to complete. Stages 4 and 5 are
computationally intensive and will use a large amount of CPU. Do not use your computer for
other tasks while the simulation is running, as it may slow down the simulation significantly
or cause instability.

---

## Requirements

- Python 3.10 or higher
- pip

To check your Python version:

```
python --version
```

---

## Installation

Extract the zip file and navigate to the project folder:

```
cd isr-uav-radar-threat-simulation-main
```

Install the required dependencies:

```
pip install -r requirements.txt
```

---

## Running the Simulation

All scripts must be run from the `src` directory:

```
cd src
```

Run each script in the order listed below. Wait for each one to finish before starting the next.

Output figures are saved automatically to `results/figures/`.

---

### Stage 1 — Radar Detection Model

```
python radar_detection.py
```

Validates the radar detection model and generates a characterisation figure.

---

### Stage 2 — Radar Network Construction

```
python network_builder.py
```

Places radars on the grid and builds ER, BA, and WS network topologies. Validates network metrics.

---

### Stage 3 — UAV Trajectory Simulation

```
python uav_trajectory.py
```

Simulates solo UAV mission trajectories across all three network topologies using Gaussian and
Levy random walks.

---

### Stage 3b — ELINT UAV Trajectory Simulation

```
python uav_elint_trajectory.py
```

Same as Stage 3 but with an ELINT-equipped UAV that progressively learns radar positions during
the mission.

---

### Stage 4 — Phase Transition Analysis

```
python phase_transition.py
```

Runs a large sweep of radar density values across all topologies and walk types to identify the
survivability phase transition. This stage runs tens of thousands of individual simulations and
takes a long time to complete.

---

### Stage 5 — Cooperative ELINT Mission

```
python cooperative_elint.py
```

Simulates cooperative missions with two UAVs sharing ELINT information in real time. Runs 500
missions per parameter combination across multiple radar densities, topologies, and walk types.
This stage takes a long time to complete.

---

### Mission Animation

```
python mission_animation.py
```

Generates animated figures of solo UAV missions for both Gaussian and Levy walks.

---

### Visualization

```
python visualization.py
```

Generates a mission overview figure showing multiple trajectories across the simulation grid.

---

### Additional Analysis

```
python aditinonal_analysis.py
```

Runs supplementary analyses including Levy distribution verification and enriched network
metrics (clustering coefficient and Pearson correlation).

---

## Summary Table

| Order | Script                    | Stage       |
|-------|---------------------------|-------------|
| 1     | radar_detection.py        | Stage 1     |
| 2     | network_builder.py        | Stage 2     |
| 3     | uav_trajectory.py         | Stage 3     |
| 4     | uav_elint_trajectory.py   | Stage 3b    |
| 5     | phase_transition.py       | Stage 4     |
| 6     | cooperative_elint.py      | Stage 5     |
| 7     | mission_animation.py      | Animation   |
| 8     | visualization.py          | Figures     |
| 9     | aditinonal_analysis.py    | Extra       |

---

## Output

All figures are saved to:

```
results/figures/
```

The folder is created automatically when any script is run.
