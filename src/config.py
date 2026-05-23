"""
config.py — Simulation constants for:
"Mission survivability of an ISR UAV in a stochastic contested radar environment"

All physical quantities use an abstract 2D spatial unit system.
Time is discretised into steps of "STEP_DURATION_S" seconds.

Spatial reference:
  - 1 spatial unit ≈ 1 km (approximate, for physical intuition only)
  - Grid: 150 × 150 units  →  ~150 × 150 km operational area
  - Drone operational range: ~100 km  →  consistent with grid scale

Temporal reference:
  - 1 step = 30 seconds of simulated flight
  - MAX_STEPS = 500  →  ~4.2 hours per mission
  - Drone max endurance: 16h  →  1920 steps absolute max
"""

# ─── Space ─────────────────────────────────────────────────────────────────────

GRID_SIZE       = 150        # side length of the square simulation domain [units]

UAV_START       = (5.0, 5.0)          # Start position [units]
MISSION_TARGET  = (145.0, 145.0)      # mission objective position [units]
TARGET_RADIUS   = 5.0                 # mission success if UAV reaches within this radius [units]

# ─── Time ──────────────────────────────────────────────────────────────────────

STEP_DURATION_S = 30         # real-world duration of one simulation step [seconds]
MAX_STEPS       = 500        # maximum steps per trajectory (~4.2 h mission)

# ─── UAV movement ──────────────────────────────────────────────────────────────

UAV_STEP_SIZE   = 0.65       # mean displacement per step [units]
                             # at 30s/step → 0.65 km/step → ~78 km/h ground speed
                             # (AR3 cruise: ~90 km/h; 0.65 accounts for heading
                             #  corrections within each 30s interval)

DRIFT_WEIGHT    = 0.65       # [0, 1] — how strongly the UAV biases toward the target
                             # 1.0 = straight line; 0.0 = pure random walk

# Lévy flight parameters (used in Stage 3 variant)
LEVY_ALPHA      = 1.5        # stability index μ [1, 2]; 2 → Gaussian, 1 → Cauchy
                             # literature value for animal search: 1.4–2.0
                             # (Humphries et al., Nature 465, 2010)
LEVY_S_MIN      = 0.1        # minimum step size for Lévy sampling [units]
LEVY_S_MAX      = 5.0        # maximum step size [units] → ~600 km/h physical ceiling

# ─── Radar network ─────────────────────────────────────────────────────────────

N_RADARS_DEFAULT    = 20     # number of radar nodes in the network
RADAR_RANGE         = 10.0   # base detection radius [units] (~10 km)
RADAR_RANGE_MIN     = 5.0    # minimum for parameter sweeps [units]
RADAR_RANGE_MAX     = 20.0   # maximum for parameter sweeps [units]

# Detection probability model: P(d) = exp(-d / lambda) + noise
DETECTION_LAMBDA    = 6.0    # decay length [units]; controls how steeply P drops with distance
DETECTION_NOISE_STD = 0.05   # std of additive Gaussian noise on P(d); models sensor imperfection

# Radar density sweep (Stage 4 — phase transition)
RHO_MIN         = 0.001      # min radar density [radars / unit²]
RHO_MAX         = 0.015      # max radar density [radars / unit²]
RHO_STEPS       = 30         # number of density values to sweep

# ─── Network topologies (Stage 2) ──────────────────────────────────────────────

# Erdős–Rényi
ER_P            = 0.15       # edge probability

# Barabási–Albert (scale-free)
BA_M            = 2          # edges added per new node

# Watts–Strogatz (small-world)
WS_K            = 4          # each node connected to K nearest neighbours
WS_BETA         = 0.3        # rewiring probability

# ─── Alert propagation — connected radar network ───────────────────────────────

ALERT_GAMMA     = 0.5        # alert amplification factor γ [0, 1]
                             # P_eff = clip(P_base * (1 + γ * A_i), 0, 1)
                             # γ = 0.5 → alerted radar has up to 1.5× base sensitivity
ALERT_TAU       = 10         # alert duration [steps] → 5 minutes of heightened sensitivity
                             # after a neighbour detects the UAV

# ─── Passive sensing (Stage 5 — SpectraLoc model) ──────────────────────────────

PASSIVE_RANGE   = 6.0        # radius within which UAV passively detects a radar [units]
                             # shorter than RADAR_RANGE: the drone sees less than it is seen
PASSIVE_BETA    = None       # evasion weight β — calibrated empirically in Stage 5

# ─── Ensemble / Monte Carlo ────────────────────────────────────────────────────

N_TRAJECTORIES  = 1000       # trajectories per condition (statistical robustness)
N_PHASE_RUNS    = 500        # trajectories per density value in phase transition sweep
RANDOM_SEED     = 42         # global seed for reproducibility (set None to disable)

# ─── Output paths ──────────────────────────────────────────────────────────────

import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
DOCS_DIR        = os.path.join(BASE_DIR, "docs")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")

# Ensure output directories exist at import time
os.makedirs(FIGURES_DIR, exist_ok=True)
