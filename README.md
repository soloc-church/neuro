# neuro — Neural Time‑Series → Ising/QUBO

> _From spikes to spins._ Convert neural time‑series into rigorously constrained Ising/QUBO models, run parameterized QAOA on real hardware (qBraid/AWS, IonQ) with readout correction, and benchmark against classical baselines — in a reproducible, containerized workflow.

<p align="center">
  <img alt="neuro matte banner" src="docs/assets/hero_matte.svg" width="880">
</p>

<p align="center">
  <a href="https://github.com/soloc-church/neuro/actions"><img alt="CI" src="https://img.shields.io/badge/CI-setup-334155?style=for-the-badge&logo=github"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10_|_3.11-0f766e?style=for-the-badge&logo=python">
  <img alt="QAOA" src="https://img.shields.io/badge/QAOA-hardware_ready-4338ca?style=for-the-badge">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-reproducible-1f2937?style=for-the-badge&logo=docker">
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Examples](#examples)
- [Core Concepts](#core-concepts)
- [Benchmarks & Reproducibility](#benchmarks--reproducibility)

---

## Introduction

**neuro** provides clean, composable APIs to transform neural time‑series into **Ising/QUBO** instances with **2/3‑local** interactions and **k‑hot** constraints, then solve them with **QAOA** (via **qBraid/AWS** and **IonQ**) or classical baselines. The design emphasizes:

- **Separation of concerns**: `io → hamiltonian → solvers → reporting`.
- **Scientific clarity**: explicit locality, penalties, constraints, and metadata.
- **Reproducibility**: containerized runs and parameterized notebooks.

A small, professional aesthetic is applied throughout (matte neutrals with a restrained indigo/teal accent) to keep focus on the science while signaling modern engineering craft.

---

## Architecture

```mermaid
flowchart LR
    A[Neural Time‑Series] --> B[Preprocessing\n(binning • filtering)]
    B --> C[Feature Maps\n(co‑activation • windows)]
    C --> D[Hamiltonian Builder\nIsing/QUBO\n2/3‑local + k‑hot]
    D --> E{Solver}
    E --> E1[QAOA\n(qBraid/AWS, IonQ)]
    E --> E2[Classical Baselines]
    E1 --> F[Readout Correction]
    E2 --> F
    F --> G[Reporting\n(metrics • figs • tables)]
```

> **Design note**: modules remain small and orthogonal. Builders produce serializable Hamiltonians; solvers consume them interchangeably.

---

## Examples

### Minimal Python sketch

```python
from neuro.io import load_timeseries
from neuro.hamiltonian import IsingBuilder, constraints
from neuro.solvers import QAOASolver, SimulatedAnnealing
from neuro.reporting import BenchmarkReport

# 1) Load neural time‑series (CSV/Parquet/NumPy)
ts = load_timeseries("data/spikes.parquet", rate_hz=1000)

# 2) Build Ising with 3‑local terms and k‑hot constraint (k=5)
H = (IsingBuilder()
     .from_timeseries(ts, window_ms=50, coactivation="pearson")
     .add_local_terms(order=3)
     .add_constraint(constraints.k_hot(k=5, weight=3.0))
     .finalize())

# 3a) QAOA on hardware/simulator with readout correction
aq = QAOASolver(p=2, optimizer="cobyla", shots=2000, readout_correction=True)
res_q = aq.solve(H, backend="qbraid:aws:sv1")  # or ionq:ideal / local backends

# 3b) Classical baseline
sa = SimulatedAnnealing(steps=50000, schedule="geometric")
res_c = sa.solve(H)

# 4) Report
BenchmarkReport() \
  .add("QAOA(p=2)", res_q) \
  .add("SA",        res_c) \
  .to_markdown("reports/qaoa_vs_sa.md")
```

### CLI sketch (optional)

```bash
neuro ingest data/spikes.parquet --rate-hz 1000 --out data/ts.pkl
neuro build  data/ts.pkl --order 3 --k-hot 5 --out data/H.json
neuro solve  data/H.json --backend qbraid:aws:sv1 --qaoa-p 2 --shots 2000 --readout-correction
neuro bench  data/H.json --baselines sa,tabu --out reports/
```

> **Accent highlight**: QAOA runs include **readout correction** by default when enabled; calibration artifacts are tracked alongside results.

---

## Core Concepts

### Ising / QUBO with Structured Constraints
- **Locality**: explicit **2‑local** and **3‑local** couplings to capture higher‑order co‑activation.
- **k‑hot**: enforce exactly _k_ active spins via penalty terms (weight is tunable).
- **Regularization**: optional L1/L2 penalties to stabilize fits and avoid degeneracy.

### Solvers
- **QAOA**: depth `p`, mixer variants, shot budgets; optimizers such as **SPSA**, **COBYLA**, **Adam**.
- **Hardware**: **qBraid/AWS Braket** backends and **IonQ** access.
- **Classical**: simulated annealing, tabu/greedy, with restarts for robust baselines.

### Readout Correction
Mitigates measurement bias using calibration matrices per backend; matrices can be cached and reused across similar runs.

### Reporting
Standardized metrics (objective value, constraint violation), runtime, shot budgets, and calibration metadata; outputs to Markdown with figures.

---

## Benchmarks & Reproducibility

- **Reproduce** via containerized runs and parameterized notebooks.
- **Seeds** fixed across solvers; raw metrics emitted to `artifacts/` for traceability.
- **Sweeps**: scan QAOA depths, mixers, and optimizer settings; aggregate results and Pareto frontiers in a single report.

<!--
Matte palette reference (non-rendered):
- Neutral: #111827, #1F2937, #334155
- Accent‑indigo: #4338CA
- Accent‑teal:   #0F766E
-->
