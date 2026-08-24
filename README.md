# Symmetry-aware photonic-state measurement code

This repository contains the end-to-end code and raw inputs behind the data-bearing items requested from the paper:

- Main Figure 3
- Supplementary Figures 1, 8, 9, 10, and 11
- Supplementary Table 1

Each `run.py` starts from a Hamiltonian and a quantum state (or the raw detector counts), constructs the Compact or `sym_average` measurement estimator, and prints the numerical result as JSON. SG, Derand, AP, and standalone Pauli-baseline implementations are intentionally omitted.

## Repository layout

```text
inputs/
  hamiltonians/          Original Hamiltonians used by the paper
  states/                Experimental tomography density matrices
  experimental_counts/  Raw detector bit strings and Pauli-basis maps
src/compact_measurement/
  hamiltonian.py         Hamiltonian I/O and permutation twirling
  nonlinear.py           Two-copy expansion for tr(rho^2 H)
  measurement.py         OGM grouping and probability optimization
  estimator.py           Simulated and raw-experiment estimators
  sym_average.py         Measurement-compatible symmetry averaging
  variance.py            Exact OGM single-shot variance
main_fig3/run.py
si_fig1/run.py
si_fig8/run.py
si_fig9/run.py
si_fig10/run.py
si_fig11/run.py
si_table1/run.py
tests/
```

## Installation

Python 3.9 or newer is supported.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

All runners print JSON to standard output. To save a newly generated result outside version control, add for example `--output results/main_fig3.generated.json`. The `results/` directory and `*.generated.json` are ignored by Git.

## Method pipeline

### Compact

The Compact path is fully implemented, rather than loading a precomputed curve:

1. Load `H = sum_j alpha_j Q_j` from the input text file (`0=I`, `1=X`, `2=Y`, `3=Z`).
2. Construct the permutation-twirled observable. The ordered six-decimal serialization used by the original paper pipeline is retained because OGM tie-breaking depends on Pauli-row order.
3. Greedily construct overlapping qubit-wise-commuting measurement settings.
4. Optimize their probabilities with the same diagonal objective used by the paper, `sum_j alpha_j^2 / chi_j`, where `chi_j` is the probability that term `Q_j` is covered.
5. Allocate or sample measurement settings, read/simulate outcomes, pool every outcome that covers a term, and compute the estimator.
6. Compute RMSE over the requested independent repetitions or evaluate the exact OGM variance.

Core code: `hamiltonian.py`, `measurement.py`, `estimator.py`, and `variance.py`.

For the nonlinear panels, the code first twirls the physical 3-qubit `H_3`, then expands the resulting observable into the two-copy product-Pauli representation. Twirling the already expanded 6-qubit file would be incorrect and produces 480 terms instead of the 144 terms used in the paper.

### sym_average

`sym_average` keeps the OGM protocol designed for the original Hamiltonian fixed. For each measured setting, it groups covered terms by their permutation-orbit signature, averages all orbit members compatible with that setting, and reweights by the total coverage of the original term. This is implemented in `src/compact_measurement/sym_average.py` and is used only by `si_fig1/run.py`.

## Figure-by-figure provenance

### Main Figure 3

Entry point: `main_fig3/run.py`

```bash
python main_fig3/run.py
```

This is the raw experimental-count path. Each point uses 20 disjoint blocks from the relevant ZIP archive. RMSE is evaluated against the ideal symmetric-state reference, as in the main comparison.

| Panels | Observable/state | Hamiltonian input | State/count input |
|---|---|---|---|
| a-b | random 3-qubit `tr(rho H)`, W/GHZ | `inputs/hamiltonians/rand_H_3.txt` | `bin_W3.zip`, `bin_GHZ3.zip`, and matching basis CSV |
| c-d | 4-qubit spin `tr(rho H)`, W/GHZ | `inputs/hamiltonians/H_4.txt` | `bin_W4.zip`, `bin_GHZ4.zip`, and matching basis CSV |
| e-f | 3-qubit `tr(rho^2 H)`, W/GHZ | `inputs/hamiltonians/H_3.txt` | two independent slices of `bin_W3.zip` or `bin_GHZ3.zip` |

Default linear shot counts are `12, 45, 160, 572, 2038, 7259, 25848`; nonlinear panels use `12, 45, 160, 572, 2038, 7259`. Output rows contain the 20 estimates, their mean, standard deviation, RMSE, and uncovered-term count.

### Supplementary Figure 1

Entry point: `si_fig1/run.py`

```bash
python si_fig1/run.py
```

Input: `inputs/hamiltonians/H_8.txt` and the analytic 8-qubit GHZ state. The runner constructs and evaluates only:

- `Compact`: twirl `H_8`, design OGM for the compact Hamiltonian, then estimate it.
- `sym_average`: design OGM for the original `H_8`, hold that protocol fixed, then apply measurement-compatible orbit averaging.

The SG, Derand, AP, and baseline OGM curves from the paper are not generated.

### Supplementary Figure 8

Entry point: `si_fig8/run.py`

```bash
python si_fig8/run.py
```

This is the noiseless counterpart of Main Figure 3. It uses the same three Hamiltonians and analytic W/GHZ density matrices. Measurement outcomes are generated from the input state; no experimental ZIP or old data point is read. Only Compact results are returned.

### Supplementary Figure 9

Entry point: `si_fig9/run.py`

```bash
python si_fig9/run.py
```

Inputs are `H_8.txt`, `H_12.txt`, and `H_14.txt`, evaluated on analytic GHZ states. The runner independently regenerates each spin Hamiltonian in memory and verifies the checked-in file against the paper seeds before doing any measurement calculation:

| Qubits | Seed |
|---:|---:|
| 8 | 20260508 |
| 12 | 20260512 |
| 14 | 20260513 |

Only the Compact calculation is returned.

### Supplementary Figure 10

Entry point: `si_fig10/run.py`

```bash
python si_fig10/run.py
```

This uses the same raw detector archives and six panels as Main Figure 3. The error reference is instead the expectation value of the compact observable on the corresponding tomography matrix (`rho_W3.mat`, `rho_GHZ3.mat`, `rho_W4.mat`, or `rho_GHZ4.mat`), matching the noisy-state post-processing path.

### Supplementary Figure 11

Entry point: `si_fig11/run.py`

```bash
python si_fig11/run.py
```

Inputs are the three Hamiltonians and four tomography matrices. `variance.py` evaluates the full state-dependent OGM second moment, including joint coverage and `Tr(rho Q_j Q_k)`, then reports `Var^(T)=Var^(1)/T`. Only Compact is returned.

### Supplementary Table 1

Entry point: `si_table1/run.py`

```bash
python si_table1/run.py
```

The inputs and variance calculation are the same as Supplementary Figure 11. The runner prints only the Compact single-shot-variance column; it does not load the old table CSV.

## Input provenance

- `rand_H_3.txt`, `H_3.txt`, `H_4.txt`, `H_8.txt`, `H_12.txt`, and `H_14.txt` were copied from the paper workspace Hamiltonian/preprocessing directories.
- `rho_*.mat` are the experimental tomography outputs. The original fidelity-bearing filenames were normalized to ASCII names without changing file content.
- `bin_*.zip` are the original detector bit-string archives. They are read directly with `zipfile`; no archive member is extracted to disk.
- `pauli_*.csv` maps each detector file number to its physical local-Pauli basis.
- `CHECKSUMS.sha256` records all checked-in input hashes.


For a fast smoke test of the simulation/experimental runners, use `--quick`; it uses shots `12,45` and two repetitions. `--quick` is for code validation, not for paper-level statistics.
