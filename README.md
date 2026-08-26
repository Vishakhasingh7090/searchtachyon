Tachyonic Arrival-Time Search in Multi-Messenger Astrophysics

Vishakha Singh
Department of Applied Sciences, National Institute of Technology Delhi

Overview

This project investigates tachyonic (superluminal) propagation through the arrival times of multi-messenger astrophysical transients.

The central idea is to test whether a hypothetical tachyonic signal produces an energy-dependent arrival-time advance relative to a luminal reference. The framework combines cosmological propagation, simulated multi-messenger catalogs, generalized least-squares (GLS) estimation, robustness tests, and an independent gradient-boosted machine-learning classifier.

The complete computational workflow is implemented in Python and can be run from a single entry point.

────────

Scientific Idea

For a tachyonic energy scale (E_t), the propagation model predicts an arrival-time advance that approximately scales as

[
\Delta t \propto E_t^2 E^{-2} I_2(z),
]

where (E) is the observed energy and (I_2(z)) is the cosmological propagation kernel.

The (E^{-2}) dependence makes the lowest-energy, well-timed observations particularly important. The analysis therefore combines MeV, GeV, TeV, neutrino, and gravitational-wave timing information to determine whether a common propagation signature can be distinguished from source-intrinsic timing variations.

────────

What the Code Contains

The main program is:

```text
tachyon_search.py
```

It contains the complete workflow in the following order:

1. Cosmological propagation
  • Flat-(\Lambda)CDM cosmology
  • Hubble expansion function
  • (I_1(z)) and (I_2(z)) propagation kernels
  • Tachyonic group-velocity excess
  • Exact and leading-order arrival-time calculations
  • Linear Lorentz-invariance-violation (LIV) propagation template
2. Multi-messenger catalog simulation
  • MeV photons
  • GeV photons
  • TeV photons
  • High-energy neutrinos
  • Optional gravitational-wave anchoring
  • Intrinsic spectral-lag scatter
  • Timing uncertainties
  • Redshift distribution and redshift uncertainty
  • Optional systematic spectral lag
  • Optional jet-launch delay
  • Optional heavy-tailed lag distributions
  • Optional tachyon and LIV signal injection
3. Generalized least-squares matched filter
  • Event-level covariance treatment
  • Estimation of (\theta = E_t^2)
  • Statistical uncertainty
  • One-sided upper limits
  • Nuisance-template fitting
  • Joint tachyon + LIV fitting
4. Robust estimation
  • Huber-weighted fitting
  • Reduced-(\chi^2) error rescaling
  • Protection against non-Gaussian/heavy-tailed timing residuals
5. Machine-learning validation
  • Catalog-level feature extraction
  • Histogram Gradient Boosting classifier
  • Stratified cross-validation
  • ROC curves
  • AUC evaluation at different injected tachyon scales
6. Validation and sensitivity tests
  • Injection-recovery linearity
  • Catalog-size scaling
  • Frequentist coverage
  • Event-level bootstrap
  • Information-budget calculation
  • Systematic spectral-lag test
  • Jet-launch-delay test
  • Heavy-tail test
  • Redshift-error test
  • Gravitational-wave anchoring test
  • Tachyon/LIV confusion test
  • GW170817/GRB 170817A single-event constraint
7. Reproducible outputs
  • Numerical results in JSON
  • Numerical values for LaTeX through numbers.tex
  • Optional CSV exports
  • Automatically generated plots

────────

Requirements

Use Python 3.9 or later.

Required packages:

```text
numpy
scipy
matplotlib
scikit-learn
```

Install them with:

```bash
pip install numpy scipy matplotlib scikit-learn
```

If a requirements.txt file is included, you can instead run:

```bash
pip install -r requirements.txt
```

────────

Recommended Repository Structure

```text
tachyon-multimessenger-search/
│
├── README.md
├── tachyon_search.py
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── figures/
│   └── generated plots
│
└── data/
    └── exported numerical results
```

The figures/ and data/ directories are created automatically when required by the workflow.

────────

How to Run

1. Clone the repository

```bash
git clone <repository-url>
cd tachyon-multimessenger-search
```

Replace <repository-url> with the URL of this GitHub repository.

2. Create a virtual environment

Recommended:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run a quick test first

```bash
python tachyon_search.py --quick
```

This uses reduced Monte Carlo settings and is the best way to confirm that the environment and dependencies are working correctly.

5. Run the complete analysis

```bash
python tachyon_search.py
```

The full run performs the larger Monte Carlo calculations and writes the principal numerical and graphical outputs.

6. Export the numerical data

```bash
python tachyon_search.py --export
```

This performs the run and additionally exports the underlying numerical results to CSV/JSON files in the data/ directory.

────────

Recommended Workflow

For a new user, proceed in this order:

```text
Install Python
      ↓
Create virtual environment
      ↓
Install requirements
      ↓
Run --quick
      ↓
Check that the run finishes successfully
      ↓
Inspect generated outputs
      ↓
Run the complete analysis
      ↓
Run --export when numerical tables are required
      ↓
Modify one physical/simulation parameter at a time
      ↓
Re-run and compare the resulting statistics
```

Do not begin by changing several simulation parameters simultaneously. First reproduce the default run. This provides a reference point for checking whether later modifications behave as expected.

────────

Main Simulation Settings

The default configuration is controlled by the SimConfig dataclass in tachyon_search.py.

Important parameters include:

• n_events — number of simulated transients
• E_t_inject_meV — injected tachyonic energy scale
• sigma_lag — intrinsic timing-lag scatter
• lag_distribution — Gaussian or Student-(t)
• anchored_fraction — fraction with an external gravitational-wave clock
• systematic_lag_per_decade — systematic energy-dependent lag
• jet_delay_mean and jet_delay_sigma — anchored-event emission-delay model
• redshift_error_frac — fractional redshift uncertainty
• liv_linear_coeff — optional linear-LIV injection
• seed — random seed for reproducibility

Change these parameters carefully and record the values used for every experiment.

────────

Understanding the Main Analysis Sequence

Step 1 — Build the cosmological propagation model

The code evaluates the redshift-dependent propagation kernels and computes the expected tachyonic arrival-time advance.

Step 2 — Simulate a transient population

A population of astrophysical events is generated with redshifts, available observing bands, timing noise, and intrinsic source lags.

Step 3 — Inject a propagation signal when required

A chosen (E_t) can be injected into the simulated population. A linear-LIV contribution can also be injected independently to test signal confusion.

Step 4 — Construct event-level timing differences

For unanchored events, timing is measured relative to the highest-energy available reference band. Anchored events use an external reference clock.

Step 5 — Construct the covariance

The reference band contributes correlated uncertainty to timing differences from the same event. The GLS calculation therefore uses a block covariance rather than treating every timing point as statistically independent.

Step 6 — Estimate the tachyonic parameter

The matched filter estimates

[
\theta = E_t^2
]

and its uncertainty. A physical one-sided upper limit on (E_t) is then obtained.

Step 7 — Add nuisance parameters when necessary

The same GLS framework can simultaneously fit effects such as:

• systematic spectral lag,
• anchored-event delay,
• linear LIV.

This is important because an unmodelled timing effect can bias the recovered tachyonic parameter.

Step 8 — Validate the estimator

Injection-recovery, bootstrap, coverage, heavy-tail, redshift-error, and catalog-scaling experiments test whether the inferred limits remain statistically reliable.

Step 9 — Perform the independent ML check

A gradient-boosted classifier uses catalog-level timing/template statistics to distinguish null catalogs from tachyon-injected catalogs. Its ROC/AUC behavior provides an independent comparison with the parametric matched-filter sensitivity.

────────

Output Files

A standard run produces numerical outputs such as:

```text
results.json
numbers.tex
```

With --export, the data/ directory can additionally contain outputs including:

```text
kinematics_vs_energy.csv
kernels_I1_I2.csv
advance_vs_z_1MeV_1meV.csv
advance_by_band_1meV.csv
fiducial_catalog_seed42.csv
fiducial_fit.json
fiducial_bootstrap_theta.csv
classifier_auc.csv
scaling_UL_vs_N.csv
coverage.csv
meanlag_bias.csv
jet_delay.csv
heavy_tails.csv
redshift_error.csv
anchored_fraction.csv
liv_confusion.csv
gw170817_bound_vs_energy.csv
results_all.json
```

These files allow individual stages of the analysis to be inspected without relying only on terminal output.

────────

Reproducibility Guidelines

To obtain reproducible results:

• Use the same Python/package versions whenever possible.
• Keep the random seed fixed when reproducing a specific experiment.
• Run --quick before a full computation after changing the environment.
• Change only one physical assumption at a time when testing robustness.
• Keep units consistent: energies are generally in eV and times in seconds internally.
• Remember that the fitted parameter is (\theta=E_t^2), not (E_t) directly.
• Do not interpret a single stochastic realization as the expected sensitivity; use the ensemble/Monte Carlo results.
• Preserve the event-level covariance when modifying the timing model.
• When introducing an additional propagation or source effect, consider whether it requires an additional nuisance template in the GLS design matrix.

────────

Extending the Framework

The current implementation can be extended to:

• alternative cosmological models,
• different transient redshift distributions,
• additional electromagnetic energy bands,
• different neutrino energies,
• alternative intrinsic-lag distributions,
• different gravitational-wave anchoring fractions,
• event-specific timing uncertainties,
• real multi-messenger catalogs,
• additional propagation hypotheses,
• alternative machine-learning classifiers.

When adding a new effect, first implement it in the simulation, then determine whether the fitting model also needs a corresponding nuisance or signal template. Finally, test recovery with controlled injections before applying it to a heterogeneous dataset.

────────

Citation

If this repository contributes to your research, please cite:

Vishakha Singh, “A Population-Level Matched-Filter Search for Tachyonic Arrival-Time Advances in Gamma-Ray Bursts and Their Gravitational-Wave Counterparts.”

Repository author: Vishakha Singh, National Institute of Technology Delhi.

────────

Contact

Vishakha Singh
Department of Applied Sciences
National Institute of Technology Delhi
Delhi, India
