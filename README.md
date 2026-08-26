# Tachyon Multi-Messenger Search

Reproducibility code for the manuscript:

**A population-level matched-filter search for tachyonic arrival-time advances in gamma-ray bursts and their gravitational-wave counterparts**

**Author:** Vishakha Singh  
Department of Applied Sciences, National Institute of Technology Delhi, India

## Overview

This repository contains the simulation and analysis pipeline used to study hypothetical tachyonic propagation in multi-messenger transients. The code includes:

- cosmological time-of-flight kernels for tachyonic propagation;
- synthetic multi-messenger catalog generation;
- a generalised-least-squares matched-filter estimator;
- nuisance-parameter treatments for systematic spectral lags, jet-launch delays, and linear Lorentz-invariance violation;
- Huber-weighted robust fitting and event-level bootstrap validation;
- a gradient-boosted catalog-level classifier;
- generation of the figures and numerical values used in the manuscript;
- optional export of simulated data products.

## Repository structure

```text
tachyon-multimessenger-search/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── tachyon_search.py
├── figures/          # generated figures
├── data/             # optional exported CSV files
└── paper/            # generated numbers.tex for the manuscript
```

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the full Monte Carlo analysis:

```bash
python tachyon_search.py
```

Run a reduced version for a faster check:

```bash
python tachyon_search.py --quick
```

Run the analysis and also export CSV data products:

```bash
python tachyon_search.py --export
```

The script generates figures, numerical results, and manuscript macros. The exact runtime depends on the machine and selected mode.

## Main outputs

Typical outputs include:

- `figures/*.png` and `figures/*.pdf` — manuscript figures;
- `results.json` — numerical results from the analysis;
- `paper/numbers.tex` — LaTeX macros containing values used in the manuscript;
- `data/*.csv` — optional exported simulation data when `--export` is used.

## Reproducibility

The analysis uses fixed random seeds in the simulations where required so that the main reported results and figures can be regenerated from the code.

## Citation

If you use this code, please cite the associated manuscript:

> Vishakha Singh, *A population-level matched-filter search for tachyonic arrival-time advances in gamma-ray bursts and their gravitational-wave counterparts*.

A DOI and journal citation can be added here after publication.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

Vishakha Singh  
Department of Applied Sciences  
National Institute of Technology Delhi  
Email: vishakha@nitdelhi.ac.in
