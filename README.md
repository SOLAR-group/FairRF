# FairRF

This repository contains the code for the paper _"FairRF: Multi-Objective Search for Single and Intersectional Software Fairness"_, accepted for publication at ICSE-SEIS 2026.

## Repository Structure

- `analysis`: Contains the raw data and the jupyter notebooks used for the analysis and the generation of the plots.
- `baseline`: Contains the code to run the base classification models (RQ2).
- `baseline_intersectional`: Contains the code to run the FairHOME approach for intersectional fairness (RQ4).
- `bias_mitigation_methods`: Contains the results of the state-of-the-art bias mitigation methods used for comparison (RQ3).
- `fair_rf`: Contains the implementation of FairRF and its variants (RQ1).
- `fair_rf_multi_attribute`: Contains the implementation of FairRF for intersectional fairness (RQ4).
- `random_search`: Contains the implementation of the random search algorithm used as a baseline for comparison (RQ2).

In all folders, we provide a `run_all.sh` script to reproduce the experiments.

## Installation

Install Python 3.9 or higher.

### Conda

To create the conda environment, run:

```bash
conda env create -f environment.yml
conda activate mutrf
```

### Pip

Alternatively, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Datasets

Follow the instructions reported here <https://github.com/Trusted-AI/AIF360/tree/7c4f172f4a81b1d14d19400e95c468aaf3134952/aif360/data> to download the datasets used in the paper.


## Citation Request

Please cite our paper if you use FairRF in your work

```bibtex
@inproceedings{d2026fairrf,
  title={FairRF: Multi-Objective Search for Single and Intersectional Software Fairness},
  author={d'Alosio, Giordano and Hort, Max and Moussa, Rebecca and Sarro, Federica},
  booktitle={2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE-SEIS '26), April 12--18, 2026, Rio de Janeiro, Brazil},
  year={2026}
}
```
