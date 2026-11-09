# AS-HMM-BO

### Adaptive-State Hidden Markov Model-Guided Bayesian Optimization for Deep Learning Image Segmentation

**Author:** Sourav Guha  
**Primary implementation:** Python, PyTorch, Jupyter Notebook  
**Dataset:** Warwick-QU / GlaS (Gland Segmentation in Colon Histology Images)

AS-HMM-BO is an adaptive hyperparameter-optimization research project for deep-learning-based image segmentation. It combines Bayesian Optimization (BO) with Hidden Markov Model (HMM)-based estimation of the optimizer's evolving state. The estimated state is used to guide the search policy, while multi-fidelity evaluation is intended to manage training cost.

> **Research status and reproducibility:** This repository contains the research proposal and the supplied experimental notebook. Notebook outputs have been cleared and machine-specific paths have been made project-relative. No dataset, trained checkpoint, or independently verified final performance result is included. Run the notebook with the dataset and required compute environment to reproduce experiments; do not interpret planned methodology or expected contributions as measured results.

## Core idea

Conventional Bayesian optimization uses a surrogate model and acquisition function to propose configurations. AS-HMM-BO additionally models the *behavior of the optimization process* over time. An HMM estimates a latent state from optimization-history features, and an adaptive controller uses that estimate to influence subsequent candidate generation.

The proposed state space contains:

- **Exploration** — broaden search diversity and investigate uncertain regions.
- **Exploitation** — refine promising regions.
- **Stagnation** — diversify or expand the search after limited progress.
- **Overfitting** — favor configurations with better validation generalization.
- **Convergence** — verify explicit stopping criteria and reduce unnecessary search.

The HMM guides Bayesian Optimization; it does not replace the Bayesian surrogate or acquisition function.

## Workflow

1. Inspect the GlaS dataset and validate image/annotation pairs.
2. Build binary gland masks and apply consistent preprocessing.
3. Create reproducible patient-aware splits and cross-validation folds.
4. Train and evaluate a compact U-Net segmentation model.
5. Extract optimization feedback such as improvement, variance, surrogate uncertainty, search diversity, generalization gap, and compute cost.
6. Estimate the latent optimization state with the HMM.
7. Adapt candidate generation and evaluate candidates under controlled budgets.
8. Compare AS-HMM-BO with Random Search, TPE, rule-based adaptation, and state-independent adaptation; run ablations.
9. Freeze the selection protocol and evaluate the final model on an independent test set.

## Repository layout

```text
AS-HMM-BO/
├── configs/                  # Reserved for experiment configuration files
├── data/
│   └── raw/                  # Place the dataset here (not committed)
├── docs/
│   └── FINAL_PROPOSAL.md     # Full research proposal and methodology
├── notebooks/
│   └── AS-HMM-BO_Research.ipynb
├── outputs/                  # Generated metrics, histories, figures, checkpoints (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

The notebook currently holds the implementation and experiment workflow in one place. The modular directories are provided for future extraction of reusable modules.

## Dataset setup

This project notebook expects the Warwick QU / GlaS dataset at:

```text
data/raw/Warwick_QU_Dataset/
```

The notebook expects the dataset's image/annotation files and `Grade.csv` metadata in the dataset directory. Obtain the dataset from its official academic distribution and review its terms of use. **The dataset is not included in this repository.** See [`data/raw/README.md`](data/raw/README.md) for the expected layout.

## Environment setup

Python 3.10 or 3.11 is recommended. A CUDA-enabled PyTorch installation is optional; the notebook falls back to CPU, although model training will be slower.

```bash
# Clone the repository
git clone https://github.com/<your-github-username>/AS-HMM-BO.git
cd AS-HMM-BO

# Create and activate a virtual environment (Windows PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux:
# python3.11 -m venv .venv
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU support, install the PyTorch build that matches your CUDA environment by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

Launch Jupyter:

```bash
jupyter lab
```

Open `notebooks/AS-HMM-BO_Research.ipynb` and run the notebook from the first cell after placing the dataset in the expected location.

## Experiments and evaluation

The notebook includes dataset audits, U-Net training, patient-level validation/cross-validation, HMM-guided search development, controlled optimizer comparisons, and ablation/diagnostic analysis. The proposal describes the intended experimental protocol and comparisons.

Recommended reporting includes:

- Dice score and Intersection over Union (IoU)
- Precision and recall
- Best-so-far validation Dice against trial count and elapsed time
- Area under the optimization curve (AUC)
- Runtime and computational budget
- Mean and variability across random seeds
- Paired comparisons and ablation results

Use identical data partitions, search spaces, model definitions, fidelity budgets, and stopping rules for optimizer comparisons. Keep the independent test set out of hyperparameter optimization and model selection. Report only results produced by completed, reproducible runs.

## Research documentation

See [`docs/FINAL_PROPOSAL.md`](docs/FINAL_PROPOSAL.md) for the complete framework description, mathematical formulation, workflow, experimental controls, ablations, and reproducibility protocol.

## Author

**Sourav Guha**

## Disclaimer

This repository is a research/academic project. The proposed benefits of AS-HMM-BO must be established through controlled experiments; they are not guaranteed by the framework design alone.
