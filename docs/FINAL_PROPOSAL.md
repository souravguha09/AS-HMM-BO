# AS-HMM-BO
## Adaptive-State Hidden Markov Model-Guided Bayesian Optimization for Image Segmentation

AS-HMM-BO is a general-purpose adaptive hyperparameter optimization framework designed for deep-learning-based image segmentation.

The framework combines Bayesian Optimization (BO), Hidden Markov Models (HMMs), adaptive optimization strategies, and multi-fidelity evaluation to improve the efficiency of hyperparameter search.

Unlike conventional optimization approaches that primarily depend on objective-function observations, AS-HMM-BO additionally models the evolving behavior of the optimization process. It estimates the current optimization state and dynamically adapts its search strategy according to the observed progress.

The primary objective is to balance exploration, exploitation, optimization stability, generalization, and computational efficiency when searching for effective segmentation configurations.

---

# 1. Complete AS-HMM-BO Workflow

The following flowchart describes the complete optimization pipeline, from dataset preparation to final model evaluation.

```text
                         START
                           |
                           v
               Load Segmentation Dataset
                           |
                           v
                Validate Images and Masks
                           |
                           v
                    Data Preprocessing
                 - Resize images and masks
                 - Normalize input images
                 - Validate image-mask pairs
                           |
                           v
              Training and Validation Split
                           |
                           v
                Define Search Space
                 - Learning rate
                 - Weight decay
                 - Batch size
                 - Architecture parameters
                 - Loss function parameters
                           |
                           v
                Initialize AS-HMM-BO
                 - Initialize Bayesian surrogate
                 - Initialize HMM
                 - Define optimization states
                 - Configure adaptive strategies
                           |
                           v
             Generate Initial Candidate Configurations
                           |
                           v
                 Initial Candidate Evaluation
                 - Low-fidelity training
                 - Validation metric calculation
                 - Runtime measurement
                           |
                           v
                 Update Optimization History
                           |
                           v
              Extract Optimization State Features
                 - Improvement magnitude
                 - Best-so-far performance
                 - Performance variance
                 - Surrogate uncertainty
                 - Training-validation gap
                 - Recent unsuccessful trials
                           |
                           v
                 Construct HMM Observation
                           |
                           v
                   HMM State Estimation
                 - Transition probability update
                 - Observation likelihood
                 - State probability calculation
                 - Current state estimation
                           |
                           v
                 Identify Optimization State
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      EXPLORATION      EXPLOITATION      STAGNATION
          |                |                |
          v                v                v
      Increase          Focus on         Expand or
      Search            Promising        Restart
      Diversity         Regions          Search
          |                |                |
          +----------------+----------------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
      OVERFITTING                       CONVERGENCE
          |                                 |
          v                                 v
      Prefer Better                    Check Stopping
      Generalization                   Conditions
          |                                 |
          +----------------+----------------+
                           |
                           v
                  Adaptive Action Selection
                           |
                           v
                 Bayesian Candidate Generation
                           |
                           v
                  Select Next Configuration
                           |
                           v
                  Multi-Fidelity Evaluation
                 - Low fidelity
                 - Medium fidelity
                 - High fidelity
                           |
                           v
                 Calculate Validation Metrics
                           |
                           v
                 Update Bayesian Surrogate
                           |
                           v
                 Update Optimization History
                           |
                           v
                  Stopping Condition Met?
                           |
                    +------+------+
                    |             |
                   NO            YES
                    |             |
                    v             v
             Return to HMM    Select Best Validated
             State Estimation Configuration
                                  |
                                  v
                         Train Final Model
                                  |
                                  v
                       Freeze Model and Protocol
                                  |
                                  v
                      Independent Test Evaluation
                                  |
                                  v
                       Calculate Final Metrics
                                  |
                                  v
                        Compare Optimization
                             Algorithms
                                  |
                                  v
                          Analyze Results
                                  |
                                  v
                                 END
```

---

# 2. Core Components

AS-HMM-BO consists of five major components.

| Component | Responsibility |
|---|---|
| Bayesian Optimization | Generates promising hyperparameter configurations |
| Hidden Markov Model | Estimates the current optimization state |
| Adaptive Action Selection | Adjusts the search strategy based on the estimated state |
| Multi-Fidelity Evaluation | Allocates computational resources to candidate configurations |
| Optimization History | Stores trial results and provides feedback for subsequent decisions |

These components operate together to create an adaptive optimization loop.

---

# 3. Dataset Preparation

The framework begins by loading and validating the segmentation dataset.

Each dataset sample consists of:

- Input image.
- Corresponding ground-truth segmentation mask.
- Sample identifier.
- Optional patient or group identifier.

### Preprocessing

The preprocessing stage may include:

1. Image resizing.
2. Mask resizing using appropriate interpolation.
3. Image normalization.
4. Mask binarization or class encoding.
5. Image-mask correspondence validation.
6. Dataset integrity checks.

The preprocessing pipeline must remain consistent across optimization methods.

### Dataset Splitting

The dataset is divided into training and validation partitions.

- Training data is used to update model parameters.
- Validation data is used to evaluate candidate configurations.
- Independent test data is reserved for final evaluation.

Where multiple samples belong to the same patient, group-aware splitting should be used to avoid patient-level data leakage.

**Important:** Test data must never influence hyperparameter optimization, HMM state estimation, threshold selection, or model selection.

---

# 4. Hyperparameter Search Space

AS-HMM-BO searches for an effective combination of model and training hyperparameters.

A candidate configuration can be represented as:

\[
\theta =
\{
\eta,\lambda,B,C,\alpha,p,\ldots
\}
\]

Where:

| Parameter | Description |
|---|---|
| \(\eta\) | Learning rate |
| \(\lambda\) | Weight decay |
| \(B\) | Batch size |
| \(C\) | Model width or base channels |
| \(\alpha\) | Loss-function weighting parameter |
| \(p\) | Dropout probability |

Additional parameters can be incorporated depending on the segmentation architecture.

The search space should support:

- Continuous parameters.
- Integer parameters.
- Categorical parameters.
- Logarithmic parameter ranges.
- Conditional parameters.

The search-space definition is modular and should not be restricted to a particular dataset or segmentation architecture.

---

# 5. Initial Candidate Generation

Before adaptive optimization starts, the framework generates an initial collection of candidate configurations.

These candidates provide the first observations required by the Bayesian surrogate and HMM.

Initial sampling can use:

- Random sampling.
- Latin Hypercube Sampling.
- Prior configurations.
- Other suitable initialization strategies.

Every candidate is evaluated using a predefined initial computational budget.

The resulting observations are stored in the optimization history.

---

# 6. Multi-Fidelity Evaluation

Training every candidate for the maximum number of epochs can be computationally expensive.

AS-HMM-BO supports multi-fidelity evaluation to allocate resources according to candidate performance.

A candidate can initially receive a limited training budget. Promising configurations can subsequently receive additional resources.

### Fidelity Levels

| Fidelity | Evaluation |
|---|---|
| Low | Limited training budget |
| Medium | Increased training budget |
| High | Extended training budget |

Depending on the experiment, fidelity can be defined using training epochs, iterations, dataset subsets, or input resolution.

### Evaluation Process

1. Generate a candidate configuration.
2. Train the model using a low-fidelity budget.
3. Calculate validation performance.
4. Determine whether the candidate should receive additional resources.
5. Promote promising candidates to higher fidelity.
6. Stop or prune candidates that do not satisfy the promotion criteria.
7. Store the evaluation results.

The promotion policy must be consistent across comparative experiments.

Multi-fidelity evaluation should also account for the possibility that some configurations learn more slowly than others.

---

# 7. Optimization Feedback and Feature Extraction

After each evaluation, the framework extracts information describing the current behavior of the optimization process.

This information is used as the observation input for the HMM.

Possible features include:

| Feature | Purpose |
|---|---|
| Current validation performance | Measures candidate quality |
| Best-so-far performance | Tracks overall optimization progress |
| Improvement magnitude | Measures recent progress |
| Improvement frequency | Measures consistency |
| Performance variance | Measures optimization instability |
| Surrogate uncertainty | Measures prediction uncertainty |
| Exploration ratio | Measures search diversity |
| Generalization gap | Helps identify possible overfitting |
| Recent unsuccessful trials | Helps identify stagnation |
| Computational cost | Measures resource consumption |

A general observation vector can be represented as:

\[
z_t =
[
\Delta f_t,
\overline{\Delta f}_t,
\sigma_t,
u_t,
r_t,
g_t,
c_t
]
\]

Where:

- \(\Delta f_t\): Recent improvement.
- \(\overline{\Delta f}_t\): Average recent improvement.
- \(\sigma_t\): Performance variability.
- \(u_t\): Surrogate uncertainty.
- \(r_t\): Search diversity measurement.
- \(g_t\): Generalization-gap measurement.
- \(c_t\): Computational-cost information.

The exact feature definitions, normalization procedures, and observation windows should be fixed before comparative experiments.

---

# 8. Hidden Markov Model State Estimation

The Hidden Markov Model is responsible for estimating the current state of the optimization process.

The optimization state is not directly observable. Instead, it is inferred from the sequence of observations collected during previous optimization iterations.

Let:

\[
S_t \in \mathcal{S}
\]

represent the hidden optimization state at iteration \(t\).

The state space consists of five states:

\[
\mathcal{S} =
\{
E,X,G,O,C
\}
\]

Where:

| State | Description |
|---|---|
| Exploration | The search process needs greater diversity |
| Exploitation | Promising search regions have been identified |
| Stagnation | Recent optimization iterations produce insufficient improvement |
| Overfitting | Training performance improves without corresponding validation improvement |
| Convergence | Optimization improvements have become consistently small |

### HMM Components

The HMM uses three principal components:

**1. Initial State Probabilities**

Represent the initial probability distribution over optimization states.

**2. Transition Probabilities**

Represent the probability of transitioning from one optimization state to another.

\[
A_{ij}=P(S_t=j\mid S_{t-1}=i)
\]

**3. Observation Probabilities**

Represent the likelihood of observing the extracted feature vector under a particular hidden state.

The HMM updates its state probabilities using the previous posterior distribution, transition probabilities, and current observation likelihood.

The resulting posterior distribution is:

\[
P(S_t\mid z_{1:t})
\]

The framework can use the most probable state or the complete posterior distribution when selecting the next optimization action.

Using the posterior distribution allows uncertainty in the estimated optimization state to be considered during decision-making.

---

# 9. Adaptive State-Dependent Action Selection

After estimating the optimization state, AS-HMM-BO selects an appropriate optimization strategy.

Each state corresponds to a different search behavior.

| Optimization State | Adaptive Strategy |
|---|---|
| Exploration | Increase candidate diversity and investigate uncertain regions |
| Exploitation | Focus on promising regions of the search space |
| Stagnation | Expand the search region or introduce more exploration |
| Overfitting | Prefer configurations with better validation generalization |
| Convergence | Reduce search activity and verify stopping conditions |

### Exploration

The framework increases search diversity when the current search region has insufficient coverage or the surrogate model indicates substantial uncertainty.

Possible actions:

- Sample less-explored regions.
- Increase exploration pressure.
- Generate more diverse candidates.

### Exploitation

When promising configurations have been identified, the framework focuses on refining their surrounding search regions.

Possible actions:

- Generate candidates near high-performing configurations.
- Increase exploitation pressure.
- Prioritize candidates with strong predicted performance.

### Stagnation

Stagnation occurs when the optimization process produces little or no improvement over a predefined observation window.

Possible actions:

- Expand the search region.
- Increase candidate diversity.
- Modify the exploration-exploitation balance.
- Restart or diversify part of the search.

### Overfitting

The framework monitors training and validation behavior to identify possible generalization problems.

Possible actions:

- Prefer configurations with smaller generalization gaps.
- Consider stronger regularization.
- Avoid configurations that show substantial validation deterioration.

A generalization-gap signal is an indicator rather than definitive proof of overfitting.

### Convergence

Convergence is considered when the optimization process demonstrates consistently small improvements and satisfies predefined stability conditions.

Possible actions:

- Reduce search activity.
- Avoid unnecessary candidate evaluations.
- Verify stopping criteria.
- Prepare the final configuration.

---

# 10. Bayesian Candidate Generation

Bayesian Optimization generates new candidate configurations using information collected from previous evaluations.

The surrogate model estimates the relationship between hyperparameters and objective performance.

The general Bayesian optimization process consists of:

1. Collect evaluated configurations.
2. Fit or update the surrogate model.
3. Predict candidate performance.
4. Estimate prediction uncertainty.
5. Apply an acquisition function.
6. Select the next candidate.

An acquisition function balances predicted performance and uncertainty.

A general representation is:

\[
\theta_{t+1}
=
\arg\max_{\theta\in\Theta}
a(\theta\mid\mathcal{D}_t)
\]

Where:

- \(\Theta\): Hyperparameter search space.
- \(\mathcal{D}_t\): Previously evaluated configurations.
- \(a\): Acquisition function.

The adaptive action-selection component influences how the next candidate is generated or selected.

This is the main connection between HMM state estimation and Bayesian Optimization.

The HMM does not replace Bayesian Optimization. Instead, it guides the behavior of the optimization process.

---

# 11. Optimization History and Feedback Loop

Every evaluated candidate contributes to the optimization history.

The history should contain:

- Trial identifier.
- Hyperparameter configuration.
- Fidelity level.
- Training metrics.
- Validation metrics.
- Best-so-far performance.
- Estimated optimization state.
- State probabilities.
- Selected adaptive action.
- Surrogate prediction.
- Surrogate uncertainty.
- Training duration.
- Computational budget.

The history is used to update:

1. The Bayesian surrogate model.
2. The HMM observation sequence.
3. The current best configuration.
4. The stopping-condition checks.

This creates a continuous feedback loop in which each evaluation influences subsequent optimization decisions.

---

# 12. Stopping Conditions

The optimization process continues until a predefined stopping condition is satisfied.

Possible stopping criteria include:

- Maximum number of trials reached.
- Maximum computational budget reached.
- Maximum optimization time reached.
- Insufficient improvement over a predefined window.
- Convergence conditions satisfied.

Convergence should not be determined solely by the HMM's most probable state.

The framework should use explicit stopping criteria to prevent premature termination.

Once the stopping condition is satisfied, the framework selects the best configuration according to the predefined validation objective.

---

# 13. Final Model Training and Evaluation

After optimization, the best validated configuration is selected.

The final model is trained according to a predefined training protocol.

The evaluation process consists of:

1. Select the best validated configuration.
2. Freeze the optimization procedure.
3. Freeze the model-selection protocol.
4. Train the final model according to the predefined protocol.
5. Save the final checkpoint.
6. Evaluate using the independent test dataset.
7. Calculate final segmentation metrics.
8. Record computational cost.

Test data must not be used to modify the optimization strategy or select hyperparameters.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Dice Score | Measures segmentation overlap |
| Intersection over Union | Measures region overlap |
| Precision | Measures positive prediction accuracy |
| Recall | Measures foreground detection |
| Validation Loss | Measures validation objective |
| Runtime | Measures computational efficiency |
| AUC | Measures best-so-far optimization progress |

For binary segmentation:

\[
Dice=\frac{2TP}{2TP+FP+FN}
\]

\[
IoU=\frac{TP}{TP+FP+FN}
\]

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Recall=\frac{TP}{TP+FN}
\]

---

# 14. Experimental Comparison

AS-HMM-BO should be evaluated against appropriate optimization baselines under controlled experimental conditions.

Suggested comparison methods:

| Method | Description |
|---|---|
| Random Search | Randomly samples hyperparameter configurations |
| TPE | Uses Tree-structured Parzen Estimator optimization |
| Multi-Fidelity TPE | Combines TPE with resource allocation |
| Rule-Based Adaptive | Uses predefined rules to change search behavior |
| State-Independent Adaptive | Uses adaptive optimization without HMM state estimation |
| AS-HMM-BO | Uses HMM-guided adaptive Bayesian Optimization |

### Experimental Controls

For a fair comparison, methods should use:

- Identical training and validation partitions.
- Identical search-space definitions.
- Identical model architectures.
- Identical evaluation metrics.
- Comparable computational budgets.
- Consistent fidelity definitions.
- Multiple random seeds.
- The same stopping and model-selection protocol.

Where possible, report both trial-based and compute-based performance.

### Primary Evaluation

- Best-so-far validation Dice versus optimization trials.
- Best-so-far validation Dice versus elapsed time.
- Area under the optimization curve.
- Final validation performance.
- Total optimization runtime.

### Secondary Evaluation

- Mean trial performance.
- Performance variability across seeds.
- Number of completed evaluations.
- Computational cost per evaluation.
- State-transition behavior.
- Frequency of adaptive actions.

---

# 15. Ablation Studies

Ablation experiments help identify the contribution of individual AS-HMM-BO components.

Suggested ablations:

| Experiment | Component Removed or Modified |
|---|---|
| Full AS-HMM-BO | Complete framework |
| Without HMM | No hidden-state estimation |
| Without Adaptive Actions | Fixed optimization strategy |
| Without Multi-Fidelity | Full-budget candidate evaluation |
| Without State Feedback | State-independent search |
| Without Overfitting State | Removes overfitting-specific state handling |
| Without Stagnation State | Removes stagnation-specific adaptation |

The full framework should be compared against each ablation using identical experimental conditions.

This helps determine whether improvements arise from HMM state estimation, adaptive action selection, multi-fidelity evaluation, or their combination.

---

# 16. Reproducibility and Research Protocol

Reproducibility is essential for meaningful comparisons.

The following information should be recorded for every experiment:

- Dataset version.
- Training and validation split identifiers.
- Random seed.
- Model architecture.
- Search-space configuration.
- Optimizer configuration.
- HMM initialization.
- HMM transition probabilities.
- Observation-feature definitions.
- Acquisition-function configuration.
- Fidelity schedule.
- Stopping criteria.
- Software environment.
- Hardware configuration.
- Training and validation metrics.
- Checkpoint metadata.

All experiments should preserve their configurations and outputs separately.

The independent test set should remain untouched until the optimization procedure and final model-selection protocol have been frozen.

---

# 17. Project Architecture

A modular implementation can be organized as follows:

```text
AS-HMM-BO/
|
+-- configs/
|   +-- dataset.yaml
|   +-- search_space.yaml
|   +-- optimizer.yaml
|   +-- hmm.yaml
|   +-- training.yaml
|
+-- data/
|   +-- manifests/
|   +-- processed/
|
+-- models/
|   +-- unet.py
|   +-- segmentation_models.py
|
+-- optimization/
|   +-- bayesian_optimizer.py
|   +-- surrogate_model.py
|   +-- acquisition.py
|   +-- candidate_generator.py
|
+-- hmm/
|   +-- state_features.py
|   +-- observation_model.py
|   +-- transition_model.py
|   +-- state_estimator.py
|
+-- adaptive/
|   +-- action_selector.py
|   +-- exploration.py
|   +-- exploitation.py
|   +-- stagnation.py
|   +-- overfitting.py
|   +-- convergence.py
|
+-- evaluation/
|   +-- fidelity_evaluator.py
|   +-- segmentation_metrics.py
|   +-- validation.py
|   +-- test_evaluation.py
|
+-- experiments/
|   +-- random_search.py
|   +-- tpe.py
|   +-- multifidelity_tpe.py
|   +-- rule_based.py
|   +-- state_independent.py
|   +-- as_hmm_bo.py
|
+-- analysis/
|   +-- optimization_curves.py
|   +-- state_analysis.py
|   +-- ablation_analysis.py
|   +-- statistical_analysis.py
|
+-- outputs/
|   +-- checkpoints/
|   +-- histories/
|   +-- metrics/
|   +-- figures/
|
+-- tests/
|
+-- notebooks/
|
+-- requirements.txt
+-- README.md
+-- main.py
```

This is a suggested modular architecture. Individual files and modules can be combined or separated according to implementation requirements.

---

# 18. Design Principles

AS-HMM-BO follows these design principles:

1. **General-purpose optimization:** The framework should support different segmentation architectures and datasets.
2. **State-aware adaptation:** Optimization behavior should respond to estimated hidden states.
3. **Bayesian candidate generation:** Candidate selection should use information from previous evaluations.
4. **Computational efficiency:** Multi-fidelity evaluation should reduce unnecessary resource consumption.
5. **Reproducibility:** All optimization decisions and experimental configurations should be recorded.
6. **Leakage prevention:** Independent test data must remain isolated from optimization.
7. **Modularity:** HMM estimation, candidate generation, adaptive actions, and model evaluation should be independently testable.
8. **Evidence-based evaluation:** Performance claims must be supported by controlled experiments.

---

# 19. Expected Research Contribution

The central research idea behind AS-HMM-BO is to introduce explicit optimization-state estimation into Bayesian hyperparameter optimization.

The framework aims to move beyond a fixed exploration-exploitation strategy by allowing the optimization process to respond to its estimated behavioral state.

Its potential contributions include:

- HMM-based modeling of optimization dynamics.
- State-dependent adaptive search behavior.
- Integration of state estimation with Bayesian candidate generation.
- Multi-fidelity resource allocation.
- Optimization-history-based decision-making.
- A reproducible framework for evaluating adaptive hyperparameter optimization in image segmentation.

The effectiveness of these contributions must be established through controlled comparisons and ablation studies rather than assumed from the framework design.

---

# 20. Summary

AS-HMM-BO combines Bayesian Optimization with Hidden Markov Model-based optimization-state estimation.

The Bayesian optimizer proposes candidate configurations, the HMM estimates the current optimization state, and the adaptive action-selection mechanism modifies subsequent search behavior.

Multi-fidelity evaluation controls computational expenditure, while the optimization history provides continuous feedback to the framework.

The overall objective is to develop a general-purpose, reproducible, and computationally efficient optimization framework for deep-learning-based image segmentation.

**AS-HMM-BO: Adaptive optimization through hidden-state estimation and Bayesian search.**