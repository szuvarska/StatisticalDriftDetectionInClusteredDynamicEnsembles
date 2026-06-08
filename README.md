# Statistical Drift Detection in Clustered Dynamic Ensembles

This repository implements a modified **Streaming Random Patches (SRP) Classifier** that integrates **Cluster-Based Dynamic Ensemble Selection (C-DES)** with a novel **Statistical Drift Detection Method (SDDM)**.

The core innovation is transitioning from global ensemble averaging and global hard resets to **localized, data-driven drift adaptation**. By detecting distributional shifts within specific spatial clusters, our framework can perform "soft resets" to quickly adapt to non-stationary environments while preserving valuable historical knowledge in stable regions.

## 🚀 Project Overview
Standard drift detection methods (like ADWIN) monitor global error rates, meaning the model only adapts *after* it starts making mistakes. This framework introduces a localized, data-driven approach:
1. **C-DES** defines localized regions of expertise using clustering.
2. **SDDM** monitors the actual data distribution within those clusters to detect statistical divergences *before* they catastrophically impact accuracy.

## 🛠 Key Features
- **Context Mapping**: Instances are passed through a **K-Means** clusterer to map them to a localized region in the feature space.
- **Rolling Localized Competency**: Uses a **Test-then-Train** protocol with a rolling window to track cluster-specific accuracy. This allows the ensemble to smoothly adapt to evolving cluster boundaries.
- **Dynamic Selection & Fallback**: Identifies the "context" of incoming data and only listens to classifiers that outperform the **local majority class baseline**. If all classifiers are deemed incompetent, the ensemble safely falls back to predicting the region's true majority class.
- **Minor vs. Major Drift Responses**:
  - **Minor Drift (Soft Reset):** Triggered when drift magnitude is within standard deviations. Wipes competence scores for the affected cluster, forcing models to re-earn their voting weight while preserving their structural trees.
  - **Major Drift (Hard Reset):** Triggered when drift magnitude is statistically severe (exceeds a Z-score threshold). Completely replaces the underlying base learners to adapt to permanent conceptual shifts.

## 🏗 Supported Configurations
To systematically evaluate the impact of localized selection and data-driven drift detection, this framework supports four distinct structural configurations:

1. **Baseline SRP (Global Error-Driven):** The standard River implementation relying entirely on error-driven ADWIN detectors and global hard resets.
2. **Base C-DES (Localized Error-Driven Resets):** Utilizes context mapping and dynamic selection. Employs independent ADWIN detectors for each cluster to trigger soft resets when regional accuracy drops.
3. **C-DES-SDDM (Data-Driven Resets):** Integrates SDDM to monitor data distribution. Triggers localized soft resets for minor drifts and structural hard resets for statistically severe major drifts, entirely replacing error-based tracking.
4. **C-DES-Hybrid (Data & Error-Driven Resets):** SDDM monitors data distribution for preemptive soft resets, while internal ADWIN detectors remain enabled as a fail-safe to trigger hard resets if a learner's predictive error drops significantly.

## 📦 Installation
Ensure you have Python 3.10+ installed. Install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```
Dependencies: `river`, `numpy`.

## 📂 Repository Structure
- `src/streaming_random_patches.py`: The core ensemble implementation containing `SRPClassifierADWIN` and `SRPClassifierSDDM`.
- `src/river_sddm.py`: Implementation of the Statistical Drift Detection Method (SDDM).
- `notebooks/01_test_c-des_impl.ipynb`: Basic implementation tests.
- `notebooks/02_test_c-des_vs_original_srp.ipynb`: Base C-DES vs. Vanilla SRP benchmarks.
- `notebooks/03_experiments.ipynb`: Recurring concept and number of clusters sensitivity experiments.
- `notebooks/04_major_vs_minor_drifts.ipynb`: Evaluating SDDM magnitude thresholds and reset strategies.
- `experiments_v3.ipynb`: Final experiments described thoroughly in the report.

## 🧪 Usage
You can use the he C-DES classifiers in a similar way to other River models. Here is a quick example of initializing the C-DES-SDDM configuration:

```python
from src.streaming_random_patches import SRPClassifierSDDM, make_sddm
from river.drift import ADWIN

# Initialize the Data-Driven C-DES-SDDM Ensemble
model = SRPClassifierSDDM(
    n_models=10,
    n_clusters=5,
    sddm_constructor=make_sddm,
    disable_detector="drift", # Disables ADWIN hard resets, delegating to SDDM
    major_drift_factor=3.0,   # Z-score threshold for Major Drifts (Hard Resets)
    training_method="patches",
    seed=42
)

# Standard River streaming loop
# for x, y in dataset:
#     y_pred = model.predict_one(x)
#     model.learn_one(x, y)
```

## 👥 Team
- Michał Iwicki ([@Michal-Iwicki](https://github.com/Michal-Iwicki))
- Dawid Sroczyk ([@dawidsroczyk](https://github.com/dawidsroczyk))
- Marta Szuwarska ([@szuvarska](https://github.com/szuvarska))
