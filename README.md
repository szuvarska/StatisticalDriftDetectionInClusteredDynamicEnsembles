# Statistical Drift Detection in Clustered Dynamic Ensembles

This repository implements a modified **Streaming Random Patches (SRP) Classifier** based on the research for Clustered Dynamic Ensemble Selection. The core innovation is transitioning from global ensemble averaging to localized, dynamic competence tracking.

## 🚀 Project Overview
Standard drift detection methods often monitor only global model performance. This implementation introduces **Cluster-Based Dynamic Ensemble Selection (C-DES)**, which uses clustering to define localized regions of expertise. When a localized drift is detected, we only adapt the affected models, preserving valuable historical knowledge in stable regions.

## 🛠 Key Features
- **Context Mapping**: Instances are passed through **K-Means** to map them to a localized region (cluster) in the feature space.
- **Localized Competency Scoring**: Uses a **Test-then-Train** protocol to track accuracy for each classifier on a per-cluster basis.
- **Dynamic Selection (DES)**: Identifies the "context" of incoming data and only listens to classifiers that are proven experts in that specific region (better than random guessing threshold).
- **Soft Reset on Drift**: Instead of a global reset (deleting trees), this method wipes competence scores for the affected cluster, ensuring faster recovery and smaller accuracy drops.

## 📦 Installation
Ensure you have Python 3.10+ installed. Install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```
Dependencies: `river`, `numpy`.

## 📂 Repository Structure
- `src/streaming_random_patches.py`: The main implementation of `SRPClassifier` and `BaseSRPClassifier` with C-DES logic.
- `notebooks/01_test_c-des_impl.ipynb`: Basic implementation tests.
- `notebooks/02_test_c-des_vs_original_srp.ipynb`: C-DES SRP vs. standard River SRP.
- `notebooks/03_experiments.ipynb`: Recurring concept and number of clusters sensitivity experiments.

## 🧪 Usage
You can use the `SRPClassifier` in a similar way to other River classifiers. Here's a quick example:

```python
from src.streaming_random_patches import SRPClassifier
from river.drift import ADWIN

# Initialize the Cluster-Based Dynamic Ensemble
model = SRPClassifier(
    n_models=10,
    n_clusters=3,
    drift_detector=ADWIN(),
    training_method="patches"
)

# Standard River learning cycle
# model.predict_one(x)
# model.learn_one(x, y)
```

## 📊 Evaluation Metrics
For performance evaluation on imbalanced, non-stationary streams, utilize:
- Balanced Accuracy
- Macro F1-Score
- Geometric Mean (G-mean)

## 👥 Team
- Michał Iwicki ([@Michal-Iwicki](https://github.com/Michal-Iwicki))
- Dawid Sroczyk ([@dawidsroczyk](https://github.com/dawidsroczyk))
- Marta Szuwarska ([@szuvarska](https://github.com/szuvarska))