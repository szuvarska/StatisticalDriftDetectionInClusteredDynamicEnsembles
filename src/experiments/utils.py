import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools
import os

from river import metrics, datasets
from river.datasets import synth
from river.drift import ADWIN
from river.utils import Rolling
from river.ensemble import SRPClassifier as VanillaSRP

from src.streaming_random_patches import SRPClassifierADWIN as CDES_SRP, SRPClassifierSDDM as CDES_SRP_SDDM

def make_models(n_models=10, n_clusters=2, drift_delta=1e-5, warn_delta=1e-4,
                include_vanilla=True, include_cdes=True, include_cdes_sddm=True, include_cdes_sddm_adwin=True, srp_seed=42, extra_cdes_kwargs=None):
    extra_cdes_kwargs = extra_cdes_kwargs or {}
    models = {}
    if include_vanilla:
        models["Vanilla SRP"] = VanillaSRP(
            n_models=n_models,
            drift_detector=ADWIN(delta=drift_delta),
            warning_detector=ADWIN(delta=warn_delta),
            seed=srp_seed,
        )
    if include_cdes:
        models["Base C-DES"] = CDES_SRP(
            n_models=n_models,
            n_clusters=n_clusters,
            drift_detector=ADWIN(delta=drift_delta),
            warning_detector=ADWIN(delta=warn_delta),
            disable_detector='drift',
            seed=srp_seed,
            **extra_cdes_kwargs
        )
    if include_cdes_sddm:
        models["C-DES-SDDM"] = CDES_SRP_SDDM(
            n_models=n_models,
            n_clusters=n_clusters,
            drift_detector=ADWIN(delta=drift_delta),
            warning_detector=ADWIN(delta=warn_delta),
            seed=srp_seed,
            disable_detector="drift",
            printer=False,
            major_drift_factor=3.0,
            **extra_cdes_kwargs
        )
    if include_cdes_sddm_adwin:
        models["C-DES-Hybrid"] = CDES_SRP_SDDM(
            n_models=n_models,
            n_clusters=n_clusters,
            drift_detector=ADWIN(delta=drift_delta),
            warning_detector=ADWIN(delta=warn_delta),
            seed=srp_seed,
            disable_detector="off",
            printer=False,
            major_drift_factor=3.0,
            **extra_cdes_kwargs
        )
    return models

class RegionalDriftStream:
    def __init__(self, drift_interval=2000, n_samples=10000):
        self.drift_interval = drift_interval
        self.n_samples = n_samples
        self.concept_stable = synth.SEA(variant=0, seed=42)
        self.concept_drift_1 = synth.SEA(variant=1, seed=42)
        self.concept_drift_2 = synth.SEA(variant=2, seed=42)
    def __iter__(self):
        iter_stable = iter(self.concept_stable)
        iter_d1 = iter(self.concept_drift_1)
        iter_d2 = iter(self.concept_drift_2)
        for i in range(self.n_samples):
            is_region_a = (i % 2 == 0)
            if is_region_a:
                x, y = next(iter_stable)
                x['context_feature'] = 0.1
                region_label = "Region A"
            else:
                if (i // self.drift_interval) % 2 == 0:
                    x, y = next(iter_d1)
                else:
                    x, y = next(iter_d2)
                x['context_feature'] = 0.9
                region_label = "Region B"
            yield x, y, region_label

class SensorFailureStream:
    def __init__(self, drift_start=3000, n_samples=10000, seed=42):
        self.drift_start = drift_start
        self.n_samples = n_samples
        self.rng = random.Random(seed)
    def __iter__(self):
        for i in range(self.n_samples):
            temp = self.rng.uniform(0, 1)
            sensor_a = self.rng.uniform(0, 1)
            sensor_b = self.rng.uniform(0, 1)
            is_high_temp = temp > 0.5
            region_label = "High Temp (Drifting)" if is_high_temp else "Low Temp (Stable)"
            if is_high_temp and i >= self.drift_start:
                y = 1 if (sensor_a + sensor_b) < 1.0 else 0
            else:
                y = 1 if (sensor_a + sensor_b) >= 1.0 else 0
            x = {'temp': temp, 'sensor_a': sensor_a, 'sensor_b': sensor_b}
            yield x, y, region_label

class RecurrentDriftStream:
    def __init__(self, phase_length=5000):
        self.phase_length = phase_length
        self.total_samples = phase_length * 3
        self.concept_a = synth.Agrawal(classification_function=1, seed=42)
        self.concept_b = synth.Agrawal(classification_function=2, seed=42)
    def __iter__(self):
        iter_a = iter(self.concept_a)
        iter_b = iter(self.concept_b)
        for i in range(self.total_samples):
            if i < self.phase_length:
                x, y = next(iter_a); x['env_mode'] = 0.0; phase = "Phase 1 (Concept A)"
            elif i < self.phase_length * 2:
                x, y = next(iter_b); x['env_mode'] = 1.0; phase = "Phase 2 (Concept B)"
            else:
                x, y = next(iter_a); x['env_mode'] = 0.0; phase = "Phase 3 (Concept A Returns)"
            yield x, y, phase

class RapidFlickerStream:
    def __init__(self, flicker_interval=250, n_samples=10000, seed=42):
        self.flicker_interval = flicker_interval
        self.n_samples = n_samples
        self.concept_alpha = synth.Agrawal(classification_function=1, seed=seed)
        self.concept_beta = synth.Agrawal(classification_function=2, seed=seed)
    def __iter__(self):
        iter_alpha = iter(self.concept_alpha)
        iter_beta = iter(self.concept_beta)
        for i in range(self.n_samples):
            if (i // self.flicker_interval) % 2 == 0:
                x, y = next(iter_alpha); x['system_state'] = 0.0
            else:
                x, y = next(iter_beta); x['system_state'] = 1.0
            yield x, y

class MultiContextStream:
    def __init__(self, phase_length=2500, n_samples=10000, seed=42):
        self.phase_length = phase_length
        self.n_samples = n_samples
        self.concepts = [
            synth.Agrawal(classification_function=1, seed=seed),
            synth.Agrawal(classification_function=2, seed=seed),
            synth.Agrawal(classification_function=3, seed=seed),
            synth.Agrawal(classification_function=4, seed=seed)
        ]
    def __iter__(self):
        iters = [iter(c) for c in self.concepts]
        for i in range(self.n_samples):
            active_idx = (i // self.phase_length) % 4
            x, y = next(iters[active_idx])
            x['observable_state'] = float(active_idx)
            yield x, y

class NoisyAgrawalStream:
    def __init__(self, n_noise_features=0, n_samples=5000, seed=42):
        self.n_noise_features = n_noise_features
        self.n_samples = n_samples
        self.base_stream = synth.Agrawal(classification_function=1, seed=seed)
        self.rng = random.Random(seed)
    def __iter__(self):
        base_iter = iter(self.base_stream)
        for _ in range(self.n_samples):
            x, y = next(base_iter)
            x_noisy = x.copy()
            for i in range(self.n_noise_features):
                x_noisy[f'noise_{i}'] = self.rng.uniform(0, 1)
            yield x_noisy, y

class BenchmarkDriftStream:
    def __init__(self, seed=42):
        self.concepts = [
            synth.SEA(variant=0, noise=0.1, seed=seed),
            synth.SEA(variant=1, noise=0.1, seed=seed),
            synth.SEA(variant=2, noise=0.1, seed=seed),
            synth.SEA(variant=3, noise=0.1, seed=seed)
        ]
        self.phase_length = 10000
        self.n_samples = self.phase_length * 4
    def __iter__(self):
        iters = [iter(c) for c in self.concepts]
        for i in range(self.n_samples):
            active_idx = i // self.phase_length
            x, y = next(iters[active_idx])
            yield x, y

class ConflictingContextStream:
    def __init__(self, n_samples=10000, flicker_rate=2000):
        self.n_samples = n_samples
        self.flicker_rate = flicker_rate
        self.concept_a = synth.Agrawal(classification_function=1, seed=42)
        self.concept_b = synth.Agrawal(classification_function=2, seed=42)
    def __iter__(self):
        iter_a = iter(self.concept_a)
        iter_b = iter(self.concept_b)
        for i in range(self.n_samples):
            is_concept_a = (i // self.flicker_rate) % 2 == 0
            if is_concept_a:
                x, y = next(iter_a); x['context'] = 0.0
            else:
                x, y = next(iter_b); x['context'] = 1.0
            yield x, y

class NoiseQuarantineStream:
    def __init__(self, n_samples=10000, noise_start=3000, seed=42):
        self.n_samples = n_samples
        self.noise_start = noise_start
        self.rng = random.Random(seed)
        self.concept_a = synth.Agrawal(classification_function=1, seed=seed)
        self.concept_b = synth.Agrawal(classification_function=2, seed=seed)
    def __iter__(self):
        iter_a = iter(self.concept_a)
        iter_b = iter(self.concept_b)
        for i in range(self.n_samples):
            region_choice = self.rng.choice([1,2,3])
            if region_choice == 1:
                x, y = next(iter_a); x['context']=0.0; region="Stable Regions"
            elif region_choice == 2:
                x, y = next(iter_b); x['context']=0.5; region="Stable Regions"
            else:
                if i >= self.noise_start:
                    x = {'feature_1': self.rng.random(), 'feature_2': self.rng.random(), 'context': 1.0}
                    y = self.rng.choice([0,1]); region="Noisy Region"
                else:
                    x, y = next(iter_a); x['context'] = 1.0; region="Stable Regions"
            yield x, y, region

def load_csv_as_stream(csv_path: str, target_col: str, n_samples: int | None = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    total = len(df) if n_samples is None else min(len(df), n_samples)
    for _, row in df.iloc[:total].iterrows():
        x = row.drop(labels=[target_col]).to_dict()
        for k, v in list(x.items()):
            if pd.isna(v):
                x.pop(k)
            elif isinstance(v, (np.floating, float)):
                x[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                x[k] = int(v)
        y = row[target_col]
        yield x, y

def fetch_dynamic_dataset_stream(name: str, n_samples: int = 15000):
    try:
        from sklearn.datasets import fetch_covtype, fetch_openml
    except ImportError:
        print(f"[WARN] Please install scikit-learn (`pip install scikit-learn`) to auto-download {name}.")
        return []

    if name == "covtype":
        print("  -> Downloading/Loading Forest Covertype (this may take a moment)...")
        data = fetch_covtype(as_frame=True)
        df = data.frame
        target_col = "Cover_Type"
    elif name == "airlines":
        print("  -> Downloading/Loading Airlines dataset from OpenML (this may take a moment)...")
        data = fetch_openml("airlines", version=1, as_frame=True, parser="auto")
        df = data.frame
        target_col = "Delay"
    else:
        return []

    for col in df.columns:
        if col != target_col and df[col].dtype.name in ['category', 'object', 'string']:
            df[col] = df[col].astype('category').cat.codes.astype(float)
            
    if df[target_col].dtype.name in ['category', 'object', 'string']:
        df[target_col] = df[target_col].astype('category').cat.codes

    total = min(len(df), n_samples)
    for _, row in df.iloc[:total].iterrows():
        x = row.drop(labels=[target_col]).to_dict()
        for k, v in list(x.items()):
            if pd.isna(v):
                x.pop(k)
            elif isinstance(v, (np.floating, float)):
                x[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                x[k] = int(v)
            elif isinstance(v, str):
                x[k] = float(hash(v) % 10000)
                
        y_val = row[target_col]
        if pd.isna(y_val) or (isinstance(y_val, (int, np.integer)) and y_val < 0):
            continue
        yield x, int(y_val)

def plot_time_series(x_axis, series_dict, title="", ylabel="Accuracy", vlines=None, y_lim=None, figsize=(12,4), real_drifts=None, drift_detections_base=None, drift_detections_ens=None):
    plt.figure(figsize=figsize)
    for name, series in series_dict.items():
        line = plt.plot(x_axis[:len(series)], series, label=name)[0]
        
        if drift_detections_base and name in drift_detections_base:
            drifts = drift_detections_base[name]
            if drifts:
                x_coords = list(x_axis[:len(series)])
                y_vals = np.interp(drifts, x_coords, series)
                plt.scatter(drifts, y_vals, marker='o', color=line.get_color(), edgecolors='black', linewidths=1.0, alpha=0.8, s=60, zorder=9)
                
        if drift_detections_ens and name in drift_detections_ens:
            drifts = drift_detections_ens[name]
            if drifts:
                x_coords = list(x_axis[:len(series)])
                y_vals = np.interp(drifts, x_coords, series)
                plt.scatter(drifts, y_vals, marker='X', color=line.get_color(), edgecolors='black', linewidths=1.5, alpha=1.0, s=120, zorder=10)
                
    if vlines:
        for v in vlines:
            plt.axvline(x=v, color='red', linestyle=':', alpha=0.8)
    
    if real_drifts:
        for i, v in enumerate(real_drifts):
            plt.axvline(x=v, color='black', linestyle='--', linewidth=2, alpha=0.7, label="Real Drift" if i == 0 else "")
            
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if y_lim:
        plt.ylim(y_lim)
    plt.tight_layout()
    plt.show()

def get_total_drifts(model):
    base_count = 0
    ens_count = 0
    
    try:
        base_learners = getattr(model, "models", [])
        if not base_learners and hasattr(model, '__iter__'):
            base_learners = model
        for bm in base_learners:
            val = getattr(bm, "n_drifts_detected", 0)
            if isinstance(val, (int, float)):
                base_count += int(val)
    except Exception:
        pass
        
    for attr in ["n_drift_detected", "n_drifts_detected"]:
        val = getattr(model, attr, 0)
        if isinstance(val, (int, float)):
            ens_count += int(val)
            
    return base_count, ens_count

def run_experiment_1():
    print("Initializing Experiment 1: Regional Concept Drift...")
    models = make_models(n_models=10, n_clusters=2)
    results = defaultdict(lambda: {
        "Global": Rolling(metrics.Accuracy(), window_size=500),
        "Region A": Rolling(metrics.Accuracy(), window_size=250),
        "Region B": Rolling(metrics.Accuracy(), window_size=250),
        "history_A": [],
        "history_B": [],
        "history_Global": [],
        "drifts_base": [],
        "drifts_ens": [],
        "last_drifts_base": 0,
        "last_drifts_ens": 0
    })
    stream = RegionalDriftStream(drift_interval=2500, n_samples=10000)
    print("Processing stream...")
    for i, (x, y, region) in enumerate(stream):
        for name, model in models.items():
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Global"].update(y, y_pred)
                results[name][region].update(y, y_pred)
            model.learn_one(x, y)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
                
            if i % 10 == 0:
                results[name]["history_Global"].append(results[name]["Global"].get())
                results[name]["history_A"].append(results[name]["Region A"].get())
                results[name]["history_B"].append(results[name]["Region B"].get())
        if (i + 1) % 2500 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plots...")
    x_axis = range(0, 10000, 10)
    
    r_drifts = [2500, 5000, 7500]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    
    plot_time_series(x_axis,
                     {name: results[name]["history_Global"] for name in results.keys()},
                     title="Global Accuracy (Prequential, Window=500)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    plot_time_series(x_axis,
                     {name: results[name]["history_A"] for name in results.keys()},
                     title="Region A Accuracy (Stable Concept)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    plot_time_series(x_axis,
                     {name: results[name]["history_B"] for name in results.keys()},
                     title="Region B Accuracy (Drifting Concept)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)

def run_experiment_2():
    print("Initializing Experiment 2: Contextual Sensor Drift...")
    models = make_models(n_models=10, n_clusters=2)
    results = defaultdict(lambda: {
        "Global": Rolling(metrics.Accuracy(), window_size=500),
        "Low Temp (Stable)": Rolling(metrics.Accuracy(), window_size=250),
        "High Temp (Drifting)": Rolling(metrics.Accuracy(), window_size=250),
        "history_Global": [],
        "history_Stable": [],
        "history_Drifting": [],
        "drifts_base": [],
        "drifts_ens": [],
        "last_drifts_base": 0,
        "last_drifts_ens": 0
    })
    stream = SensorFailureStream(drift_start=4000, n_samples=10000)
    print("Processing stream...")
    for i, (x, y, region) in enumerate(stream):
        for name, model in models.items():
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Global"].update(y, y_pred)
                results[name][region].update(y, y_pred)
            model.learn_one(x, y)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
                
            if i % 20 == 0:
                results[name]["history_Global"].append(results[name]["Global"].get())
                results[name]["history_Stable"].append(results[name]["Low Temp (Stable)"].get())
                results[name]["history_Drifting"].append(results[name]["High Temp (Drifting)"].get())
        if (i + 1) % 2500 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plots...")
    x_axis = range(0, 10000, 20)
    r_drifts = [4000]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    
    plot_time_series(x_axis,
                     {name: results[name]["history_Global"] for name in results.keys()},
                     title="Global Accuracy (Window=500)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    plot_time_series(x_axis,
                     {name: results[name]["history_Stable"] for name in results.keys()},
                     title="Low Temp Region (Concept remains stable)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    plot_time_series(x_axis,
                     {name: results[name]["history_Drifting"] for name in results.keys()},
                     title="High Temp Region (Concept Inverts at Sample 4000)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)

def run_experiment_3():
    print("Initializing Experiment 3: Recurrent Drift (A -> B -> A)...")
    models = make_models(n_models=10, n_clusters=2)
    results = defaultdict(lambda: {"Accuracy": Rolling(metrics.Accuracy(), window_size=250), "history": [], "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0})
    stream = RecurrentDriftStream(phase_length=5000)
    print("Processing stream...")
    for i, (x, y, phase) in enumerate(stream):
        for name, model in models.items():
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Accuracy"].update(y, y_pred)
            model.learn_one(x, y)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
                
            if i % 25 == 0:
                results[name]["history"].append(results[name]["Accuracy"].get())
        if (i + 1) % 3000 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plot...")
    x_axis = range(0, 15000, 25)
    r_drifts = [5000, 10000]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    plot_time_series(x_axis, {name: results[name]["history"] for name in results.keys()},
                     title="Recurrent Drift (A->B->A) - Rolling Accuracy", y_lim=(0.4,1.05), real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)

def run_experiment_4():
    print("Initializing Experiment 4: Rapid Concept Flickering...")
    models = make_models(n_models=10, n_clusters=2)
    results = defaultdict(lambda: {
        "Rolling_Acc": Rolling(metrics.Accuracy(), window_size=100),
        "Cumulative_Acc": metrics.Accuracy(),
        "history_rolling": [],
        "history_cumulative": [],
        "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0
    })
    stream = RapidFlickerStream(flicker_interval=250, n_samples=10000)
    print("Processing stream...")
    for i, (x, y) in enumerate(stream):
        for name, model in models.items():
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Rolling_Acc"].update(y, y_pred)
                results[name]["Cumulative_Acc"].update(y, y_pred)
            model.learn_one(x, y)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
                
            if i % 10 == 0:
                results[name]["history_rolling"].append(results[name]["Rolling_Acc"].get())
                results[name]["history_cumulative"].append(results[name]["Cumulative_Acc"].get())
        if (i + 1) % 2500 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plots...")
    x_axis = range(0, 10000, 10)
    r_drifts = list(range(250, 10000, 250))
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    plot_time_series(x_axis, {name: results[name]["history_rolling"] for name in results.keys()},
                     title="Responsiveness: Prequential Accuracy (Window = 100)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    plot_time_series(x_axis, {name: results[name]["history_cumulative"] for name in results.keys()},
                     title="Overall Robustness: Cumulative Accuracy", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)

def run_experiment_5():
    print("Initializing Experiment 5: Hyperparameter Sensitivity (Cluster Count k)...")
    k_values = [2,4,8,20]
    models = {"Vanilla SRP": VanillaSRP(n_models=10, drift_detector=ADWIN(delta=1e-5), seed=42)}
    for k in k_values:
        models[f"C-DES (k={k})"] = CDES_SRP(n_models=10, n_clusters=k, drift_detector=ADWIN(delta=1e-5), seed=42)
    results = defaultdict(lambda: {"Rolling_Acc": Rolling(metrics.Accuracy(), window_size=500),
                                   "Cumulative_Acc": metrics.Accuracy(),
                                   "history_rolling": [], "total_time": 0.0,
                                   "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0})
    stream = MultiContextStream(phase_length=2500, n_samples=10000)
    print("Processing stream and measuring latency...")
    for i, (x, y) in enumerate(stream):
        for name, model in models.items():
            start_time = time.perf_counter()
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Rolling_Acc"].update(y, y_pred)
                results[name]["Cumulative_Acc"].update(y, y_pred)
            model.learn_one(x, y)
            results[name]["total_time"] += (time.perf_counter() - start_time)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
            
            if i % 25 == 0:
                results[name]["history_rolling"].append(results[name]["Rolling_Acc"].get())
        if (i + 1) % 2500 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plots...")
    x_axis = range(0, 10000, 25)
    r_drifts = [2500, 5000, 7500]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    plot_time_series(x_axis, {name: results[name]["history_rolling"] for name in results.keys()},
                     title="Prequential Accuracy (Window = 500)", real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    
    names = list(results.keys())
    accuracies = [results[n]["Cumulative_Acc"].get() * 100 for n in names]
    latencies = [(results[n]["total_time"] / 10000) * 1000 for n in names]
    plt.figure(figsize=(8,5))
    plt.scatter(latencies, accuracies)
    for i, name in enumerate(names):
        plt.annotate(name, (latencies[i], accuracies[i]), xytext=(0,10), textcoords="offset points", ha='center')
    plt.title("Trade-off: Accuracy vs. Computational Latency")
    plt.xlabel("Average Latency (ms / sample)")
    plt.ylabel("Cumulative Accuracy (%)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def run_experiment_6(n_trials=5):
    print(f"Initializing Experiment 6: The Curse of Dimensionality (Noise Injection) - {n_trials} Trials per level...")
    noise_levels = [0,10,50,100]
    n_samples = 5000
    
    final_results = defaultdict(lambda: defaultdict(list))
    
    for noise_val in noise_levels:
        print(f"\n--- Testing with {noise_val} injected noise features ---")
        for trial in range(n_trials):
            seed = 42 + trial
            models = make_models(n_models=10, n_clusters=2, srp_seed=seed)
            acc_trackers = {name: metrics.Accuracy() for name in models.keys()}
            stream = NoisyAgrawalStream(n_noise_features=noise_val, n_samples=n_samples, seed=seed)
            
            for i, (x, y) in enumerate(stream):
                for name, model in models.items():
                    y_pred = model.predict_one(x)
                    if y_pred is not None:
                        acc_trackers[name].update(y, y_pred)
                    model.learn_one(x, y)
                    
            for name in models.keys():
                final_results[name][noise_val].append(acc_trackers[name].get())
                
        for name in models.keys():
            avg_acc = np.mean(final_results[name][noise_val])
            print(f"{name} Avg Final Acc: {avg_acc:.2%}")

    print("\nExperiment complete. Generating grouped boxplots...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(models.keys())
    colors = ['blue', '#d4a017', 'green', 'red']
    
    step = 0.15 
    width = 0.1
    
    for m_idx, name in enumerate(model_names):
        color = colors[m_idx % len(colors)]
        offset = (m_idx - (len(model_names) - 1) / 2) * step
        positions = [i + offset for i in range(len(noise_levels))]
        
        data = [final_results[name][n] for n in noise_levels]
        
        ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.4),
                   medianprops=dict(color=color, linewidth=2),
                   manage_ticks=False)
        
        medians = [np.median(d) for d in data]
        ax.plot(positions, medians, color=color, marker='o', linestyle='-', linewidth=2.0, label=name)

    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels(noise_levels)
    ax.set_title("Robustness to Irrelevant Features (Distribution over Multiple Trials)")
    ax.set_xlabel("Number of Injected Noise Features")
    ax.set_ylabel("Final Cumulative Accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def run_experiment_7():
    print("Initializing Experiment 7: Standard Synthetic Benchmarks (SEA)...")
    from river.forest import ARFClassifier as ARF
    from river.tree import HoeffdingAdaptiveTreeClassifier as HAT
    models = {
        "HAT (Single Tree)": HAT(seed=42),
        "ARF (Ensemble)": ARF(n_models=10, seed=42),
        "Vanilla SRP": VanillaSRP(n_models=10, seed=42),
        "C-DES SRP": CDES_SRP(n_models=10, n_clusters=4, seed=42)
    }
    results = defaultdict(lambda: {
        "Rolling_Acc": Rolling(metrics.Accuracy(), window_size=1000),
        "Cumulative_Acc": metrics.Accuracy(),
        "Kappa": metrics.CohenKappa(),
        "history_rolling": [], "total_time": 0.0,
        "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0
    })
    stream = BenchmarkDriftStream(seed=42)
    n_samples = 40000
    print("Processing stream...")
    for i, (x, y) in enumerate(stream):
        for name, model in models.items():
            start_time = time.perf_counter()
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Rolling_Acc"].update(y, y_pred)
                results[name]["Cumulative_Acc"].update(y, y_pred)
                results[name]["Kappa"].update(y, y_pred)
            model.learn_one(x, y)
            results[name]["total_time"] += (time.perf_counter() - start_time)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
            
            if i % 100 == 0:
                results[name]["history_rolling"].append(results[name]["Rolling_Acc"].get())
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} samples (Concept Drift Triggered)...")
    print("\nExperiment complete. Generating plot and summary table...")
    x_axis = range(0, n_samples, 100)
    r_drifts = [10000, 20000, 30000]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    plot_time_series(x_axis, {name: results[name]["history_rolling"] for name in results.keys()},
                     title="Benchmark Performance on SEA Concepts (Rolling Window = 1000)", 
                     real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    data = []
    for name in results.keys():
        acc = results[name]["Cumulative_Acc"].get()
        kappa = results[name]["Kappa"].get()
        latency = (results[name]["total_time"] / n_samples) * 1000
        data.append({"Model": name, "Accuracy (%)": f"{acc * 100:.2f}%", "Cohen's Kappa": f"{kappa:.4f}", "Latency (ms/sample)": f"{latency:.2f}"})
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

def run_experiment_8():
    print("Initializing Experiment 8: Real-World Benchmarks...\n")
    
    benchmark_datasets = [
        ("Phishing Website Detection", datasets.Phishing(), 1250),
        ("Electricity Pricing", itertools.islice(datasets.Elec2(), 15000), 15000)
    ]
    
    covtype_stream = fetch_dynamic_dataset_stream("covtype", 15000)
    if covtype_stream:
        benchmark_datasets.append(("Forest Covertype", covtype_stream, 15000))
        
    airlines_stream = fetch_dynamic_dataset_stream("airlines", 15000)
    if airlines_stream:
        benchmark_datasets.append(("Airlines (Spatial/Geo)", airlines_stream, 15000))

    all_results = {}
    for dataset_name, stream, n_samples in benchmark_datasets:
        print(f"--- Evaluating on: {dataset_name} ({n_samples} samples) ---")
        models = make_models(n_models=10, n_clusters=5)
        window_size = 100 if n_samples < 5000 else 1000
        dataset_results = defaultdict(lambda: {"Rolling_Acc": Rolling(metrics.Accuracy(), window_size=window_size),
                                               "Cumulative_Acc": metrics.Accuracy(),
                                               "history_rolling": [], "total_time": 0.0,
                                               "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0})
        for i, (x, y) in enumerate(stream):
            for model_name, model in models.items():
                start_time = time.perf_counter()
                y_pred = model.predict_one(x)
                if y_pred is not None:
                    dataset_results[model_name]["Rolling_Acc"].update(y, y_pred)
                    dataset_results[model_name]["Cumulative_Acc"].update(y, y_pred)
                model.learn_one(x, y)
                dataset_results[model_name]["total_time"] += (time.perf_counter() - start_time)
                
                base_count, ens_count = get_total_drifts(model)
                if base_count > dataset_results[model_name]["last_drifts_base"]:
                    if not dataset_results[model_name]["drifts_base"] or (i - dataset_results[model_name]["drifts_base"][-1]) > 50:
                        dataset_results[model_name]["drifts_base"].append(i)
                    dataset_results[model_name]["last_drifts_base"] = base_count
                    
                if ens_count > dataset_results[model_name]["last_drifts_ens"]:
                    if not dataset_results[model_name]["drifts_ens"] or (i - dataset_results[model_name]["drifts_ens"][-1]) > 50:
                        dataset_results[model_name]["drifts_ens"].append(i)
                    dataset_results[model_name]["last_drifts_ens"] = ens_count
                
                if i % (max(1, window_size // 10)) == 0:
                    dataset_results[model_name]["history_rolling"].append(dataset_results[model_name]["Rolling_Acc"].get())
            if (i + 1) % max(1, (n_samples // 4)) == 0:
                print(f"  Processed {i + 1}/{n_samples} samples...")
        all_results[dataset_name] = dataset_results
        print(f"Finished {dataset_name}.\n")
    print("All streams processed. Generating plots and tables...")
    
    for dataset_name, _, n in benchmark_datasets:
        results = all_results[dataset_name]
        window_size = 100 if n < 5000 else 1000
        x_axis = range(0, n, max(1, window_size // 10))
        d_dets_base = {m: results[m]["drifts_base"] for m in results.keys()}
        d_dets_ens = {m: results[m]["drifts_ens"] for m in results.keys()}
        plot_time_series(x_axis, {m: results[m]["history_rolling"] for m in results.keys()},
                         title=f"Real-World Data: {dataset_name} (Window={window_size})", drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    
    data = []
    for dataset_name, _, n_samples in benchmark_datasets:
        results = all_results[dataset_name]
        for model_name in results.keys():
            acc = results[model_name]["Cumulative_Acc"].get()
            latency = (results[model_name]["total_time"] / n_samples) * 1000
            data.append({"Dataset": dataset_name, "Model": model_name, "Final Accuracy": f"{acc:.2%}", "Latency (ms/sample)": f"{latency:.2f} ms"})
    df = pd.DataFrame(data)
    df = df.sort_values(by=["Dataset", "Model"], ascending=[True, False])
    print(df.to_string(index=False))

def run_experiment_9():
    print("Initializing Experiment 9: Ablation Study - The Value of Dynamic Deactivation...")
    models = {
        "Vanilla SRP (Baseline)": VanillaSRP(n_models=10, drift_detector=ADWIN(), seed=42),
        "C-DES (Ablated: Pure Majority Vote)": CDES_SRP(n_models=10, n_clusters=2, disable_weighted_vote=True, drift_detector=ADWIN(), seed=42),
        "C-DES (Full: Soft Deactivation)": CDES_SRP(n_models=10, n_clusters=2, disable_weighted_vote=False, drift_detector=ADWIN(), seed=42)
    }
    results = defaultdict(lambda: {"Rolling_Acc": Rolling(metrics.Accuracy(), window_size=250), "Cumulative_Acc": metrics.Accuracy(), "history_rolling": [], "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0})
    stream = ConflictingContextStream(n_samples=12000, flicker_rate=3000)
    print("Processing stream...")
    for i, (x, y) in enumerate(stream):
        for name, model in models.items():
            y_pred = model.predict_one(x)
            if y_pred is not None:
                results[name]["Rolling_Acc"].update(y, y_pred)
                results[name]["Cumulative_Acc"].update(y, y_pred)
            model.learn_one(x, y)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
            
            if i % 20 == 0:
                results[name]["history_rolling"].append(results[name]["Rolling_Acc"].get())
        if (i + 1) % 3000 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plots...")
    x_axis = range(0, 12000, 20)
    r_drifts = [3000, 6000, 9000]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    plot_time_series(x_axis, {name: results[name]["history_rolling"] for name in results.keys()},
                     title="Ablation Study: Impact of Soft Deactivation on Conflicting Concepts",
                     real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)

def run_experiment_10():
    print("Initializing Experiment 10: The Quarantine Effect (Ensemble Churn)...")
    models = {
        "Vanilla SRP (Global Hard Resets)": VanillaSRP(n_models=10, drift_detector=ADWIN(delta=1e-4), seed=42),
        "C-DES (Local Soft Resets)": CDES_SRP(n_models=10, n_clusters=3, drift_detector=ADWIN(delta=1e-4), seed=42)
    }
    results = defaultdict(lambda: {"Stable_Acc": Rolling(metrics.Accuracy(), window_size=500),
                                   "history_stable_acc": [], "history_drifts": [], "drifts_base": [], "drifts_ens": [], "last_drifts_base": 0, "last_drifts_ens": 0})
    stream = NoiseQuarantineStream(n_samples=12000, noise_start=3000)
    print("Processing stream...")
    for i, (x, y, region) in enumerate(stream):
        for name, model in models.items():
            y_pred = model.predict_one(x)
            if y_pred is not None and region == "Stable Regions":
                results[name]["Stable_Acc"].update(y, y_pred)
            model.learn_one(x, y)
            
            base_count, ens_count = get_total_drifts(model)
            if base_count > results[name]["last_drifts_base"]:
                if not results[name]["drifts_base"] or (i - results[name]["drifts_base"][-1]) > 50:
                    results[name]["drifts_base"].append(i)
                results[name]["last_drifts_base"] = base_count
                
            if ens_count > results[name]["last_drifts_ens"]:
                if not results[name]["drifts_ens"] or (i - results[name]["drifts_ens"][-1]) > 50:
                    results[name]["drifts_ens"].append(i)
                results[name]["last_drifts_ens"] = ens_count
            
            if i % 50 == 0:
                results[name]["history_stable_acc"].append(results[name]["Stable_Acc"].get())
                results[name]["history_drifts"].append(base_count + ens_count)
        if (i + 1) % 3000 == 0:
            print(f"Processed {i + 1} samples...")
    print("Experiment complete. Generating plots...")
    x_axis = range(0, 12000, 50)
    r_drifts = [3000]
    d_dets_base = {n: results[n]["drifts_base"] for n in models.keys()}
    d_dets_ens = {n: results[n]["drifts_ens"] for n in models.keys()}
    plot_time_series(x_axis, {name: results[name]["history_stable_acc"] for name in results.keys()},
                     title="Collateral Damage: Accuracy on STABLE Regions Only",
                     real_drifts=r_drifts, drift_detections_base=d_dets_base, drift_detections_ens=d_dets_ens)
    plot_time_series(x_axis, {name: results[name]["history_drifts"] for name in results.keys()},
                     title="Ensemble Churn: Cumulative Drift Detections Triggered",
                     real_drifts=r_drifts)

def run_all_experiments():
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    run_experiment_7()
    run_experiment_8()
    run_experiment_9()
    run_experiment_10()