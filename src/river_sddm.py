from collections import deque
import numpy as np
from river import base

class RiverSDDM(base.DriftDetector):
    def __init__(
        self,
        n_classes=None,
        n_bins=20,
        ref_window_size=400,
        cur_window_size=100,
        test_interval=10,
        threshold=0.6,
        alpha=1.0,
        incremental=True,
        printer=False
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.alpha = alpha
        self.ref_window_size = ref_window_size
        self.cur_window_size = cur_window_size
        self.test_interval = test_interval
        self.threshold = threshold
        self.incremental = incremental
        self.printer = printer

        self.feature_names = None
        self.n_features = 0
        self.feature_map = None
        self.ref_XY = None
        self.cur_XY = None

        # Feature types and normalization statistics
        self._is_nominal = {}
        self._nominal_maps = {}
        self._mins = {}
        self._maxs = {}
        self._limits_set = False

        self.cur_window = deque(maxlen=cur_window_size)
        
        # Memory optimization for the reference window 
        # (stores only bin indices or a simple counter to avoid keeping raw data)
        self.ref_bin_queue = deque() 
        self.ref_count = 0
        
        # Temporary buffer required EXCLUSIVELY for proper warm-up / re-scaling
        self._warmup_buffer = []

        self.t = 0
        self._drift_detected = False
        self._drift_mag = 0.0
        self._source_feature = None
        self._last_drifts_list = []

        # Window to track the history of drift magnitudes (e.g., last 50 measurements)
        # Used for dynamic Z-score thresholding
        self.mag_window = deque(maxlen=50)
        
        # Track current mean and std internally to avoid breaking Z-test logic
        self._current_mag_mean = 0.0
        self._current_mag_std = 0.0

    def _init_structures(self, x, y):
        self.feature_names = list(x.keys())
        self.n_features = len(self.feature_names)
        self.feature_map = {name: i for i, name in enumerate(self.feature_names)}
        
        if self.n_classes is None:
            self.n_classes = int(y + 1) if y is not None else 1
        
        for name, val in x.items():
            if isinstance(val, str) or not isinstance(val, (int, float)):
                self._is_nominal[name] = True
                self._nominal_maps[name] = {}
            else:
                self._is_nominal[name] = False
                self._mins[name] = float('inf')
                self._maxs[name] = float('-inf')

        self.ref_XY = np.zeros((self.n_features, self.n_classes, self.n_bins))
        self.cur_XY = np.zeros((self.n_features, self.n_classes, self.n_bins))
        self._last_drifts_list = [0.0] * self.n_features

    def _bin(self, feature_name, value):
        if self._is_nominal.get(feature_name, False):
            nom_map = self._nominal_maps[feature_name]
            if value not in nom_map:
                nom_map[value] = len(nom_map) % self.n_bins
            return nom_map[value]

        f_min = self._mins[feature_name]
        f_max = self._maxs[feature_name]
        
        if f_max == f_min:
            return 0
        
        norm_val = (value - f_min) / (f_max - f_min)
        idx = int(norm_val * self.n_bins)
        return min(max(idx, 0), self.n_bins - 1)

    def _get_bin_indices(self, x):
        return [self._bin(name, x.get(name, 0)) for name in self.feature_names]

    def _update_hist(self, x, y, hist, add=True):
        y_idx = int(y) if y is not None else 0
        if y_idx >= self.n_classes:
            return 

        for name, val in x.items():
            if name in self.feature_map:
                f_idx = self.feature_map[name]
                b_idx = self._bin(name, val)
                hist[f_idx, y_idx, b_idx] += 1 if add else -1

    def _update_hist_with_bins(self, y, bins, hist, add=True):
        """Helper method for the optimized queue (uses bin indices directly)."""
        y_idx = int(y) if y is not None else 0
        if y_idx >= self.n_classes:
            return

        for f_idx, b_idx in enumerate(bins):
            hist[f_idx, y_idx, b_idx] += 1 if add else -1

    def _kl_divergence(self, p_counts, q_counts):
        p = p_counts + self.alpha
        q = q_counts + self.alpha
        p /= p.sum()
        q /= q.sum()
        kl = 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
        return round(float(kl), 4)

    def update(self, x, y=None):
        if self.feature_names is None:
            self._init_structures(x, y)

        self.t += 1
        self._drift_detected = False

        # --- PHASE 1: Boundary Collection (Warm-up / Re-scaling) ---
        if not self._limits_set:
            self._warmup_buffer.append((x, y))
            
            # CALCULATE LIMITS ONLY WHEN THE BUFFER IS FULL
            if len(self._warmup_buffer) >= self.ref_window_size:
                
                # Step 1: Scan the entire collected buffer to establish global min/max boundaries
                for cx, cy in self._warmup_buffer:
                    for name, val in cx.items():
                        if self._is_nominal.get(name, False):
                            nom_map = self._nominal_maps[name]
                            if val not in nom_map:
                                nom_map[val] = len(nom_map) % self.n_bins
                        else:
                            if val < self._mins[name]: self._mins[name] = val
                            if val > self._maxs[name]: self._maxs[name] = val

                # Step 2: With boundaries set, bin all buffered data into the reference histogram (H_ref)
                self.ref_XY.fill(0) 
                for old_x, old_y in self._warmup_buffer:
                    bins = self._get_bin_indices(old_x)
                    self._update_hist_with_bins(old_y, bins, self.ref_XY, add=True)
                    
                    if not self.incremental:
                        self.ref_bin_queue.append((old_y, bins))
                    else:
                        self.ref_count += 1
                
                # Freeze limits and clean up initialization structures
                self._limits_set = True
                self.cur_window.clear()
                self.cur_XY.fill(0)
                self._warmup_buffer.clear()
                
            return self

        # --- PHASE 2: Normal Operation (Sliding Window) ---
        if len(self.cur_window) >= self.cur_window.maxlen:
            # Remove the oldest element from the current window
            passed_x, passed_y = self.cur_window.popleft()
            self._update_hist(passed_x, passed_y, self.cur_XY, add=False)

            # Manage the reference baseline capacity (if non-incremental)
            if not self.incremental:
                if len(self.ref_bin_queue) >= self.ref_window_size:
                    old_ref_y, old_ref_bins = self.ref_bin_queue.popleft()
                    self._update_hist_with_bins(old_ref_y, old_ref_bins, self.ref_XY, add=False)

            # Move the instance that left the current window into the reference baseline
            passed_bins = self._get_bin_indices(passed_x)
            self._update_hist_with_bins(passed_y, passed_bins, self.ref_XY, add=True)
            
            # UNCONDITIONAL addition to the reference queue
            if not self.incremental:
                self.ref_bin_queue.append((passed_y, passed_bins))
            else:
                self.ref_count += 1

        # Add the newest element to the current window
        self.cur_window.append((x, y))
        self._update_hist(x, y, self.cur_XY, add=True)

        # --- PHASE 3: Statistical Drift Testing ---
        current_ref_size = self.ref_count if self.incremental else len(self.ref_bin_queue)
        
        if (self.t % self.test_interval == 0 and 
            current_ref_size >= self.ref_window_size and 
            len(self.cur_window) >= self.cur_window.maxlen):
            
            drifts = []
            for f in range(self.n_features):
                p = self.cur_XY[f].flatten()
                q = self.ref_XY[f].flatten()
                drifts.append(self._kl_divergence(p, q))

            self._last_drifts_list = drifts
            self._drift_mag = max(drifts)

            # Calculate historical stats BEFORE adding the new magnitude (Z-test requirement)
            if len(self.mag_window) == self.mag_window.maxlen:
                self._current_mag_mean = float(np.mean(self.mag_window))
                self._current_mag_std = float(np.std(self.mag_window))
            else:
                self._current_mag_mean = 0.0
                self._current_mag_std = 0.0

            # Add current magnitude to the history window
            self.mag_window.append(self._drift_mag)
            
            # Trigger drift if divergence exceeds the baseline threshold
            if self._drift_mag > self.threshold:
                self._drift_detected = True
                self._source_feature = self.feature_names[np.argmax(drifts)]
                
                if self.printer:
                    r = self.get_drift_report()
                    print(f" [DRIFT @ {self.t}] Feature: {r['source']} | Mag: {r['magnitude']}")

                # --- RETURN TO PHASE 1 WITH DATA TRANSFER ---
                self._limits_set = False 
                
                # Clean transfer of rescued observations from the current window back to the warm-up buffer
                self._warmup_buffer.clear()
                self._warmup_buffer.extend(self.cur_window)
                
                # Full memory reset of active structures
                self.ref_bin_queue.clear()
                self.ref_count = 0
                self.cur_window.clear()
                self.ref_XY.fill(0)
                self.cur_XY.fill(0)
                
                # Reset normalization boundaries
                for name in self.feature_names:
                    if self._is_nominal.get(name, False):
                        self._nominal_maps[name].clear()
                    else:
                        self._mins[name] = float('inf')
                        self._maxs[name] = float('-inf')
                
        return self
    
    def get_drift_report(self):
        """Generates a comprehensive report of the latest drift test."""
        return {
            "detected": self._drift_detected,
            "magnitude": self._drift_mag,
            "source": self._source_feature,
            "ref_size": self.ref_count if self.incremental else len(self.ref_bin_queue),
            "limits_set": self._limits_set,
            "all_feature_drifts": self._last_drifts_list,
            "mag_mean": self._current_mag_mean,
            "mag_std": self._current_mag_std
        }