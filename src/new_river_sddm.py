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

        # Słowniki do typowania i normalizacji
        self._is_nominal = {}
        self._nominal_maps = {}
        self._mins = {}
        self._maxs = {}
        self._limits_set = False

        # cur_window trzyma surowe obserwacje (potrzebne do re-binningu po drifcie)
        self.cur_window = deque(maxlen=cur_window_size)
        # ref_bin_queue przechowuje TYLKO indeksy binów i klas dla przesunięć okna sliding
        self.ref_bin_queue = deque() if incremental else deque(maxlen=ref_window_size)
        self.ref_element_count = 0

        self.t = 0
        self._drift_detected = False
        self._drift_mag = 0.0
        self._source_feature = None
        self._last_drifts_list = []

    def _init_structures(self, x, y):
        self.feature_names = list(x.keys())
        self.n_features = len(self.feature_names)
        self.feature_map = {name: i for i, name in enumerate(self.feature_names)}
        
        if self.n_classes is None:
            self.n_classes = int(y + 1) if y is not None else 1
        
        # Weryfikacja typu na podstawie pierwszej instancji
        for name, val in x.items():
            if isinstance(val, str) or not isinstance(val, (int, float)):
                self._is_nominal[name] = True
                self._nominal_maps[name] = {}  # Słownik: wartość_nominalna -> id_binu
            else:
                self._is_nominal[name] = False
                self._mins[name] = float('inf')
                self._maxs[name] = float('-inf')

        self.ref_XY = np.zeros((self.n_features, self.n_classes, self.n_bins))
        self.cur_XY = np.zeros((self.n_features, self.n_classes, self.n_bins))
        self._last_drifts_list = [0.0] * self.n_features
        self.ref_element_count = 0

    def _bin(self, feature_name, value):
        """Mapuje wartość na indeks kubła z osobną logiką dla nominalnych i ciągłych."""
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

    def _update_hist(self, x, y, hist, add=True):
        y_idx = int(y) if y is not None else 0
        if y_idx >= self.n_classes:
            return 

        for name, val in x.items():
            if name in self.feature_map:
                f_idx = self.feature_map[name]
                b_idx = self._bin(name, val)
                hist[f_idx, y_idx, b_idx] += 1 if add else -1

    def _recalculate_limits_from_cur_window(self):
        """Wyznacza nowe buckety (ciągłe i nominalne) wyłącznie na podstawie cur_window."""
        for name in self.feature_names:
            if self._is_nominal.get(name, False):
                self._nominal_maps[name] = {}
            else:
                self._mins[name] = float('inf')
                self._maxs[name] = float('-inf')
            
        for cx, _ in self.cur_window:
            for name, val in cx.items():
                if self._is_nominal.get(name, False):
                    nom_map = self._nominal_maps[name]
                    if val not in nom_map:
                        nom_map[val] = len(nom_map) % self.n_bins
                else:
                    if val < self._mins[name]: self._mins[name] = val
                    if val > self._maxs[name]: self._maxs[name] = val

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
            self.cur_window.append((x, y))
            return self

        self.t += 1
        self._drift_detected = False

        if not self._limits_set:
            self.cur_window.append((x, y))
            if len(self.cur_window) >= self.cur_window_size:
                # 1. Ustalenie koszyków na podstawie pierwszego pełnego okna
                self._recalculate_limits_from_cur_window()
                
                # 2. Reset i przeniesienie elementów do bazy referencyjnej H_ref
                self.ref_XY.fill(0)
                self.ref_bin_queue.clear()
                self.ref_element_count = 0
                
                for cx, cy in self.cur_window:
                    cy_idx = int(cy) if cy is not None else 0
                    if cy_idx < self.n_classes:
                        bin_indices = [self._bin(name, cx.get(name, 0)) for name in self.feature_names]
                        for f_idx, b_idx in enumerate(bin_indices):
                            self.ref_XY[f_idx, cy_idx, b_idx] += 1
                        
                        if not self.incremental:
                            self.ref_bin_queue.append((cy_idx, bin_indices))
                        else:
                            self.ref_element_count += 1
                
                # 3. Wyczyszczenie okna bieżącego – zaczyna zbierać strumień od zera
                self.cur_window.clear()
                self.cur_XY.fill(0)
                self._limits_set = True
            return self

        # --- MECHANIZM PRZESUWANIA OKNA (Sliding / Incremental) ---
        if len(self.cur_window) >= self.cur_window_size:
            passed_x, passed_y = self.cur_window.popleft()
            self._update_hist(passed_x, passed_y, self.cur_XY, add=False)

            y_idx = int(passed_y) if passed_y is not None else 0
            if y_idx < self.n_classes:
                bin_indices = [self._bin(name, passed_x.get(name, 0)) for name in self.feature_names]
                
                for f_idx, b_idx in enumerate(bin_indices):
                    self.ref_XY[f_idx, y_idx, b_idx] += 1

                if not self.incremental:
                    self.ref_bin_queue.append((y_idx, bin_indices))
                    if len(self.ref_bin_queue) > self.ref_window_size:
                        old_y_idx, old_bins = self.ref_bin_queue.popleft()
                        for f_idx, b_idx in enumerate(old_bins):
                            self.ref_XY[f_idx, old_y_idx, b_idx] -= 1
                else:
                    self.ref_element_count += 1

        self.cur_window.append((x, y))
        self._update_hist(x, y, self.cur_XY, add=True)

        # --- WARUNEK URUCHOMIENIA TESTU STATYSTYCZNEGO ---
        ready_to_test = False
        if len(self.cur_window) >= self.cur_window_size:
            if not self.incremental and len(self.ref_bin_queue) >= self.ref_window_size:
                ready_to_test = True
            elif self.incremental and self.ref_element_count >= self.ref_window_size:
                ready_to_test = True

        if ready_to_test and (self.t % self.test_interval == 0):
            drifts = []
            for f in range(self.n_features):
                p = self.cur_XY[f].flatten()
                q = self.ref_XY[f].flatten()
                drifts.append(self._kl_divergence(p, q))

            self._last_drifts_list = drifts
            self._drift_mag = max(drifts)
            
            if self._drift_mag > self.threshold:
                self._drift_detected = True
                self._source_feature = self.feature_names[np.argmax(drifts)]
                
                if self.printer:
                    r = self.get_drift_report()
                    print(f" [DRIFT @ {self.t}] Feature: {r['source']} | Mag: {r['magnitude']}")

                # --- POPRAWIONA STRATEGIA RESETU (Zgodna z metodologią) ---
                # 1. Rekalkulacja limitów matematycznych wyłącznie na bazie próbek z W_cur (nowa koncepcja)
                self._recalculate_limits_from_cur_window()
                
                # 2. Czyszczenie starych struktur historycznych
                self.ref_bin_queue.clear()
                self.cur_XY.fill(0)
                self.ref_XY.fill(0)
                self.ref_element_count = 0
                
                # 3. Transfer koncepcji: Budowanie nowego baseline H_ref z elementów zawartych w W_cur
                for cx, cy in self.cur_window:
                    cy_idx = int(cy) if cy is not None else 0
                    if cy_idx < self.n_classes:
                        bin_indices = [self._bin(name, cx.get(name, 0)) for name in self.feature_names]
                        for f_idx, b_idx in enumerate(bin_indices):
                            self.ref_XY[f_idx, cy_idx, b_idx] += 1
                        
                        if not self.incremental:
                            self.ref_bin_queue.append((cy_idx, bin_indices))
                        else:
                            self.ref_element_count += 1
                
                # 4. Wyczyszczenie okna bieżącego w celu zbierania nowych, świeżych próbek
                self.cur_window.clear()
                
                # Zostawiamy _limits_set = True, ponieważ nowe limity zostały już zamrożone.
                # Testy zostaną automatycznie wstrzymane (ready_to_test = False), dopóki 
                # okna ponownie nie wypełnią się świeżymi danymi ze strumienia.
                self._limits_set = True 
                
        return self
    
    def get_drift_report(self):
        ref_size = self.ref_element_count if self.incremental else len(self.ref_bin_queue)
        return {
            "detected": self._drift_detected,
            "magnitude": self._drift_mag,
            "source": self._source_feature,
            "ref_size": ref_size,
            "limits_set": self._limits_set,
            "all_feature_drifts": self._last_drifts_list
        }