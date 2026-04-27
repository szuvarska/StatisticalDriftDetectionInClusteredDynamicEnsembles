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
        printer = False
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

        # Statystyki do normalizacji
        self._mins = {}
        self._maxs = {}
        self._limits_set = False

        self.cur_window = deque(maxlen=cur_window_size)
        ref_maxlen = None if incremental else ref_window_size
        self.ref_window = deque(maxlen=ref_maxlen)

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
            # Próba estymacji liczby klas, jeśli nie podano
            self.n_classes = int(y + 1) if y is not None else 1
        
        # Inicjalizacja min/max wartościami ekstremalnymi
        for name in self.feature_names:
            self._mins[name] = float('inf')
            self._maxs[name] = float('-inf')

        self.ref_XY = np.zeros((self.n_features, self.n_classes, self.n_bins))
        self.cur_XY = np.zeros((self.n_features, self.n_classes, self.n_bins))
        self._last_drifts_list = [0.0] * self.n_features

    def _bin(self, feature_name, value):
        """Mapuje wartość na indeks kubła używając wyliczonych granic min-max."""
        f_min = self._mins[feature_name]
        f_max = self._maxs[feature_name]
        
        if f_max == f_min:
            return 0
        
        # Normalizacja min-max do zakresu [0, 1]
        norm_val = (value - f_min) / (f_max - f_min)
        idx = int(norm_val * self.n_bins)
        
        # Zabezpieczenie przed wartościami poza pierwotnym zakresem (clamping)
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

        # --- FAZA 1: Zbieranie granic (Warm-up / Re-scaling) ---
        if not self._limits_set:
            # Aktualizuj globalne min/max dla nowych danych
            for name, val in x.items():
                if val < self._mins[name]: self._mins[name] = val
                if val > self._maxs[name]: self._maxs[name] = val
            
            self.ref_window.append((x, y))
            
            # Czekamy na zapełnienie okna referencyjnego, aby "zamrozić" nowe granice
            if len(self.ref_window) >= self.ref_window_size:
                # Wypełnij histogram referencyjny na podstawie nowo zebranych granic
                self.ref_XY.fill(0) # Upewniamy się, że jest czysty
                for old_x, old_y in self.ref_window:
                    self._update_hist(old_x, old_y, self.ref_XY, add=True)
                self._limits_set = True
                
                # Ważne: czyścimy okno bieżące, aby zaczęło zbierać dane od zera po warm-upie
                self.cur_window.clear()
                self.cur_XY.fill(0)
                
            return self

        # --- FAZA 2: Normalne działanie (Sliding Window) ---
        
        # Zarządzanie oknami i histogramami
        if len(self.cur_window) >= self.cur_window.maxlen:
            passed_x, passed_y = self.cur_window.popleft()
            self._update_hist(passed_x, passed_y, self.cur_XY, add=False)

            # Przesunięcie najstarszej próbki z bieżącego do referencyjnego
            if not self.incremental:
                if len(self.ref_window) >= self.ref_window_size:
                    old_ref_x, old_ref_y = self.ref_window.popleft()
                    self._update_hist(old_ref_x, old_ref_y, self.ref_XY, add=False)
            
            self.ref_window.append((passed_x, passed_y))
            self._update_hist(passed_x, passed_y, self.ref_XY, add=True)

        self.cur_window.append((x, y))
        self._update_hist(x, y, self.cur_XY, add=True)

        # Testowanie dryfu: 
        # Dodany warunek: len(self.cur_window) >= self.cur_window.maxlen
        # Gwarantuje to, że oba okna są pełne przed obliczeniem KL Divergence
        if (self.t % self.test_interval == 0 and 
            len(self.ref_window) >= self.ref_window_size and 
            len(self.cur_window) >= self.cur_window.maxlen):
            
            drifts = []
            for f in range(self.n_features):
                # Używamy spłaszczonych rozkładów (Feature x Class)
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

                # --- POWRÓT DO FAZY 1 ---
                self._limits_set = False 
                
                # Resetowanie struktur
                self.ref_window.clear()
                self.cur_window.clear()
                self.ref_XY.fill(0)
                self.cur_XY.fill(0)
                
                # Resetowanie granic min/max dla ponownego skalowania
                for name in self.feature_names:
                    self._mins[name] = float('inf')
                    self._maxs[name] = float('-inf')
                
        return self
    
    def get_drift_report(self):
        return {
            "detected": self._drift_detected,
            "magnitude": self._drift_mag,
            "source": self._source_feature,
            "ref_size": len(self.ref_window),
            "limits_set": self._limits_set,
            "all_feature_drifts": self._last_drifts_list
        }