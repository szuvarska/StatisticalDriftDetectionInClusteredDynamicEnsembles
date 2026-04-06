import numpy as np
from scipy.stats import entropy

class FullSDDM:
    def __init__(
        self,
        p_size,
        num_features,
        num_classes,           # DODANE: Wymagane dla stabilności wymiarów na realnych danych
        num_bins=None,         # Zmienione na opcjonalne (dobierane automatycznie)
        distance="kl",         # "kl", "hellinger", "tv"
        threshold=None,        # Zmienione na opcjonalne (dobierane automatycznie)
        use_posterior=True,
        use_class=True,
        drift_type="abrupt"    # "abrupt" (0.6) lub "gradual" (0.3)
    ):
        self.p_size = p_size
        self.num_features = num_features
        self.num_classes = num_classes
        self.distance = distance
        
        self.use_posterior = use_posterior
        self.use_class = use_class

        # --- AUTOMATYCZNY DOBÓR PARAMETRÓW (na bazie Tabeli 3 z artykułu) ---
        # Liczba binów (Tabela 3)
        if num_bins is None:
            self.num_bins = 5 if p_size < 500 else 10
        else:
            self.num_bins = num_bins

        # Próg odcięcia (Tabela 3: 0.6 dla abrupt, 0.3 dla gradual)
        if threshold is None:
            self.threshold = 0.6 if drift_type == "abrupt" else 0.3
        else:
            self.threshold = threshold

        # Rozmiar okna historycznego (Tabela 3: m * p_size)
        self.w_multiplier = 20 if p_size < 500 else 10
        self.max_batches_in_w = self.w_multiplier

        # --- STAN WEWNĘTRZNY ---
        self.hist_X = []
        self.hist_Y = []
        self.hist_XY = []

        self.last_drift_index = 0
        self.t = 0

    # ============================================================
    # HISTOGRAMY (Z poprawionym zakresem dla klas)
    # ============================================================
    def _histogram(self, x, bins, data_range=(0, 1)):
        counts, _ = np.histogram(x, bins=bins, range=data_range)
        return counts + 1.0  # Wygładzanie Laplace'a (alpha=1)

    def _joint_histogram(self, x, y):
        # Generowanie wspólnego rozkładu P(X, Y) z użyciem stałej liczby klas
        h = []
        for f in range(self.num_features):
            joint = []
            for cls in range(self.num_classes):
                mask = (y == cls)
                # Nawet jeśli maska jest pusta, liczymy histogram (wyjdą same zera), 
                # a wygładzanie Laplace'a zapobiegnie dzieleniu przez zero.
                counts, _ = np.histogram(
                    x[mask, f], bins=self.num_bins, range=(0, 1)
                )
                joint.append(counts + 1.0)
            h.append(np.array(joint))
        return np.array(h)

    # ============================================================
    # DISTANCE
    # ============================================================
    def _distance(self, p, q):
        p = p / np.sum(p)
        q = q / np.sum(q)

        if self.distance == "kl":
            # Symetryczna dywergencja KL
            return (entropy(p, q) + entropy(q, p)) / 2.0
        elif self.distance == "hellinger":
            return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)
        elif self.distance == "tv":
            return 0.5 * np.sum(np.abs(p - q))
        else:
            raise ValueError("Unknown distance measure")

    # ============================================================
    # DRIFT COMPONENTS
    # ============================================================
    def _covariate_drift(self, curr_X, win_X):
        return np.array([self._distance(curr_X[f], win_X[f]) for f in range(self.num_features)])

    def _class_drift(self, curr_Y, win_Y):
        return self._distance(curr_Y, win_Y)

    def _posterior_drift(self, curr_XY, win_XY):
        return np.array([
            self._distance(curr_XY[f].flatten(), win_XY[f].flatten()) 
            for f in range(self.num_features)
        ])

    # ============================================================
    # MAIN
    # ============================================================
    def process_batch(self, X_batch, y_batch):
        # Sprawdzenie założeń o normalizacji danych wejściowych
        if np.max(X_batch) > 1.0 or np.min(X_batch) < 0.0:
            raise ValueError("Dane X muszą być znormalizowane do przedziału [0, 1] przed podaniem do SDDM!")

        self.t += 1

        # --- Current histograms ---
        curr_X = np.array([
            self._histogram(X_batch[:, f], self.num_bins, data_range=(0, 1))
            for f in range(self.num_features)
        ])

        # Y_batch musi być zakodowane jako integer od 0 do num_classes-1
        curr_Y = self._histogram(y_batch, bins=self.num_classes, data_range=(0, self.num_classes))
        curr_XY = self._joint_histogram(X_batch, y_batch)

        # --- Adaptive window (Reducing the propagation of old drifts) ---
        effective_window = min(len(self.hist_X), self.t - self.last_drift_index - 1)

        if effective_window <= 0:
            # Zbyt mało danych do porównania historycznego, dodajemy tylko do okna
            self._update(curr_X, curr_Y, curr_XY)
            return {
                "drift": False, "magnitude": 0.0,
                "covariate": np.zeros(self.num_features),
                "posterior": np.zeros(self.num_features),
                "class": 0.0
            }

        win_X = np.sum(self.hist_X[-effective_window:], axis=0)
        win_Y = np.sum(self.hist_Y[-effective_window:], axis=0)
        win_XY = np.sum(self.hist_XY[-effective_window:], axis=0)

        # --- Compute components ---
        cov_drift = self._covariate_drift(curr_X, win_X)
        
        post_drift = (
            self._posterior_drift(curr_XY, win_XY)
            if self.use_posterior else np.zeros(self.num_features)
        )

        class_drift = self._class_drift(curr_Y, win_Y) if self.use_class else 0.0

        # --- Decyzja (Opieramy się na Posterior Drifcie, tak jak nakazuje paper dla zmiany koncepcji) ---
        if self.use_posterior:
            decision_magnitude = np.max(post_drift)  # Opcja "Max per-feature magnitude" z Tabeli 3
        else:
            decision_magnitude = np.max(cov_drift)   # Fallback na wirtualny dryf
            
        drift_detected = decision_magnitude > self.threshold

        if drift_detected:
            self.last_drift_index = self.t

        self._update(curr_X, curr_Y, curr_XY)

        return {
            "drift": bool(drift_detected),
            "magnitude": float(decision_magnitude),
            "covariate": cov_drift,
            "posterior": post_drift,
            "class": float(class_drift)
        }

    # ============================================================
    # UPDATE
    # ============================================================
    def _update(self, X, Y, XY):
        self.hist_X.append(X)
        self.hist_Y.append(Y)
        self.hist_XY.append(XY)

        if len(self.hist_X) > self.max_batches_in_w:
            self.hist_X.pop(0)
            self.hist_Y.pop(0)
            self.hist_XY.pop(0)