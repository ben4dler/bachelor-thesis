# Shapes und Transformationen: Welches Modell erwartet welche Form?

Stand: 26.11.2026. Bezieht sich auf `preprocess.py`, `epoch_creator.py`, `functions.py` und die Notebooks in `notebooks/BunDLe-Net/notebooks/`.

Diese Notiz erklärt die **Regeln**. Sie enthält keine fertigen Lösungen für deine Notebooks. Am Ende stehen Übungen, mit denen du prüfen kannst, ob du es verstanden hast.

---

## 0. Die eine Frage, die alles entscheidet

> **Was ist bei diesem Modell eine Zeile, also *ein* Beispiel (Sample)?**

Jede Shape-Frage lässt sich darauf zurückführen. Je nach Analyse ist ein Beispiel etwas anderes:

| Analyse | Ein Beispiel ist … | Label pro Beispiel |
|---|---|---|
| Spatio-temporaler Klassifikator | eine ganze Epoche | Kategorie der Epoche |
| Sliding Estimator | eine Epoche, **zu einem Zeitpunkt** (für jeden Zeitpunkt ein eigenes Modell) | Kategorie der Epoche |
| „Gepoolter“ Klassifikator pro Zeitpunkt | ein Paar (Epoche, Zeitpunkt) | Label zu genau diesem Zeitpunkt |
| BunDLe-Net | ein Fensterpaar (Epoche, Startpunkt) | Label am letzten Zeitpunkt des zweiten Fensters |

Sobald du weißt, was ein Beispiel ist, folgt die Shape von selbst:
- **X:** erste Achse = Anzahl Beispiele
- **y:** erste (und einzige) Achse = **dieselbe** Anzahl Beispiele, in **derselben Reihenfolge**

---

## 1. Bezeichnungen

| Symbol | Bedeutung | Bei dir (pro Proband, ungefähr) |
|---|---|---|
| E | Anzahl Epochen | ca. 1638 Train / 410 Test (80/20 aus 2048) |
| C | Anzahl Kanäle | 64 |
| T | Zeitpunkte pro Epoche | −0,2 bis 0,8 s bei 100 Hz → ca. 101 |
| W | Fensterlänge (`WINDOW_LENGTH`) | 15 (= 150 ms) |
| P | Fensterpositionen pro Epoche | T − W |
| D | latente Dimension von τ | 3 |
| K | Anzahl Klassen | 4 Kategorien (+ Klasse 0 „vor dem Bild“ bei Labels pro Zeitpunkt) |

⚠️ Die konkreten Zahlen ändern sich, wenn du Epochenfenster, Abtastrate oder Probanden änderst. Ältere Notebook-Ausgaben zeigen z. B. noch T = 120 und P = 105. **Gewöhn dir an, nach jedem Schritt `.shape` auszugeben**, statt dich auf diese Tabelle zu verlassen.

---

## 2. Woher die Daten kommen

```
Raw (kontinuierlich)
  │  mne.Epochs(..., tmin=-0.2, tmax=0.8) + resample(100)
  ▼
epochs.get_data()                     → (E, C, T)      ← MNE-Konvention
preprocessor.scale(...)               → (E, C, T)      ← Shape bleibt gleich
create_labels_for_epochs(epochs)      → labels:             (E, T)   Label pro Zeitpunkt
                                        category_per_epoch: (E,)     Label pro Epoche
```

Zwei verschiedene Label-Arrays, für zwei verschiedene Fragen:
- `category_per_epoch` (E,): Werte 1–4. Für alles, wo **eine Epoche = ein Beispiel** ist.
- `labels` (E, T): Werte 0–4. 0 vor dem Bild, danach die Kategorie. Für alles, wo **ein Zeitpunkt = ein Beispiel** ist (BunDLe-Net, gepoolter Dummy).

⚠️ `create_labels_for_epochs` gibt `labels` als **float32** zurück. sklearn kommt damit zurecht, nennt die Klassen dann aber `1.0, 2.0, …`. TensorFlows `SparseCategoricalCrossentropy` braucht **ganze Zahlen**. Deshalb steht im Evaluationscode `.astype(np.int64)`.

---

## 3. Die drei „Sprachen“: MNE, sklearn, Keras

### 3.1 sklearn: immer 2D
Fast jedes sklearn-Modell (LogReg, LDA, SVM, PCA, StandardScaler, DummyClassifier) erwartet:

```
X: (n_samples, n_features)     ← genau 2 Achsen
y: (n_samples,)                ← genau 1 Achse
```

Gibst du 3D hinein, kommt `ValueError: Found array with dim 3. Estimator expected <= 2`.
Gibst du y als (E, T) hinein, wird es als Multi-Output interpretiert, oder es kommt ein Fehler.

### 3.2 MNE-Decoding: 3D, Zeit hinten
MNE-Werkzeuge arbeiten mit der Epochen-Form und **erwarten die Zeit als letzte Achse**:

```
X: (n_epochs, n_channels, n_times)
y: (n_epochs,)
```

### 3.3 Keras / BunDLe-Net: was die erste Schicht verlangt
Keras erwartet `(batch, ...)`. Was hinter `batch` kommt, legt das Modell fest. Bei BunDLe-Net ist die **Zeit vor den Kanälen** (siehe Abschnitt 6). Das ist genau umgekehrt zu MNE.

---

## 4. Übersicht: welches Werkzeug welche Form

| Werkzeug | Input X | y | Output |
|---|---|---|---|
| `DummyClassifier` | (n, beliebig) — X wird ignoriert, nur `len(X)` zählt | (n,) | Vorhersage (n,) |
| `LogisticRegression`, `LinearDiscriminantAnalysis`, `SVC` | (n, f) | (n,) | (n,) |
| `StandardScaler` (sklearn) | (n, f) | – | (n, f), skaliert **pro Feature** |
| `PCA` (sklearn) | (n, f) | – | (n, k) |
| `LDA.transform` | (n, f) | (n,) beim Fit | (n, min(K−1, f)) |
| `mne.decoding.Vectorizer` | (E, C, T) | – | (E, C·T) |
| `mne.decoding.Scaler` | (E, C, T) | – | (E, C, T), skaliert **pro Kanal** |
| `mne.decoding.UnsupervisedSpatialFilter(PCA(k))` | (E, C, T) | – | (E, k, T) |
| `mne.decoding.CSP` | (E, C, T) | (E,) | (E, n_components) |
| `mne.decoding.SlidingEstimator(clf)` | (E, F, T) | (E,) | `score` → (T,) |
| `mne.decoding.GeneralizingEstimator(clf)` | (E, F, T) | (E,) | `score` → (T, T) |
| `cross_val_multiscore(SlidingEstimator, …)` | (E, F, T) | (E,) | (n_folds, T) |
| `pyriemann XdawnCovariances` | (E, C, T) | (E,) | (E, k, k) |
| `pyriemann TangentSpace` | (E, k, k) | – | (E, k(k+1)/2) |
| `BunDLeNet` (`model(X)`) | (B, 2, W, C) | – | τ(t+1): (B, D), Ŷ(t+1): (B, D), Logits: (B, 5) |
| `model.tau` | (B, W, C) | – | (B, D) |

---

## 5. Der Sliding Estimator im Detail

`SlidingEstimator` ist im Kern eine Schleife:

```
für jeden Zeitpunkt t in 0 … T−1:
    X_t = X[:, :, t]            → (E, F)   ← 2D, also passend für sklearn
    clf_t = Kopie von clf
    clf_t.fit(X_t, y)           → y: (E,)
```

Daraus folgt:
1. **Die Zeit muss die letzte Achse sein.** Ist sie es nicht, „sliden“ die Schritte über die falsche Achse, und das Modell läuft trotzdem ohne Fehler durch.
2. **Die mittlere Achse F sind die Features.** Was dort steht, ist dem SWE egal:
   - rohe Kanäle → F = C = 64
   - τ-Werte → F = D = 3
   - Rohfenster → F = C · W = 960
3. **Der Basis-Klassifikator bekommt 2D.** Deshalb darf in der Pipeline `StandardScaler` (sklearn) stehen und kein `Vectorizer`.
4. **y ist pro Epoche**, also `category_per_epoch` und nicht `labels`.

Der `GeneralizingEstimator` erwartet dieselben Formen, testet aber das Modell von t auf allen t'.

### Spatio-temporal (alles auf einmal)
```
X (E, C, T) ──Vectorizer──▶ (E, C·T) ──StandardScaler──▶ ──LogReg──▶ (E,)
```
Hier gibt es *ein* Modell und *eine* Zahl als Ergebnis, keine Zeitkurve.

---

## 6. BunDLe-Net im Detail

### Was `build_training_batch` baut
```python
one_epoch = eeg_data[epoch_number].T     # (C, T) → (T, C)   ← hier passiert der Achsentausch!
first_window  = one_epoch[s     : s + W]         # (W, C)
second_window = one_epoch[s + 1 : s + W + 1]     # (W, C)
np.stack([first, second])                         # (2, W, C)
```
Über einen Batch von B Fensterpaaren: **X_batch: (B, 2, W, C)**, **y_batch: (B,)** mit dem Label `labels[epoch, s + W]`.

### Was das Modell damit macht (`functions.py`, `BunDLeNet.call`)
```
X[:, 1]  (B, W, C) ──τ: Flatten──▶ (B, W·C) ──Dense…──▶ Y(t+1): (B, D) ──predictor──▶ (B, 5) Logits
X[:, 0]  (B, W, C) ──τ──▶ Y(t): (B, D) ──+T_Y──▶ Ŷ(t+1): (B, D)
```

Die Ausgabeschicht des Predictors hat **5** Einheiten: Klasse 0 („vor dem Bild“) + 4 Kategorien. Ändert sich die Klassenzahl, muss sich diese Zahl mitändern.

`Flatten` macht aus (W, C) einen Vektor der Länge W·C. Die Reihenfolge ist „Zeit außen, Kanal innen“. Für das Netz ist egal, wie die Reihenfolge ist, solange sie **in Training und Test gleich** ist.

### Zurück in die Sliding-Estimator-Form
τ gibt pro Fenster einen Vektor aus. Für den SWE brauchst du wieder (E, F, T):

```
τ-Ausgaben, eine Zeile pro Fenster            → (E·P, D)
  reshape  (Reihenfolge: Epoche außen, Start innen,
            so wie list_all_window_pairs sie erzeugt)  → (E, P, D)
  transpose(0, 2, 1)                           → (E, D, P)   ← Zeit wieder hinten
```

Der `reshape` funktioniert **nur**, weil `list_all_window_pairs` die Fenster in genau dieser Reihenfolge erzeugt. Wenn du die Fenster vorher mischst, musst du sie vorher wieder in die richtige Reihenfolge bringen.

### Zeitachse
Der τ-Wert an Position p gehört zum Zeitpunkt `times[W + p]`, also zum letzten Zeitpunkt des zweiten Fensters. Die τ-Kurve hat P statt T Punkte und beginnt erst W Zeitpunkte nach Epochenbeginn.

---

## 7. reshape und transpose: die häufigste Fehlerquelle

### Regel 1: `reshape` sortiert nie um
`reshape` liest die Zahlen in Speicherreihenfolge („letzte Achse läuft am schnellsten“) und schneidet sie neu zu. Er weiß nichts davon, was die Achsen *bedeuten*.

Beispiel mit E = 2, C = 2, T = 3, Werte als `e c t` beschriftet:
```
X[0] = [[000 001 002]     ← Kanal 0
        [010 011 012]]    ← Kanal 1

X.reshape(-1, C)   ❌ (ohne transpose)
→ [[000 001], [002 010], [011 012], …]     ← mischt Zeit und Kanäle!

X.transpose(0, 2, 1).reshape(-1, C)   ✅
→ (E, T, C) → [[000 010], [001 011], [002 012], [100 110], …]
   Zeile = (Epoche, Zeitpunkt), Spalten = Kanäle
```

**Faustregel:** Bevor du Achsen zusammenlegst, schiebe sie mit `transpose` so, dass die Achsen, die zusammengelegt werden, **nebeneinander und vorne** stehen. Die Achse, die übrig bleiben soll, kommt **nach hinten**.

### Regel 2: X und y müssen *gleich* geplättet werden
Wenn X so aussieht: Zeile k = (Epoche e, Zeitpunkt t) mit „Epoche außen, Zeit innen“,
dann muss y genauso aussehen. `labels` hat die Form (E, T), und `labels.reshape(-1)` ergibt Epoche außen, Zeit innen. Das passt zu `X.transpose(0, 2, 1).reshape(-1, C)`.

Prüfe es immer mit einem Mini-Array, wie oben, bevor du es auf echten Daten machst.

### Regel 3: Nach jedem Schritt die Shape ansehen
```
print(X.shape, y.shape)   # erste Achse gleich?
```
Wenn die ersten Achsen nicht übereinstimmen, wirft sklearn `Found input variables with inconsistent numbers of samples`. Das ist der häufigste Fehler überhaupt, und er bedeutet immer: **X und y beschreiben nicht dieselben Beispiele.**

---

## 8. Wann wird transformiert, und auf welchen Daten wird gefittet?

Shapes sind die eine Hälfte. Die andere ist: **Alles, was aus Daten lernt, darf nur die Trainingsdaten sehen.**

| Schritt | lernt aus Daten? | fit auf | transform auf |
|---|---|---|---|
| Filter, Notch, Resample | nein | – | alles |
| Baseline-Korrektur | nein (pro Epoche) | – | alles |
| ICA | **ja**, aber unüberwacht | im Feld üblich: alles. Begründen! | alles |
| `Scaler` / `StandardScaler` | **ja** (Mittelwert, SD) | Train | Train + Test |
| PCA, CSP, xDAWN, MVNN | **ja** | Train | Train + Test |
| LDA, LogReg, SVM | **ja** | Train | Test |
| τ (BunDLe-Net) | **ja, mit Labels** | Train | Train + Test |

Der sicherste Weg: Alles, was lernt, kommt in eine `make_pipeline(...)`. Dann fittet `cross_val_multiscore` es automatisch in jedem Fold neu und nur auf dem Trainingsteil.

τ passt in keine sklearn-Pipeline. Deshalb ist es der Fall, bei dem du selbst darauf achten musst (siehe Notiz zur CV).

---

## 9. Häufige Fehler und was sie bedeuten

| Symptom | Wahrscheinliche Ursache |
|---|---|
| `Found array with dim 3. Estimator expected <= 2` | 3D-Daten an ein sklearn-Modell ohne `Vectorizer` oder `SlidingEstimator` |
| `inconsistent numbers of samples` | X und y nicht gleich geplättet, oder `labels` (E, T) statt `category_per_epoch` (E,) |
| SWE-Kurve hat 64 statt T Punkte | Zeit ist nicht die letzte Achse (Kanäle und Zeit vertauscht) |
| Accuracy genau auf Zufall, keine Fehlermeldung | Labels und Daten passen in der Reihenfolge nicht zusammen (falsch geplättet oder einzeln gemischt) |
| Klassen heißen `1.0, 2.0 …` | Labels sind float |
| Keras: `Incompatible shapes` beim Modellaufruf | Achsenreihenfolge (W, C) gegen (C, W) vertauscht, oder W ≠ Fensterlänge beim Bau des Modells |
| `SparseCategoricalCrossentropy` meckert über Werte | Labels sind float oder außerhalb von 0 … 4 |
| Gute Test-Accuracy, die zu gut aussieht | Scaling, PCA oder τ wurden auf allen Daten gefittet |

---

## 10. Übungen

Am besten mit Bleistift und anschließend mit `.shape` im Notebook prüfen.

1. **Dummy-Notebook** (`dummy_classifier.ipynb`): Du plättest X zu (E·T, C).
   - Welche Shape hat `y_train` dort, und welche müsste es haben?
   - Für welche Frage ist dieser Aufbau gedacht, „pro Epoche“ oder „pro Zeitpunkt“? Welches Label-Array passt dazu?
   - `strategy='constant'` mit `constant='M'`: Kommt die Klasse `'M'` in deinen Labels vor?
   - Welche Zahl sollte `most_frequent` ungefähr liefern, wenn du an den Anteil der Klasse 0 denkst?
2. Du willst den **SWE auf Rohfenstern** (Plan B2). Welche Shape muss X haben? Hinweis: F = ?, letzte Achse = ? Wie viele Punkte hat die Kurve, und zu welchen Zeitpunkten gehören sie?
3. Du willst **PCA(3) pro Zeitpunkt** über die Kanäle machen und danach den SWE. Welches Werkzeug aus Tabelle 4 nimmt (E, C, T) und gibt (E, 3, T) zurück? Muss es in die Pipeline des SWE oder davor? (Denk an Abschnitt 8.)
4. **LDA als 3D-Projektion:** LDA ist ein sklearn-Modell, also 2D. Was ist bei „LDA pro Zeitpunkt“ ein Beispiel, und wie viele Komponenten liefert `transform` bei 4 Klassen?
5. Warum steht in `build_training_batch` das `.T`? Was würde passieren, wenn es fehlt und W = C wäre? Würde dann eine Fehlermeldung kommen?

---

## Quellen
- MNE-Tutorial „Decoding (MVPA)“: https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html (zeigt alle Shapes für `Vectorizer`, `SlidingEstimator`, `GeneralizingEstimator`)
- MNE-API `mne.decoding`: https://mne.tools/stable/api/decoding.html
- sklearn Glossar, Einträge *X*, *y*, *n_samples*, *n_features*: https://scikit-learn.org/stable/glossary.html
- NumPy: `reshape`, `transpose` und die Speicherreihenfolge (C-Order): https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
- Keras: `Flatten` und das Batch-Konzept: https://keras.io/api/layers/reshaping_layers/flatten/

> Die Links stammen aus Claudes Wissen und wurden nicht online geprüft. Falls einer nicht mehr geht, über den Titel suchen.
