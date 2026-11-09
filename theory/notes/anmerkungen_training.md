# Anmerkungen zum Training von BunDLe-Net auf EEG (`check_eeg_data.ipynb`)

Stand: 05.11.2026. Bezieht sich auf das Training mit Producer/Consumer-Threads (`fetch_data` / `consume_data` / `train_block`).

---

## Teil 1: Mögliche grobe Fehler, die das Training beeinträchtigen

Hier stehen nur Richtungen zum Nachdenken, keine fertigen Lösungen.

### 1. Der Behaviour-Loss liegt auf Zufallsniveau (wichtigstes Warnsignal)
- Angezeigt wird `(1 - gamma) · Cross-Entropy`, etwa 0,18 bis 0,19. Zurückgerechnet ist die Cross-Entropy damit etwa 1,8 bis 1,9.
- Rechne aus, welchen Loss ein Modell hätte, das nur rät: einmal bei gleich häufigen Klassen, einmal bei der tatsächlichen Klassenverteilung in `B`.
- Vergleiche diesen Wert mit deinem Loss.
- Schau dir an, wie häufig jedes Label in `B` überhaupt vorkommt.

### 2. Skalierung der Eingabedaten
- In welcher Einheit gibt `raw.get_data()` die Werte zurück, und wie groß sind sie typischerweise?
- Was kann ein Netz mit ReLU-Schichten und `GaussianNoise(0.05)` aus so kleinen Zahlen machen?
- Der auffällig kleine DCC-Loss passt dazu. Bekommt das Netz kaum unterscheidbare Eingaben, bildet es alles auf fast denselben Punkt ab. Dann ist auch der DCC-Loss klein, obwohl nichts gelernt wurde.

### 3. Zeitskala
- Wie viele Millisekunden deckt ein Fenster mit `win = 15` bei deiner Abtastrate ab?
- Wie groß ist der Schritt von t nach t+1, und wie verschieden sind zwei so dicht beieinanderliegende Fenster?
- In welchem Zeitraum nach dem Bild zeigt sich im EEG, ob es ein Gesicht, ein Tier, ein Ort oder ein Objekt war? Stichworte: ERP-Komponenten, Downsampling.
- Vergleiche das mit der Zeitauflösung der Wurmdaten aus dem Original-BunDLe-Net.

### 4. Was die Labels genau bedeuten
- `generate_B` hält ein Label so lange, bis das nächste Event kommt.
- Was passiert laut Events-Datei und Datensatzbeschreibung zwischen zwei Events? Wie lange ist das Bild tatsächlich zu sehen?
- Beschreibt das Label einen Zustand, der zum Hirnsignal in genau diesem Moment passt?

### 5. Die Vorverarbeitung fehlt komplett
- Rohes EEG enthält Drift, Netzbrummen, Augenartefakte und eine Referenz.
- Welche Vorverarbeitung ist für diesen Datensatz oder für ähnliche Studien zur Objekterkennung üblich?

### 6. Es gibt keine faire Kontrolle
- Der Loss fällt am Ende ab, aber der letzte Block ist sehr klein, nur 585 Beispiele.
- Woran würdest du erkennen, dass das Modell etwas lernt, das auch auf ungesehenen Daten gilt?

**Tipp:** Fang mit den Punkten 1 und 2 an. Beide kannst du mit wenigen Ausgaben prüfen, ohne neu zu trainieren.

---

## Teil 2: Technische Hinweise zum Code

- **Modell, Optimizer, Trainer und `gamma` nur einmal anlegen**, außerhalb der Schleife. Sonst fängt jeder Block mit einem untrainierten Netz an. Ein neuer `BunDLeTrainer` bedeutet außerdem, dass das `@tf.function` von `train_step` neu kompiliert wird.
- **Ändern sich `win`, `block_size` oder Ähnliches, muss das Modell neu angelegt werden.** Einmal ist ein Modell, das mit `win=15` gebaut wurde, auf Daten mit `win=5` gelaufen, was einen Shape-Fehler ausgelöst hat.
- **`gamma=0.9,`** mit Komma am Ende ist ein Tupel, keine Zahl.
- **`UnboundLocalError`:** Wird eine Variable irgendwo in einer Funktion neu zugewiesen (`x = ...`), behandelt Python sie in der ganzen Funktion als lokal. Losses deshalb in einer Liste mit `.append` sammeln und erst zum Plotten in ein Array umwandeln.
- **`return` im `finally`** (in `train_model`) verschluckt jeden Fehler.
- **Threads und Queue:**
  - Stürzt der Consumer ab, holt niemand mehr Blöcke ab. Die Queue (`maxsize=2`) läuft voll, `fetch_data` wartet bei `q.put` für immer, und `producer.join()` blockiert das Notebook.
  - Dagegen helfen ein Stopp-Signal (`None` als letztes Element), `try/finally` mit `q.task_done()` und ein `threading.Event`, das den Producer anhält.
  - Nach einem Absturz den Kernel neu starten, weil alte Daemon-Threads sonst im Hintergrund weiterhängen.
  - Alternative ohne eigene Threads: `tf.data.Dataset.from_generator(...).prefetch(...)`.
- **Mischen (Shuffling):**
  - Gemischt werden die fertigen Beispiele `(X_paired[i], B_1[i])`, nicht das Rohsignal. Jedes Beispiel enthält seine Zeitstruktur (Fenster t und t+1) selbst, und die Verlustfunktion schaut immer nur in ein Beispiel.
  - Fenster und Labels mit denselben Indizes auswählen, damit sie zusammenbleiben.
  - `from_tensor_slices` kopiert alle überlappenden Fenster in den Speicher. Besser ist es, nur die Indizes zu permutieren.
  - Erst zusammenhängende Zeitabschnitte in Train und Test aufteilen, dann nur die Trainingsdaten mischen. Sonst landen fast identische, überlappende Fenster in beiden Teilen.
  - Bisher wird nur innerhalb eines Blocks gemischt. Springt der Loss stark zwischen den Blöcken, hilft ein Puffer, der Beispiele aus mehreren Blöcken mischt.

---

## Teil 3: Quellen

### 3.1 Grundlagen: EEG und ereigniskorrelierte Potenziale (ERP)
Betrifft die Punkte 3, 4 und 5.

- **Steven J. Luck**: *An Introduction to the Event-Related Potential Technique*, 2. Aufl., MIT Press, 2014. Das Standard-Einstiegsbuch.
- **Steven J. Luck**: *Applied Event-Related Potential Data Analysis*, frei online auf LibreTexts. Praxisnah: Filter, Referenz, Artefakte, Epochen.
- **Mike X Cohen**: *Analyzing Neural Time Series Data: Theory and Practice*, MIT Press, 2014. Signalverarbeitung für EEG. Dazu gibt es frei verfügbare Vorlesungsvideos auf YouTube.

### 3.2 EEG praktisch mit MNE vorverarbeiten
- **MNE-Tutorials**, Abschnitte „Preprocessing“ und „Epoching“: https://mne.tools/stable/auto_tutorials/index.html
- **Gramfort et al. (2013)**: *MEG and EEG data analysis with MNE-Python*. Frontiers in Neuroscience.
- **Pernet et al. (2019)**: *EEG-BIDS, an extension to the brain imaging data structure for electroencephalography*. Scientific Data. Erklärt die Struktur des Datensatzes (`events.tsv`, `channels.tsv`).
- **Beschreibung des Datensatzes ds005363 auf OpenNeuro**: Dort steht, was die Event-Codes bedeuten.

### 3.3 Objektkategorien aus EEG dekodieren
- **Grootswagers, Wardle & Carlson (2017)**: *Decoding dynamic brain patterns from evoked responses: A tutorial on multivariate pattern analysis applied to time series neuroimaging data*. Journal of Cognitive Neuroscience. **Für diese Arbeit vermutlich der wichtigste Text.**
- **Carlson et al. (2013)**: *Representational dynamics of object vision: The first 1000 ms*. Journal of Vision.
- **Cichy, Pantazis & Oliva (2014)**: *Resolving human object recognition in space and time*. Nature Neuroscience.
- **King & Dehaene (2014)**: *Characterizing the dynamics of mental representations: the temporal generalization method*. Trends in Cognitive Sciences.

### 3.4 Neuronale Netze sauber trainieren und debuggen
Betrifft die Punkte 1, 2 und 6.

- **Andrej Karpathy (2019)**: *A Recipe for Training Neural Networks*, https://karpathy.github.io/2019/04/25/recipe/. **Am besten als Erstes lesen.**
- **Goodfellow, Bengio & Courville (2016)**: *Deep Learning*, frei unter https://www.deeplearningbook.org. Vor allem Kapitel 5, 8 und 11 (Practical Methodology).
- **Aurélien Géron**: *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, 3. Aufl., O'Reilly.
- **Varoquaux et al. (2017)**: *Assessing and tuning brain decoders: cross-validation, caveats, and guidelines*. NeuroImage. Fair evaluieren, ohne dass Testdaten ins Training durchsickern.

### 3.5 Deep Learning speziell für EEG
- **Roy et al. (2019)**: *Deep learning-based electroencephalography analysis: a systematic review*. Journal of Neural Engineering.
- **Schirrmeister et al. (2017)**: *Deep learning with convolutional neural networks for EEG decoding and visualization*. Human Brain Mapping. Dazu die Bibliothek **Braindecode**.
- **Lawhern et al. (2018)**: *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering.

### 3.6 BunDLe-Net und verwandte Methoden
- **Kumar et al. (2023)**: *BunDLe-Net: Neuronal Manifold Learning Meets Behaviour*. bioRxiv. Code: https://github.com/akshey-kumar/BunDLe-Net
- **Kato et al. (2015)**: *Global brain dynamics embed the motor command sequence of Caenorhabditis elegans*. Cell. Die Wurmdaten aus dem Demo-Notebook. Ihre Zeitauflösung mit dem EEG vergleichen.
- **Schneider, Lee & Mathis (2023)**: *Learnable latent embeddings for joint behavioural and neural analysis* (CEBRA). Nature.
- **Cunningham & Yu (2014)**: *Dimensionality reduction for large-scale neural recordings*. Nature Neuroscience.

### 3.7 Technik: Python und TensorFlow
- Python `queue`: https://docs.python.org/3/library/queue.html
- Python `threading`: https://docs.python.org/3/library/threading.html
- `try`/`finally`: https://docs.python.org/3/reference/compound_stmts.html#the-try-statement
- tf.data-Guide: https://www.tensorflow.org/guide/data
- tf.data-Performance: https://www.tensorflow.org/guide/data_performance
- Eigene Trainingsschleife: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
- `tf.function` und Retracing: https://www.tensorflow.org/guide/function
- MNE `Raw.get_data`: https://mne.tools/stable/generated/mne.io.Raw.html

---

## Vorschlag zur Reihenfolge
1. Karpathys Blogpost lesen, ungefähr eine Stunde
2. Die Beschreibung des Datensatzes auf OpenNeuro durchgehen
3. Bei Luck die Kapitel zu ERPs und Vorverarbeitung, parallel dazu die MNE-Tutorials
4. Das Tutorial von Grootswagers et al.
5. Das BunDLe-Net-Paper noch einmal lesen, mit dem neuen Wissen im Hinterkopf

> Hinweis: Die Quellen stammen aus dem Wissen von Claude und wurden nicht online geprüft. Falls ein Link nicht mehr geht, findest du alles über Google Scholar mit Autor und Titel. Viele Bücher gibt es über die Uni-Bibliothek kostenlos online.


Die Kurzfassung: Die Architektur bleibt gleich, nur die Eingangsdaten ändern sich. Vorher war es eine einzige lange Aufnahme, jetzt sind es viele kurze. Mit jeder kurzen Aufnahme machst du dasselbe wie vorher mit der langen. Im Einzelnen:

---

## 1. Was BunDLe-Net erwartet (unverändert)

`prep_data(X, B, win)` braucht **eine** zusammenhängende Zeitreihe:
- `X`: Form **(Zeit, Kanäle)**
- `B`: Form **(Zeit,)**, also ein Label pro Zeitpunkt

Daraus baut es Trainingsbeispiele. Jedes Beispiel ist ein Paar aus zwei Fenstern, die um einen Zeitschritt verschoben sind, plus ein Label:

```
Zeit:     0  1  2  3  4  5  6  7 ...
          [  t-Fenster ]
             [ t+1-Fenster ]
                       ↑ Label = B[t + win]  (letzter Zeitpunkt des t+1-Fensters)
```

Mit `win=3` ergibt das:
- Beispiel 0: Fenster `X[0:3]` und `X[1:4]`, Label `B[3]`
- Beispiel 1: Fenster `X[1:4]` und `X[2:5]`, Label `B[4]`
- …

Ergebnis: `X_paired` mit Form **(Anzahl Paare, 2, win, Kanäle)** und `B_1` mit Form **(Anzahl Paare,)**. Genau diese Formen frisst das Modell. **Daran ändert sich nichts.**

---

## 2. Vorher: eine lange Zeitreihe

```
X: (≈2 Mio. Zeitpunkte, 64 Kanäle)    ← ganze Aufnahme
B: (≈2 Mio.,)                          ← generate_B: Label gilt bis zum nächsten Event
```
`prep_data` lief einmal über alles, deshalb waren Blöcke und Threads nötig.

---

## 3. Jetzt: viele kurze Zeitreihen

Epochs zerschneiden die Aufnahme in 2048 Stücke von je 1 s (250 Zeitpunkte bei 250 Hz). Nach `get_data()` hast du:

```
X_train: (1638, 64, 250)
          │     │    └─ Zeit innerhalb der Epoch (−0.3 s … 0.7 s)
          │     └────── Kanal
          └──────────── welche Epoch
```

`X_train[5]` ist also eine eigenständige **Mini-Aufnahme** mit der Form (64, 250), nur liegen die Achsen andersherum als bei `prep_data`. Deshalb steht dort `.T`, und `X_train[5].T` hat die Form (250, 64), also (Zeit, Kanäle), wie `prep_data` es will.

---

## 4. B: ein Label pro Zeitpunkt, pro Epoch

Früher war `B` ein langer Vektor. Jetzt braucht jede Mini-Aufnahme ihr eigenes `B`, also bekommst du eine Matrix mit **gleicher Form wie X, nur ohne Kanalachse**: (1638, 250).

`make_B` Zeile für Zeile:

```python
codes = ep.events[:, 2]
```
Das ist der Event-Code jeder Epoch, also welches Bild gezeigt wurde, z. B. `[3, 20, 51, ...]`.

```python
cat = (codes - 1) // 16 + 1
```
Damit wird aus der Bildnummer die Kategorie, weil je 16 Bilder eine Kategorie bilden:
- 3 → (2 // 16) + 1 = **1** (Gesicht)
- 20 → (19 // 16) + 1 = **2** (Tier)
- 51 → (50 // 16) + 1 = **4** (Objekt)

```python
B = np.zeros((len(ep), len(ep.times)), dtype=int)
```
Damit beginnt jede Epoch mit Label 0 an jedem Zeitpunkt.

```python
B[:, ep.times >= 0] = cat[:, None]
```
`ep.times` sind die Zeitpunkte in Sekunden (−0.3 … 0.7). Ab dem Bildbeginn (t ≥ 0) wird die Kategorie eingetragen:

```
              Zeit: −0.3 ......... 0 ............ 0.7
Epoch 0 (Bild 3):   0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
Epoch 1 (Bild 20):  0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2
Epoch 2 (Bild 51):  0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4
```

Die 0 vor dem Bild ist eine bewusste Entscheidung: Dort hat das Gehirn das Bild noch nicht gesehen, also gibt es auch nichts Kategoriespezifisches zu erkennen.

---

## 5. Epochs in die bisherige Architektur übersetzen

Die einfachste und verständlichste Variante: `prep_data` **für jede Epoch einzeln** aufrufen und die Ergebnisse aneinanderhängen.

```python
Xp_list, Bp_list = [], []
for i in range(len(X_train)):
    Xp, Bp = prep_data(X_train[i].T, B_train[i], win=win)   # wie früher, nur für 1 Epoch
    Xp_list.append(Xp)
    Bp_list.append(Bp)

X_paired = np.concatenate(Xp_list)   # (1638 × 225, 2, 25, 64)
B_1      = np.concatenate(Bp_list)   # (1638 × 225,)
```
Bei 250 Zeitpunkten und `win=25` liefert jede Epoch 225 Paare.

Danach ist alles wie früher, du gibst `X_paired` und `B_1` ans Training. Weil jede Epoch einzeln gepaart wird, gibt es **nie ein Paar, das über zwei Epochs hinweg geht**. Das wäre falsch, weil zwischen zwei Epochs in Wirklichkeit Zeit liegt.

**Einziges Problem:** `np.concatenate` kopiert alles in den Speicher, das sind etwa 4,7 GB.

---

## 6. Die speichersparende Variante (`get_batch`)

Sie macht **exakt dasselbe** wie Abschnitt 5, baut die Paare aber erst, wenn ein Batch gebraucht wird. Ich habe nachgeprüft, dass beide Varianten identische Werte liefern.

**Schritt 1: Jedes Paar durch zwei Zahlen beschreiben.**
Ein Paar ist eindeutig festgelegt durch *welche Epoch* und *wo das Fenster startet*:
```python
ep_idx, t_idx = np.meshgrid(np.arange(len(X_train)), np.arange(T - win), indexing="ij")
ep_idx, t_idx = ep_idx.ravel(), t_idx.ravel()
```
Bei 2 Epochs und 3 Startpunkten sieht das so aus:
```
ep_idx = [0, 0, 0, 1, 1, 1]
t_idx  = [0, 1, 2, 0, 1, 2]
```
Das ist eine Liste aller Paare, aber nur als Zahlen. Bei dir sind das 1638 × 225 ≈ 370 000 Einträge, also ein paar MB statt GB.

**Schritt 2: Mischen.** `perm` mischt diese Liste. Ein Batch sind dann z. B. 1024 zufällige (Epoch, Start)-Kombinationen.

**Schritt 3: Für diese 1024 die echten Daten ausschneiden.**
```python
W = X[ep[:, None], :, t[:, None] + offs]      # offs = [0, 1, ..., win]
```
Für jedes Paar wird aus Epoch `ep` der Abschnitt `t … t+win` geholt, also `win+1` Zeitpunkte über alle Kanäle. Das entspricht `X_train[ep][:, t:t+win+1]`, nur für alle 1024 auf einmal. Die Form ist **(1024, win+1, 64)**.

```python
x = np.stack([W[:, :-1], W[:, 1:]], axis=1)
```
Aus den `win+1` Zeitpunkten entstehen die beiden Fenster:
- `W[:, :-1]` sind die Zeitpunkte `t … t+win-1` (t-Fenster)
- `W[:, 1:]` sind die Zeitpunkte `t+1 … t+win` (t+1-Fenster)

Die Form ist **(1024, 2, win, 64)**, genau wie `X_paired` aus `prep_data`.

```python
b = B[ep, t + win]
```
Das Label am letzten Zeitpunkt des t+1-Fensters, genau wie `B[win:]` in `prep_data`.

---

## Zusammengefasst

| | Vorher | Jetzt |
|---|---|---|
| Zeitreihen | 1 lange | 2048 kurze (Epochs) |
| X | (Zeit, Kanäle) | (Epochs, Kanäle, Zeit), pro Epoch `.T` |
| B | (Zeit,) via `generate_B` | (Epochs, Zeit) via `make_B` |
| Paare bilden | `prep_data` einmal | `prep_data` pro Epoch, oder `get_batch` mit derselben Logik |
| Ans Modell | (Paare, 2, win, Kanäle) + (Paare,) | **identisch** |

Am Modell, am Loss und an `train_step` ändert sich nichts. Neu ist nur, woher die Paare kommen.



# Epochs oder keine

Nimm die Epochs. Genau genommen schließen sich die beiden Wege nicht aus, denn sie lösen zwei verschiedene Probleme:

- **Multithreading** regelt, *wie* Daten geladen werden. Es war ein Workaround, weil die ganze Aufnahme bei 1000 Hz mit allen Fensterpaaren nicht in den Speicher passte.
- **Epochs** regeln, *welche* Daten mit *welchem* Label ins Training gehen.

Mit Epochs bei 250 Hz belegen die Daten nur etwa 130 MB. Damit fällt der Grund für das Multithreading weg.

**Warum Epochs besser sind:**
- **Richtige Labels:** Jedes Stück gehört eindeutig zu einem Bild, gezählt ab Bildbeginn. Bei `generate_B` galt ein Label bis zum nächsten Event, auch in Pausen und nach dem Ende der Hirnantwort.
- **Weniger Fehlerquellen:** Kein Producer/Consumer, keine Queue, die hängen bleibt, kein Neustart des Kernels nach Abstürzen.
- **Sauberer Test:** Du teilst nach Epochs in Train und Test auf, ohne dass sich überlappende Fenster in beiden Teilen wiederfinden.
- **Standard im Feld:** Fast alle Studien zur Objekterkennung im EEG, auch die zu deinem Datensatz, arbeiten mit Epochs. Das lässt sich in der Arbeit leicht begründen und vergleichen.

**Was du dafür aufgibst:** BunDLe-Net wurde für kontinuierliche Daten entwickelt (die Würmer), und mit Epochs siehst du nur die Dynamik innerhalb von 1 Sekunde rund um ein Bild. Was zwischen den Bildern passiert, fällt weg. Für deine Frage, ob sich die Bildkategorien im latenten Raum trennen, ist genau dieser Ausschnitt aber der interessante. Diese Einschränkung solltest du in der Arbeit kurz erwähnen.

Falls du später doch kontinuierlich arbeiten willst, brauchst du auch dafür kein Multithreading: Gefilterte Raw-Daten bei 250 Hz passen ebenfalls in den Speicher, und das Paaren über Indizes funktioniert genauso wie jetzt mit `build_training_batch`.


