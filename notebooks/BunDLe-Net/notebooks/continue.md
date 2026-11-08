# Leitfaden: Sliding Estimator als Vergleich zu BunDLe-Net

Ziel dieses Dokuments: Du weißt jederzeit, **wo du stehst**, **was als Nächstes kommt** und **wo noch Denkfehler stecken**. Lösungen stehen hier bewusst nicht drin, sondern Fragen, Hinweise und Quellen.

---

## 1. Wo du gerade stehst

**Erledigt (`random_guesser.ipynb`, Proband 101):**
- Gesichts- und Objekt-Epochs ausgewählt (1024 Epochs)
- Drei Methoden ausprobiert: spatio-temporal (85,9 %), CSP (50,2 %), Sliding Estimator (AUC-Kurve über die Zeit)
- Patterns-Grafik (Kopfkarten) erstellt

**In Arbeit (`epoch_model.ipynb`, Proband 106):**
- BunDLe-Net wird auf Fensterpaaren von 25 Zeitpunkten (100 ms) trainiert
- Ziel-Label: 0 vor dem Bild, danach Kategorie 1–4

**Offene Hauptfrage:** Wie vergleichst du die beiden Modelle fair miteinander?

---

## 2. Was der Sliding Estimator bisher gezeigt hat

| Methode | Ergebnis | Bedeutung |
|---|---|---|
| Spatio-temporal (alle Zeitpunkte zusammen) | 85,9 % | Gesicht und Objekt lassen sich trennen, aber du weißt nicht, *wann*. |
| CSP | 50,2 % | Nur Raten. Die Methode passt nicht zur Frage (siehe unten). |
| **Sliding Estimator** | AUC-Kurve | Ein eigenes Modell pro Zeitpunkt |

**Die Decoding-Kurve:**
- **t < 0:** Die Kurve liegt um 0,5 (Ratequote) mit etwa ±0,05 Rauschen. Das ist plausibel, denn vor dem Bild gibt es noch nichts zu unterscheiden.
- **ab ca. 80 ms:** Die Kurve steigt auf ca. 0,65. Das ist früh, wahrscheinlich spielen hier einfache Bildeigenschaften (Helligkeit, Kontrast, Form) eine Rolle.
- **Gipfel bei ca. 150–170 ms, AUC ≈ 0,93:** Das ist das Zeitfenster der **N170**, der typischen Gesichtsreaktion.
- **danach 0,6–0,75:** Die Information bleibt erhalten, wird aber schwächer.

**Patterns-Grafik:** Bei 0,1 und 0,2 s gibt es Flecken hinten links und rechts (etwa P7/P8), also da, wo man die N170 typischerweise misst. Die Einheit ist wegen der Skalierung willkürlich, aussagekräftig sind nur Form und Verteilung.

**Warum CSP bei 50 % liegt:** CSP sucht Unterschiede in der *Varianz*, also in der Stärke von Schwingungen. Dein Unterschied steckt aber im *Spannungsverlauf* (dem Ausschlag der N170). Das ist also kein Bug, die Methode passt nur nicht zu deiner Frage.

> Die AUC-Kurve gibt es **nicht pro Millisekunde**, sondern pro Zeitpunkt. Bei 250 Hz liegen die Zeitpunkte **4 ms** auseinander. Achte darauf, wenn du das in der Arbeit beschreibst.

---

## 3. Denkfehler und offene Baustellen

Arbeite diese Liste ab. Zu jedem Punkt gibt es eine Stelle zum Nachschauen und eine Frage. Hake einen Punkt erst ab, wenn du die Frage in eigenen Worten beantworten kannst.

### A. Im Sliding-Estimator-Notebook

- [ ] **A1 – Datenleck durch Skalierung**
  *Wo:* `random_guesser.ipynb`, die Zelle mit `scaler.fit_transform(X_train_face_obj)`, die **vor** `cross_val_multiscore` läuft.
  *Frage:* Welche Daten hat der Scaler gesehen, bevor die Kreuzvalidierung Training und Test trennt? Die Pipeline enthält schon einen `StandardScaler`. Was ändert sich, wenn du die äußere Skalierung weglässt?

- [ ] **A2 – Dieselben Bilder in Training und Test**
  *Wo:* `epochs.events[:, 2]`, also die Bildnummer.
  *Frage:* Jedes der 32 Bilder kommt etwa 32-mal vor. Lernt das Modell „Gesicht“ oder „Bild Nr. 7“? Wie könntest du die Folds so bauen, dass ein Bild *nur* im Training oder *nur* im Test vorkommt?
  *Recherche:* `sklearn.model_selection.GroupKFold`

- [ ] **A3 – Der Sliding Estimator ist kein „dummes“ Modell**
  *Frage:* Das Notebook heißt `random_guesser`, aber der Sliding Estimator rät nicht, sondern ist ein **lineares Vergleichsmodell**. Was wäre deine echte Untergrenze, also ein Modell, das die EEG-Daten gar nicht ansieht?
  *Recherche:* `sklearn.dummy.DummyClassifier`

- [ ] **A4 – „Über dem Zufall“ nur nach Augenmaß**
  *Frage:* Ab welchem Zeitpunkt liegt die AUC *statistisch* über 0,5? Wie sähe die Kurve aus, wenn du die Labels zufällig vertauschst? Und warum ist es ein Problem, 250 Zeitpunkte einzeln zu testen?
  *Recherche:* Permutationstest, Mehrfachtest-Problem, Cluster-Permutationstest

- [ ] **A5 – Die eigene Interpretation nicht überziehen**
  *Frage:* Die Patterns-Grafik zeigt P7/P8. Beweist das schon, dass das Modell „Gesicht“ als Kategorie gelernt hat? Was bedeutet A2 für diese Aussage?

### B. Beim Vergleich mit BunDLe-Net

Füll zuerst diese Tabelle aus, indem du beide Notebooks nebeneinanderlegst:

| Was | Sliding Estimator | BunDLe-Net | Gleich? |
|---|---|---|---|
| Proband | | | |
| Vorverarbeitung (Filter, ICA, Ablehnung) | | | |
| Welche Klassen / wie viele? | | | |
| Label für t < 0 | | | |
| Metrik | | | |
| Train/Test-Aufteilung | | | |
| Zeit, die das Modell pro Vorhersage sieht | | | |
| Ein Modell pro Zeitpunkt oder eins für alle? | | | |

Die folgenden Punkte solltest du dabei entdecken:

- [ ] **B1 – Unterschiedliche Probanden**
  *Wo:* `subject=` in beiden Notebooks.
  *Frage:* Was kannst du über „Modell A ist besser als Modell B“ sagen, wenn A und B unterschiedliche Gehirne gesehen haben?

- [ ] **B2 – Unterschiedliche Vorverarbeitung**
  *Wo:* Suche in beiden Notebooks nach `ICA`.
  *Frage:* Wenn nur eines der beiden Notebooks Artefakte entfernt, woran liegt dann ein Unterschied im Ergebnis: am Modell oder an den Daten?

- [ ] **B3 – Unterschiedliche Aufgabe**
  *Wo:* `create_labels_for_epochs` in `epoch_model.ipynb` und der `predictor` mit `Dense(5)` in `functions.py`.
  *Frage:* Der Sliding Estimator unterscheidet 2 Klassen, BunDLe-Net 5. Ist 60 % bei 5 Klassen besser oder schlechter als 60 % bei 2 Klassen? Und: Ist „vor oder nach dem Bild“ (Label 0) dieselbe Frage wie „Gesicht oder Objekt“? Wie leicht ist es zu erkennen, dass das Bild gerade erschienen ist?

- [ ] **B4 – Unterschiedliche Metrik**
  *Frage:* Der Sliding Estimator liefert AUC, BunDLe-Net liefert einen Loss. Welche gemeinsame Zahl kannst du für *beide* berechnen? Welche Ratequote hat diese Zahl bei 2 bzw. 5 Klassen?

- [ ] **B5 – Wie viel Zeit sieht das Modell?**
  *Wo:* `build_training_batch`, das Ziel ist `labels[epoch_number, start_time_point + window_length]`.
  *Frage:* Der Sliding Estimator sieht für Zeitpunkt t genau einen Zeitpunkt. Welche Zeitpunkte hat BunDLe-Net gesehen, wenn es t vorhersagt? Was heißt das für den frühesten Zeitpunkt, für den BunDLe-Net überhaupt eine Vorhersage machen kann? Und ist der Vergleich „100 ms Kontext gegen 1 Zeitpunkt“ fair?

- [ ] **B6 – Was soll der Vergleich eigentlich zeigen?**
  *Frage:* BunDLe-Net soll vor allem einen **latenten Raum** mit Markov-Eigenschaft lernen, der gut zum Verhalten passt. Es ist nicht in erster Linie als Klassifikator gebaut. Was ist also deine Behauptung in der Arbeit: „BunDLe-Net klassifiziert besser“ oder „der latente Raum von BunDLe-Net enthält die Kategorie-Information genauso gut wie die rohen Kanäle, nur in 3 Dimensionen“? Die Antwort bestimmt, wie du vergleichen musst.
  *Wo:* Lies nochmal im BunDLe-Net-Paper nach, wie die Autoren ihre Einbettung bewerten.

---

## 4. Die nächsten Schritte, in dieser Reihenfolge

Jeder Schritt hat ein Ziel und eine **Prüfung**, an der du selbst siehst, ob er gelungen ist.

### Schritt 1: Tabelle aus Abschnitt 3B ausfüllen
- **Ziel:** Du weißt genau, wo sich die beiden Notebooks unterscheiden.
- **Prüfung:** Du kannst zu jeder Zeile mit „Nein“ sagen, ob der Unterschied den Vergleich verfälscht.

### Schritt 2: Eine gemeinsame Grundlage schaffen
- **Ziel:** Beide Modelle bekommen exakt dieselben Daten.
- **Fragen:** Wie stellst du sicher, dass beide Notebooks denselben Probanden, dieselbe Vorverarbeitung und **dieselben Test-Epochs** verwenden? Lohnt es sich, Vorverarbeitung und Aufteilung in eine eigene Funktion oder Datei auszulagern und die Indizes zu speichern?
- **Prüfung:** Beide Notebooks geben dieselbe Anzahl Test-Epochs aus, und die Bildnummern im Test sind identisch.

### Schritt 3: Den Sliding Estimator sauber machen (A1–A3)
- **Ziel:** Kein Datenleck, keine gleichen Bilder in Training und Test, eine Rate-Untergrenze im Plot.
- **Prüfung:** Die Kurve vor t = 0 liegt immer noch bei 0,5. Überleg dir *vorher*, ob der Gipfel nach dem Beheben von A2 höher, gleich oder niedriger sein sollte, und warum.

### Schritt 4: Signifikanz prüfen (A4)
- **Ziel:** Ein Zeitbereich, für den du sagen kannst: „hier signifikant über Zufall“.
- **Prüfung:** Vor t = 0 sollte nichts signifikant sein. Wenn doch, stimmt etwas nicht.

### Schritt 5: BunDLe-Net als Kurve über die Zeit auswerten
- **Ziel:** Eine Genauigkeits- (oder AUC-)Kurve für BunDLe-Net auf den Test-Epochs, im selben Plot wie der Sliding Estimator.
- **Hinweis:** Jede Vorhersage gehört zu einem bestimmten Zeitpunkt, dem Ende des Fensters. Wie kannst du die Vorhersagen nach diesem Zeitpunkt gruppieren?
- **Vorher klären:** B3 und B4. Vergleichst du dieselbe Aufgabe mit derselben Metrik?
- **Prüfung:** Wo erwartest du die BunDLe-Net-Kurve vor t = 0? Denk an B3.

### Schritt 6: Den latenten Raum mit dem Sliding Estimator untersuchen
- **Idee:** Der Sliding Estimator ist nicht nur ein Konkurrent, sondern auch ein **Messwerkzeug**. Was passiert, wenn du ihn auf die 3 latenten Dimensionen von BunDLe-Net anwendest statt auf die rohen Kanäle?
- **Recherche:** „linear probing“
- **Fragen:** Welche Form müssen die latenten Daten haben, damit der Sliding Estimator sie annimmt? Welche Kurve erwartest du im Vergleich zu den rohen Kanälen?

### Schritt 7: Formulieren, was der Vergleich zeigt
- Zurück zu B6: Schreib in 2–3 Sätzen auf, was du mit dem Vergleich in der Arbeit behaupten willst und was nicht.

---

## 5. Die Technik in einem Absatz (für das Methodenkapitel)

Der Sliding Estimator trainiert **für jeden Zeitpunkt t ein eigenes, unabhängiges Modell**. Für dieses Modell sind die **Epochs die Stichproben** und die **Kanäle zum Zeitpunkt t die Merkmale**. Deshalb braucht er Epochs: Erst sie machen „Zeitpunkt 150 ms nach dem Bild“ über viele Durchgänge hinweg vergleichbar. In einer ungeschnittenen Aufnahme hätte „Zeitpunkt t“ keine gemeinsame Bedeutung.

*Selbsttest:* Kannst du erklären, warum `X.shape == (Epochs, Kanäle, Zeitpunkte)` genau das Format ist, das er erwartet, und über welche Achse er „gleitet“?

---

## 6. Quellen

**Sliding Estimator / zeitaufgelöstes Decoding**
- MNE-Tutorial „Decoding (MVPA)“, Abschnitt *Temporal decoding*: https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html
- API `mne.decoding.SlidingEstimator`: https://mne.tools/stable/generated/mne.decoding.SlidingEstimator.html
- King & Dehaene (2014): *Characterizing the dynamics of mental representations: the temporal generalization method.* Trends in Cognitive Sciences.
- Grootswagers, Wardle & Carlson (2017): *Decoding dynamic brain patterns from evoked responses: A tutorial on multivariate pattern analysis applied to time series neuroimaging data.* Journal of Cognitive Neuroscience. Sehr gut für das Methodenkapitel.

**Saubere Auswertung**
- scikit-learn: `GroupKFold`, `DummyClassifier`, `permutation_test_score` (User Guide „Cross-validation“)
- Maris & Oostenveld (2007): *Nonparametric statistical testing of EEG- and MEG-data.* Journal of Neuroscience Methods. Grundlage für Cluster-Permutationstests.
- MNE-Tutorials zur Statistik (Abschnitt „Statistical analysis of sensor data“)

**Inhaltlich**
- Rossion & Jacques (2008): *Does physical interstimulus variance account for early electrophysiological face sensitive responses in the human brain? Ten lessons on the N170.* NeuroImage. Hilft beim Einordnen deines Gipfels bei 150–170 ms und deines frühen Anstiegs bei 80 ms.
- Kumar et al. (2023): *BunDLe-Net: Neuronal Manifold Learning Meets Behaviour.* bioRxiv. Wie bewerten die Autoren ihre Einbettung?

---

## 7. So arbeiten wir

- Ich gebe dir **Aufgaben, Hinweise und Quellen**, du schreibst den Code.
- Hilfe gibt es in Stufen: sag **„Hinweis“**, **„mehr Hilfe“** oder **„Lösung“**.
- Zeig mir deinen Code, dann bekommst du Feedback.
- Drei Gewohnheiten: **erst auf Deutsch planen**, **`print(x.shape)` nach jedem Schritt**, **Fehlermeldungen von unten lesen**.
