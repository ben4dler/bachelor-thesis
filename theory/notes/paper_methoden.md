# Methoden aus Haupt, Garrett & Cichy (2025)

*„Healthy aging delays and dedifferentiates high-level visual representations“, Current Biology 35, 2112–2127.*
Quelle: STAR Methods, S. e1–e6 im PDF (`1-s2.0-S0960982225003781-main.pdf`), ergänzt um Hinweise aus dem Code (github.com/marleenhaupt/ORHA).

Fokus: alles, was für EEG und Decoding relevant ist. fMRI und Verhalten nur kurz.

---

## 1. Probanden

- 21 jüngere (19–33 Jahre) und 22 ältere (60–73 Jahre) Probanden, insgesamt 43.
- Normales oder korrigiertes Sehen, Kontrastsensitivität geprüft (FrACT).
- Ältere zusätzlich mit MMSE gescreent (Cut-off < 27/30). Niemand wurde ausgeschlossen.
- EEG: keine Ausschlüsse erwähnt. (fMRI: 1 älterer Proband wegen niedriger Task-Accuracy ausgeschlossen.)

## 2. Stimuli

- 64 natürliche Bilder, je 16 pro Kategorie: **Gesichter, Tiere, Orte, Objekte**.
- Quellen: FairFace (Gesichter), Places365 (Orte), MS COCO (Tiere und Objekte).
- Supra-Kategorien: belebt (Gesichter + Tiere) und unbelebt (Orte + Objekte).

## 3. EEG-Experiment

- 1 Sitzung, 8 Blöcke.
- Bild in der Mitte eines grauen Bildschirms, weißes Fixationskreuz.
- **Reguläre Trials:** 500 ms Bild, ISI zufällig 400–500 ms. Instruktion: fixieren, nicht blinzeln.
- **Catch-Trials:** alle 4–6 Trials ein „Schlumpf“-Bild → Tastendruck **und blinzeln**. 500 ms Bild, ISI 1400–1500 ms (damit Bewegung nicht in den nächsten Trial hineinwirkt).
- **2048 reguläre Trials** = 64 Bilder × 32 Wiederholungen.
- ⚠️ Im Paper steht „2048 regular trials and 54 catch trials, amounting to a total of 2560 trials“. Das geht nicht auf (2048 + 54 ≠ 2560). Bei 2560 wären es 512 Catch-Trials, was zu „alle 4–6 Trials“ passt. In deinen Events nachzählen!

## 4. EEG-Aufnahme

| | |
|---|---|
| Kappe | 64-Kanal actiCAP snap, 10-10-System, zusätzliche Ground- und Referenzelektrode auf der Kopfhaut |
| Verstärker | BrainVision actiCHamp |
| Software | BrainVision Recorder |
| Abtastrate | 1000 Hz |
| Online-Filter | 0,03–100 Hz |

Rohdaten: OpenNeuro **ds005363**.

## 5. EEG-Vorverarbeitung

Laut Paper (FieldTrip, MATLAB 2021a), vollständig:

1. Epochen von **−200 bis 1000 ms** relativ zum Stimulus-Onset
2. **Notch-Filter bei 50 Hz**
3. **Baseline-Korrektur:** Mittelwert von −200 bis 0 ms von der ganzen Epoche abziehen

**Nicht** im Paper erwähnt: Hochpass, Tiefpass (über den Online-Filter hinaus), ICA, Re-Referenzierung, Trial-Rejection.

Zusätzlich aus dem Code (nicht im Paper):
- Der Variablenname `data_rej_channel_interpolated_timelocked` zeigt, dass **schlechte Kanäle verworfen und interpoliert** wurden. Nach welchem Kriterium, steht nirgends.
- Die Daten im Repo sind auf **240 Zeitpunkte** heruntergerechnet (1,2 s → **200 Hz**, also 5 ms Auflösung). Laut README können die Ergebnisse dadurch leicht vom Paper abweichen.

## 6. Time-resolved Decoding (das Gegenstück zu deinem SWE)

- **Klassifikator:** binäre C-SVC, linearer Kernel (libsvm). Pro Proband einzeln.
- **Features:** pro Zeitpunkt die 64 Kanalwerte als Mustervektor.
- **Ebenen:** 64 Bilder, 4 Kategorien oder 2 Supra-Kategorien.
- **Paarweise:** Jedes Paar wird einzeln klassifiziert → Zufallsniveau immer **50 %**.
  - Bilder: 2016 Paare, Kategorien: 6 Paare, Supra-Kategorien: 1 Paar.
- **Pseudo-Trials:** Trials zufällig auf **8 Bins** verteilt und gemittelt.
- **Cross-Validation:** 8-fach Leave-one-pseudo-trial-out (7 trainieren, 1 testen).
- **100 Wiederholungen** mit neuer zufälliger Bin-Zuweisung (im Repo nur 5).
- Gemittelt über Paare, Folds und Wiederholungen → eine Accuracy-Zeitkurve pro Proband.
- **Kontrolle:** Kategorie-Decoding mit Subsampling auf 4 Trials pro Pseudo-Trial, damit das SNR zum Bild-Decoding passt.
- **Kontrolle:** Bild-Decoding getrennt nach niedriger und hoher Bildkomplexität (Layer 3 von VGG-16).

Zusätzlich aus dem Code (nicht im Paper):
- **Trialzahl angleichen:** pro Klasse zufällig auf die kleinste Trialzahl heruntersampeln.
- **Multivariate Noise Normalization (MVNN)** vor den Pseudo-Trials: Kovarianz pro Zeitpunkt und Bedingung schätzen, mitteln, Daten mit Σ^(−1/2) weißen. Die Kovarianz wird auf **allen** Trials geschätzt, also auch auf denen, die später im Test-Pseudo-Trial landen.

## 7. Weitere EEG-Analysen

- **Searchlight im Sensorraum:** dasselbe Decoding pro Kanal, 5-ms-Auflösung.
- **Temporal Generalization:** bei t trainieren, auf allen t' testen (5-ms-Auflösung) → 2D-Matrix.
- **RSA:** Bild-Decoding-Accuracies als RDM (64 × 64) pro Zeitpunkt. RDMs der Älteren (pro Person) mit dem gemittelten RDM der Jüngeren korrelieren (Spearman), für alle Zeitpaare → Werte oberhalb der Diagonale zeigen Verzögerung.
- **Altersgruppen-Decoding:** SVM unterscheidet die RDMs von Jung und Alt; Leave-one-subject-out, 43 Folds, 1000 Permutationen. Danach nochmal mit den Älteren um **21 ms** verschoben (der gefundene Peak-Delay).
- **EEG-fMRI-Fusion:** EEG-RDMs pro Zeitpunkt mit fMRI-RDMs pro ROI korrelieren (V1–V3, FFA, PPA, LOC).
- **EEG–Verhalten-RSA:** EEG-RDMs mit Ähnlichkeits-RDMs aus der Multiple-Arrangement-Aufgabe korrelieren.

## 8. Statistik

- Alles nicht-parametrisch.
- **Einstichproben-Tests:** Sign-Permutation (Daten pro Proband zufällig mit ±1 multipliziert), Nullhypothese: Median = Zufallsniveau (50 % bzw. Korrelation 0).
- **Gruppenvergleich:** Gruppenlabels permutieren.
- **10.000 Permutationen**, Cluster-Definition bei q < .05, Cluster-Größen-Schwelle p < .05 (Maximum-Cluster-Verteilung).
- **FDR** (5 %) statt Cluster-Korrektur bei: Altersgruppen-Decoding, Pattern Similarity, fMRI–Verhalten-RSA.
- **Bootstrap** (10.000×, mit Zurücklegen) für Peak-Latenzen und Latenzunterschiede zwischen den Gruppen, mit 95 %-Konfidenzintervall.
- Wilcoxon-Rangsummentest für Catch-Trial-Accuracy zwischen den Gruppen.

---

## Was das für dich heißt (zum Abgleich mit deiner Pipeline)

- [ ] **Epochen und Baseline** auf −0,2 bis 1,0 s bzw. −0,2 bis 0 s umstellen.
- [ ] **Notch 50 Hz** ergänzen. Hochpass und ICA weglassen (vom Betreuer abgesegnet).
- [ ] **Abtastrate:** 200 Hz wie im Repo oder 1000 Hz wie im Paper? Für BunDLe-Net-Fenster relevant.
- [ ] **Kanalinterpolation:** Kriterium selbst festlegen und begründen (im Paper nicht beschrieben).
- [ ] **Catch-Trials** rauswerfen. Die Probanden sollten dabei blinzeln, die Trials sind also voller Artefakte.
- [ ] **SWE-Baseline vergleichbar machen:** paarweise Klassifikation (50 % Zufall), Pseudo-Trials (8), MVNN, lineare SVM. Deine bisherige AUC (Gesicht vs. Objekt) ist **eines** der 6 Kategorie-Paare, also nicht direkt mit der gemittelten Kurve aus dem Paper vergleichbar.
- [ ] **Offen:** MVNN auch auf X anwenden, bevor es in τ geht? (siehe Diagramm in `questions.md`)
- [ ] **Offen:** Kovarianz für MVNN nur auf Trainingsdaten schätzen (sauberer als im Original-Code)?
- [ ] Für eine Latenzaussage (Jung gegen Alt): **Bootstrap der Peak-Latenz** wie im Paper. Zum Vergleich: gefundener Delay = 21 ms.
