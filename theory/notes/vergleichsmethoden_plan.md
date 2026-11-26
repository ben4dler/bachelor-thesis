# Plan für die Vergleichsmethoden: BunDLe-Net gegen klassisches Decoding

Stand: 26.11.2026. Grundlage: `sliding_window_estimator.ipynb`, `swe_on_tau.ipynb`, `main.py`, `preprocess.py`, `bundle_loss.py` und die übrigen Notizen in diesem Ordner.

**Verbindliche Regel für alle Experimente:** Eine Baseline unterscheidet sich von BunDLe-Net in **genau einem** Faktor. Nur dann lässt sich ein Unterschied im Ergebnis einer Ursache zuordnen.

---

## 0. Warum überhaupt Vergleichsmethoden?

Das Ziel der Arbeit ist BunDLe-Net und die Frage, ob es altersbedingte Verzögerung und Entdifferenzierung erkennt. Die Vergleichsmethoden sind dafür kein Zusatzthema. Sie sind nötig, damit sich ein Befund von BunDLe-Net überhaupt deuten lässt.

1. **Ein Befund braucht einen Maßstab.** Angenommen, τ zeigt bei den Älteren eine Verzögerung von 40 ms. Ist das viel oder wenig, ist es echt oder ein Artefakt? Haupt et al. (2025) finden mit einer linearen SVM **21 ms**. Erst wenn ein klassisches Decoding auf *deiner* Vorverarbeitung diesen Effekt reproduziert, weißt du, dass die Pipeline stimmt und dass im Datensatz ein Effekt steckt, den BunDLe-Net finden *kann*.
2. **Das Fenster von τ verschiebt Latenzen.** τ(t) sieht die Daten von t − 140 ms bis t. Onset und Peak der Decoding-Kurve werden dadurch verschoben und verschmiert. Bei Altersunterschieden von wenigen zehn Millisekunden kann genau das den Effekt erzeugen oder verdecken. Nur ein klassischer Decoder auf **denselben Fenstern** zeigt, welcher Teil vom Fenster kommt und welcher von BunDLe-Net.
3. **Ein Gruppenunterschied in τ kann eine triviale Ursache haben.** Ältere Probanden haben oft verrauschteres EEG. Dann lernt τ bei ihnen schlechter, und die Kategorien trennen sich schwächer. Das sähe aus wie Entdifferenzierung, wäre aber nur Rauschen. Zeigt ein einfacher Decoder auf denselben Daten denselben Unterschied, liegt er in den Daten. Zeigt nur τ ihn, musst du erklären, warum.
4. **Was ist der Beitrag von BunDLe-Net?** Die Frage eines Gutachters wird sein: „Was kann BunDLe-Net, was ein linearer Klassifikator nicht kann?“ Mögliche gute Antworten: gleiche Information in nur 3 Dimensionen, interpretierbare Trajektorien, Dynamik über das Markov-Modell. Jede davon lässt sich nur **im Vergleich** belegen.
5. **Ein Negativergebnis wird verteidigbar.** Findet BunDLe-Net keinen Altersunterschied, zeigen die Baselines, ob der Effekt in deinen Daten fehlt (Pipeline-Problem) oder ob ihn nur BunDLe-Net nicht findet. Beides ist ein gültiges Ergebnis, aber nur, wenn man es unterscheiden kann.

**Folge daraus:** Du brauchst **nicht alle** Methoden aus Abschnitt 2. Pflicht ist der kleine Kern, der die Punkte 1 bis 4 abdeckt (Abschnitt 3, Stufe 1 und 2). Der Rest ist optional und nur sinnvoll, wenn Zeit bleibt.

---

## 1. Bestandsaufnahme: was schon läuft

| # | Methode | Wo | Input | Ergebnis / Anmerkung |
|---|---|---|---|---|
| 1 | Spatio-temporale LogReg (Vectorizer → LogReg, liblinear) | `sliding_window_estimator.ipynb` | ganze Epoche 64 × 120, sub-101 + 102, 5-fach-CV | 78,4 % bei **2** Kategorien |
| 2 | CSP + LogReg | gleiches Notebook | ganze Epoche | 48,9 %, also Zufall. CSP sucht Varianz- bzw. Oszillationsunterschiede, ERPs sind phasengebunden. **Weglassen oder in der Arbeit begründen.** |
| 3 | Sliding Estimator (LogReg pro Zeitpunkt), mit und ohne `LinearModel` | gleiches Notebook | 64 Kanäle pro Zeitpunkt, 3-fach-CV | Zeitverlauf |
| 4 | SWE auf τ (StandardScaler + LogReg) | `swe_on_tau.ipynb` | 3 Dimensionen × 105 Zeitpunkte, Hold-out | Kurve gegen #5 |
| 5 | SWE auf rohen Kanälen | gleiches Notebook | 64 Kanäle pro Zeitpunkt, Hold-out | Referenzkurve |
| 6 | Spatio-temporale LogReg auf Rohfenstern | gleiches Notebook | 64 · 15 | 50,2 % bei 4 Klassen. ⚠️ Der Print sagt „auf tau“, es sind aber **Rohdaten**. |
| 7 | PCA | `main.py`, `bundle_trainer.py` | latenter Raum bzw. Init von τ | nur zur Analyse, **kein** Vergleichsdecoder |
| 8 | Zufallsniveau analytisch (ln K, 1/K) | beide Notebooks | – | kein empirischer Dummy, kein Permutationstest |

`GeneralizingEstimator` ist importiert, wird aber nicht verwendet.

### ⚠️ Nebenbefund: `GAMMA` (`preprocess.py:15`, inzwischen wieder 0.7)
`total_loss = gamma · DCC + (1 − gamma) · behaviour` (`bundle_loss.py:21`). Mit γ = 0 ist der Markov-Term aus, und τ ist ein rein überwachter MLP-Encoder, also nicht BunDLe-Net im eigentlichen Sinn.
- [ ] Beide Läufe machen und vergleichen: γ = 0 (Obergrenze, reine Klassifikation) und das Ziel-γ (BunDLe-Net).

---

## 2. Was fehlt, sortiert nach dem Faktor, den es isoliert

Zwischen τ und dem rohen SWE unterscheiden sich derzeit vier Faktoren gleichzeitig: **Fenster, Supervision, Nichtlinearität, Dimension.**

### B1. Untergrenzen und Sanity Checks
- [ ] `DummyClassifier` (`most_frequent`, `stratified`) als empirische Untergrenze
- [ ] Permutationstest mit vertauschten Labels für die Baselines (`permutation_test_score`)
- [ ] **τ mit permutierten Labels trainieren.** Liegt der τ-SWE danach über dem Zufall, gibt es ein Leak.

### B2. Gleiches Fenster, ohne BunDLe-Net (isoliert: Fenster)
- [ ] SWE auf Rohfenstern (64 × 15) pro Zeitpunkt, mit derselben Pipeline wie der τ-SWE. Die Daten liegen in `train_all_windows_all_epochs` schon bereit.
- [ ] Kontrolle mit kürzeren Fenstern: Wie verschieben sich Onset und Peak?

### B3. Gleiche Dimension (3D), ohne BunDLe-Net (isoliert: Supervision und Nichtlinearität)
- [ ] PCA(3) auf den Fenstern → SWE (unüberwacht, linear)
- [ ] LDA als Projektion → SWE (überwacht, linear). Bei 4 Klassen liefert LDA höchstens K − 1 = **3** Dimensionen, genau wie τ.
- Vergleich PCA gegen LDA zeigt den Effekt der Supervision, LDA gegen τ den Effekt der Nichtlinearität.

### B4. Standardklassifikatoren aus der EEG-Literatur (optional)
- [ ] LDA mit Shrinkage (`solver="lsqr"`, `shrinkage="auto"`)
- [ ] Lineare SVM, der Klassifikator aus dem Paper
- [ ] xDAWN + Riemann (Tangent Space, `pyriemann`) als ERP-Standard, Ersatz für CSP
- [ ] Ein nichtlinearer Klassifikator (RF oder RBF-SVM): Erklärt „nichtlinear“ allein schon den Vorsprung von τ?

### B5. Paper-Protokoll von Haupt et al. (Pflicht für die Altersfrage)
- [ ] Paarweises Decoding (6 Kategorie-Paare, 50 % Zufall)
- [ ] Pseudo-Trials (8 Bins) und MVNN, Kovarianz nur auf den Trainingsdaten
- [ ] **Jung gegen Alt:** Peak-Latenz pro Gruppe mit Bootstrap-Konfidenzintervall. Wird der Delay von ca. 21 ms reproduziert?
- [ ] Temporal Generalization (`GeneralizingEstimator`) auf X und auf τ (optional)

### B6. Einheitliches Evaluationsprotokoll (Pflicht, gilt für alles)

| | jetzt | Regel ab sofort |
|---|---|---|
| Klassen | 2 bzw. 4 | **4 Kategorien** (oder festlegen und überall gleich) |
| Validierung | 5-CV, 3-CV, Hold-out | **ein** Schema; τ und Baselines auf **identischem** Split |
| Solver / Scaling | uneinheitlich | identische Pipeline für alle linearen Decoder |
| Probanden | gepoolt | pro Proband, dann über Probanden mitteln |
| Bilder | gleiche Bilder in Train und Test | `GroupKFold` nach Bildnummer prüfen |
| Seeds | 1 | mehrere, Streuung berichten |
| Metrik | Accuracy | Balanced Accuracy oder AUC |
| Trialzahl | ungleich | pro Klasse und pro Gruppe angleichen (wichtig für jung gegen alt) |
| Statistik | keine | Cluster-Permutationstest, Bootstrap für Latenzen |

---

## 3. Reihenfolge, an die ich mich halte

**Stufe 1: Fundament (Pflicht)**
1. B6: Evaluationsprotokoll vereinheitlichen
2. B1: Dummy und Permutationstest, τ mit permutierten Labels
3. γ = 0 gegen Ziel-γ

**Stufe 2: Kernvergleich für die Altersfrage (Pflicht)**

4. B5: Rohes SWE (Paper-nah) für jung und alt → Delay reproduzieren
5. B2: SWE auf Rohfenstern für jung und alt → Effekt des Fensters
6. τ-SWE und latente Trajektorien für jung und alt → Beitrag von BunDLe-Net
7. B3: PCA(3) / LDA(3) → Ist die 3D-Kompression BunDLe-spezifisch?

**Stufe 3: nur wenn Zeit bleibt (optional)**

8. B4: weitere Klassifikatoren
9. Temporal Generalization

---

## 4. Offene Fragen (mit dem Betreuer klären)
- [ ] Wird τ **gemeinsam** für jung und alt trainiert (ein gemeinsamer latenter Raum) oder pro Proband bzw. Gruppe? Bei getrennten Modellen sind die latenten Räume nicht direkt vergleichbar.
- [ ] Wie werden „Verzögerung“ und „Entdifferenzierung“ im latenten Raum gemessen? Zum Beispiel Zeitpunkt der Trennung der Trajektorien oder Abstand der Kategorie-Cluster.
- [ ] Hauptaussage festlegen: „τ ist besser decodierbar“ (dann ist B2 zentral) oder „τ bewahrt die Information in 3D“ (dann ist B3 zentral)?
- [ ] Reicht der Pflichtkern aus Stufe 1 und 2, oder werden B4 / Temporal Generalization erwartet?

---

## 5. Quellen
- Haupt, Garrett & Cichy (2025), *Healthy aging delays and dedifferentiates high-level visual representations*, Current Biology. Siehe `paper_methoden.md`.
- Grootswagers, Wardle & Carlson (2017), *Decoding Dynamic Brain Patterns from Evoked Responses*, J. Cogn. Neurosci.: Klassifikatoren, Shrinkage, Fenstergrößen.
- Varoquaux et al. (2017), *Assessing and tuning brain decoders*, NeuroImage: CV und Leakage.
- Guggenmos, Sterzer & Cichy (2018), *Multivariate pattern analysis for MEG: A comparison of dissimilarity measures*, NeuroImage: MVNN und Pseudo-Trials.
- King & Dehaene (2014), *Characterizing the dynamics of mental representations: the temporal generalization method*, Trends Cogn. Sci.
- Rivet et al. (2009), xDAWN; Barachant et al. (2013), Riemannsche Klassifikation; `pyriemann`-Doku (ERP-Beispiel).
- MNE-Tutorial „Decoding (MVPA)“; sklearn-Doku zu `DummyClassifier`, `permutation_test_score`, `LinearDiscriminantAnalysis`.

> Die Quellen stammen aus Claudes Wissen und wurden nicht online geprüft. Im Zweifel über Google Scholar mit Autor und Titel suchen.
