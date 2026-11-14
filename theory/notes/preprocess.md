Die eigentliche EEG-Vorverarbeitung ist in diesem Repo nicht enthalten. Alle Analyseskripte laden schon vorverarbeitete Daten, die man separat von OSF herunterladen muss (`data/EEG/EEG_preprocessed/subXX/stepxx_timelock.mat`). Die Details zu Filtern, ICA und so weiter stehen nur im Methodenteil des Papers. Den konnte ich nicht lesen, weil bioRxiv die Anfrage blockiert hat.

## Was sich aus dem Code ablesen lässt

**Über die Rohdaten-Vorverarbeitung (nur indirekt):**
- Der Variablenname `data_rej_channel_interpolated_timelocked` zeigt: schlechte Kanäle wurden verworfen und interpoliert, danach wurde in Epochen geschnitten (FieldTrip).
- Laut Kommentar in `eeg_time_decoding.m` sind die Daten auf **240 Zeitpunkte** heruntergerechnet.
- `helpers/acticap-64ch-standard2.mat` deutet auf eine **64-Kanal-actiCAP** hin, also Brain Products.

**Vorverarbeitung auf Decoding-Seite** (ausgeschrieben in `category_decoding.m` und den Helpers):
1. **Trialzahlen angleichen:** Jede Kategorie wird ohne Zurücklegen auf die kleinste vorhandene Trialzahl heruntergesampelt.
2. **Multivariate Noise Normalization (MVNN):** Eine Kovarianzmatrix pro Zeitpunkt und Bedingung, gemittelt über beide, dann werden die Daten mit Σ^(-1/2) geweißt.
3. **Pseudo-Trials:** Die Trials werden in 8 Gruppen gemittelt.
4. **Lineare SVM** (libsvm, C=1) mit Leave-one-pseudo-trial-out, paarweise zwischen Kategorien. Das ergibt eine RDM pro Zeitpunkt.
5. Das Ganze wird über Permutationen wiederholt, im Repo 5, im Original 100, und gemittelt.

## Was das für dich bedeutet (Denkanstöße)

- **Fairer Vergleich:** Dein SWE-Baseline (① im Diagramm) nutzt bisher weder MVNN noch Pseudo-Trials. Damit sind deine Werte nicht direkt mit dem Paper vergleichbar. Überleg dir, ob du die beiden Schritte für ① übernimmst, und dann konsequenterweise auch für ② (SWE auf Y).
- **Achte bei MVNN genau darauf, auf welchen Daten die Kovarianz geschätzt wird.** Schau dir an, ob in `multivariate_noise_normalization` auch der Pseudo-Trial steckt, auf dem später getestet wird. Das ist dieselbe Leakage-Frage wie bei τ.
- **240 Zeitpunkte:** Rechne mit deinen eigenen Epochengrenzen und deiner Abtastrate nach, welche Kombination auf 240 kommt. Daran siehst du, wie weit deine 250 Hz vom Original abweichen (vgl. Frage 18 in deiner `questions.md`).

## Wo du die fehlenden Details findest
- Der **Methodenteil des Papers**: bioRxiv `10.1101/2024.07.30.605732` oder die veröffentlichte Version. Öffne ihn im Browser und such nach „EEG preprocessing“.
- Das **OSF-Projekt** (osf.io/xeukw): Prüf, ob dort zusammen mit den Daten ein Preprocessing-Skript liegt, zum Beispiel die Schritte vor `stepxx`.

Wenn du mir den Methodenteil hier hineinkopierst, gleiche ich ihn mit deiner [preprocess.py](notebooks/BunDLe-Net/preprocess.py) ab und markiere, wo du abweichst.