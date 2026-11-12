Ich habe mir deine Notizen, den LaTeX-Entwurf, `preprocess.py` und den Datensatz angeschaut. Vorweg etwas, das du morgen auf jeden Fall ansprechen solltest:

> **Deine Forschungsfrage im LaTeX-Entwurf lautet: „Kann BunDLe-Net altersbedingte Verzögerungen und Entdifferenzierung in höheren visuellen Repräsentationen erkennen?“** Der Datensatz enthält dafür 21 jüngere und 22 ältere Probanden (sub-1xx = jung, sub-2xx = alt). Bisher arbeitest du aber nur mit einzelnen jungen Probanden (101, 106) und vergleichst Gesichter mit Objekten. Der Altersvergleich kommt noch gar nicht vor. Laut deinem Zeitplan soll die Pipeline Ende November fertig sein, also solltest du jetzt klären, ob der Altersvergleich wirklich das Ziel bleibt.

---

## 1. Umfang und Ziel (am wichtigsten)

1. **Gilt die Forschungsfrage zum Altern noch, oder reicht ein Machbarkeitsnachweis?** Also: „BunDLe-Net funktioniert auf EEG und findet Kategorie-Information.“
2. **Was gilt als Ergebnis, wenn BunDLe-Net nichts findet oder schlechter ist als ein lineares Modell?** Die Arbeit ist als explorativ angelegt, frag also, ob ein Negativergebnis mit sauberer Begründung genügt.
3. **Wie viele Probanden?** Alle 43, eine Teilmenge (z. B. 5 jung + 5 alt) oder nur einer? Soll jeder Proband ein eigenes Modell bekommen, oder sollen alle gemeinsam trainiert werden?
4. **Welche Klassen?** 2 (Gesicht/Objekt), 4 Kategorien, und braucht es zusätzlich das Label 0 für „vor dem Bild“? Catch-Trials und Tastendrücke einfach weglassen?
5. **Formales:** Ungefährer Seitenumfang, Sprache (Deutsch oder Englisch), Verhältnis Theorie zu Experimenten, ist die Abgabe im Februar realistisch, und wann sind die nächsten Zwischentermine?

## 2. Methoden und Auswertung

6. **Was ist die Hauptaussage?** „BunDLe-Net klassifiziert gut“ oder „Der 3D-latente Raum enthält die Kategorie-Information“? Davon hängt ab, wie du vergleichen musst (siehe B6 in deinen Notizen).
7. **Wie misst man „Verzögerung“ und „Entdifferenzierung“ im latenten Raum?** Zum Beispiel über den Zeitpunkt, ab dem sich die Kategorie-Trajektorien trennen, oder über den Abstand der Kategorie-Cluster bei Jung und Alt. Im Originalpaper (Haupt, Garrett & Cichy) wurde das über Decoding-Latenzen und RSA gemacht. Frag, ob du dich daran orientieren sollst.
8. **Vergleichsmodelle:** Reicht der Sliding Estimator als lineare Referenz plus ein DummyClassifier als Untergrenze? Oder erwartet er zusätzlich PCA, CEBRA oder EEGNet?
9. **Epochen oder kontinuierliche Daten für BunDLe-Net?** Ist eine Fensterlänge von 25 Zeitpunkten (100 ms) sinnvoll? Soll die Variante „BunDLe-Net continuous“ aus deinen Notizen noch angeschaut werden?
10. **Statistik:** Werden Permutationstests oder Cluster-Permutationstests erwartet? Ist GroupKFold nach Bildnummer nötig, damit dasselbe Bild nicht in Training und Test landet?
11. **Rechenressourcen:** Gibt es einen Uni-Cluster oder eine GPU? (Das Speicherproblem mit den 60 GiB steht noch im Entwurf.)

## 3. Vorverarbeitung

12. **Soll ich die Vorverarbeitung aus dem Originalpaper übernehmen** (dann ist dein Vergleich mit deren Ergebnissen fairer), oder darf ich eine eigene bauen?
13. **Filter:** Bisher gibt es nur einen Hochpass bei 0,5 Hz. Braucht es noch einen Tiefpass (z. B. 40 Hz) oder einen Notch-Filter bei 50 Hz? Für die ICA ist ein eigener Hochpass bei 1 Hz üblich.
14. **ICA:** Sollen Komponenten automatisch (ICLabel, EOG-Kanäle) oder von Hand ausgewählt werden?
15. **Referenz:** Durchschnittsreferenz, ja oder nein?
16. **Epochen verwerfen:** Autoreject oder eine feste Schwelle? Wie viele verworfene Epochen sind noch in Ordnung? Bei älteren Probanden gibt es oft mehr Artefakte. Wie bleibt der Gruppenvergleich trotzdem fair?
17. **Baseline-Korrektur und Epochenfenster:** Ist −0,3 bis 0,7 s gut?
18. **Abtastrate:** Reichen 250 Hz, oder für BunDLe-Net weiter herunterrechnen (z. B. 100 Hz), damit aufeinanderfolgende Fenster sich mehr unterscheiden?
19. **Pseudo-Trials:** Mehrere Durchgänge derselben Kategorie zu einem mitteln, um das Signal-Rausch-Verhältnis zu verbessern? In der Arbeitsgruppe von Cichy ist das üblich.
20. **PCA vor BunDLe-Net** (dein TODO): Lohnt sich das?
21. **Aufnahmepausen** (sub-101, sub-103) als `BAD_` annotieren: Ist das so okay?

---

## Was du mitbringen solltest

- Die **Sliding-Estimator-Kurve** (Gipfel um 150–170 ms, AUC ≈ 0,93). Das ist dein bisher bestes Ergebnis.
- Den **Verlauf des BunDLe-Net-Loss**, der beim Behaviour-Loss auf Zufallsniveau liegt. Frag direkt, ob er Ideen hat, woran das liegt.
- Eine kurze Liste der **Entscheidungen, die du schon getroffen hast** (Filter, Fenster, Labels), damit er sie absegnen oder korrigieren kann.

**Prüf vorher noch selbst eine Sache:** Schau dir in `preprocess.py` die Funktion `ica()` an. Welche Komponenten entfernt `ica.apply(self.raw)` eigentlich, wenn du vorher nichts festgelegt hast? Wenn die Antwort lautet, dass das Signal unverändert bleibt, solltest du das wissen, bevor du ihm sagst, dass du ICA verwendest.

Wenn du die wichtigsten 5 oder 6 Fragen auf einem Blatt zum Mitnehmen haben willst, kann ich sie dir kürzen.