Dann schlage ich vor, dass ich ab jetzt eher dein Tutor bin und nicht der, der den Code schreibt. Das habe ich mir für dieses Projekt auch so gemerkt.

## So könnten wir arbeiten

**1. Ich gebe dir Aufgaben, du schreibst den Code.**
Jede Aufgabe hat ein klares Ziel und ein Ergebnis, an dem du selbst prüfen kannst, ob es stimmt, z. B. „am Ende sollte das Array die Form (1638, 64, 250) haben“.

**2. Bei Problemen helfe ich in Stufen.**
- Zuerst eine **Frage oder ein Hinweis**, in welche Richtung du schauen solltest
- Wenn das nicht reicht, ein **konkreter Tipp**, z. B. welche Funktion du brauchst
- Die **Lösung** nur, wenn du ausdrücklich danach fragst

Sag einfach „Hinweis“, „mehr Hilfe“ oder „Lösung“.

**3. Du zeigst mir deinen Code, ich gebe Feedback.**
Was funktioniert, was fehleranfällig ist und warum. So lernst du mehr als beim Abschreiben.

**4. Konzepte erkläre ich gerne ausführlich**, wie vorhin bei den Epochs. Verstehen ist das, was du in der Arbeit und in der Verteidigung brauchst.

## Drei Gewohnheiten, die dir viel Zeit sparen

- **Erst auf Deutsch planen, dann programmieren.** Schreib als Kommentare auf, was passieren soll (`# 1. Labels holen`, `# 2. ...`), und übersetze erst dann jede Zeile in Python.
- **Ständig die Form ausgeben.** `print(variable.shape)` nach jedem Schritt. Die meisten deiner bisherigen Fehler (leerer Split, falsche Achse, Liste statt Array) hättest du damit sofort gesehen.
- **Fehlermeldungen von unten lesen.** Die letzte Zeile sagt, *was* schiefging, die Zeile darüber mit deinem Code sagt, *wo*.

## Deine erste Aufgabe: die einfache Vergleichsanalyse

**Ziel:** Herausfinden, ab wie vielen Millisekunden nach dem Bild sich Gesichter von Objekten im EEG unterscheiden lassen.

**Was du schon hast:** `epochs_clean` und die Funktion `create_labels_for_epochs`.

**Teilaufgaben:**
1. Wähle aus `epochs_clean` nur die Gesichts- und Objekt-Epochs aus. *Erwartung: 1024 Epochs.*
2. Hol dir die Daten als Array. *Welche Form erwartest du?*
3. Erzeuge ein Label pro **Epoch** (nicht pro Zeitpunkt!): 0 für Gesicht, 1 für Objekt. *Erwartung: Form (1024,).*
4. Schau dir in der MNE-Dokumentation das Tutorial **„Decoding (MVPA)“** an, besonders den Abschnitt zu `SlidingEstimator`. Versuch, es auf deine Daten zu übertragen.
5. Plotte die Genauigkeit über die Zeit.

**Fragen zum Nachdenken, bevor du anfängst:**
- Welche Genauigkeit hätte ein Modell, das nur rät?
- Was erwartest du für die Zeit *vor* dem Bild (t < 0)?

Die Antworten darauf sind gleichzeitig deine erste Prüfung, ob die Ergebnisse plausibel sind.

Fang an, wann du willst, und meld dich, wenn du hängst oder Feedback zu deinem Code möchtest.



# Was dir der Sliding Estimator gebracht hat

Du siehst jetzt, **wann** im EEG sich Gesichter von Objekten unterscheiden, und nicht mehr nur, **ob** sie es tun. Die Antwort für Proband 101: ab etwa 80–100 ms nach dem Bild, am stärksten bei etwa 150–170 ms. Das passt sehr gut zur bekannten **N170**, der typischen Gesichtsreaktion im EEG.

## Die drei Ergebnisse im Vergleich

| Methode | Ergebnis | Was es bedeutet |
|---|---|---|
| Spatio-temporal (alle Zeitpunkte zusammen) | 85,9 % | Gesicht und Objekt lassen sich trennen, aber du weißt nicht, wann. |
| CSP | 50,2 % | Nur Raten (siehe unten) |
| **Sliding Estimator** | Kurve über die Zeit | Ein eigenes Modell für jeden der 250 Zeitpunkte, also eine Genauigkeit pro Millisekunde |

## Die Decoding-Kurve („Sensor space decoding“)

- **Vor dem Bild (t < 0):** Die Kurve schwankt um 0,5. Das ist die Ratequote, und genau das solltest du erwarten, denn vor dem Bild weiß das Gehirn ja noch nicht, was kommt. Damit hast du die Plausibilitätsprüfung aus continue.md bestanden. Die Schwankung von etwa ±0,05 zeigt dir außerdem, wie groß das reine Rauschen ist.
- **Ab ca. 80 ms:** Die Kurve steigt auf ca. 0,65. Das ist früh. In dieser Zeit verarbeitet das Gehirn eher einfache Bildeigenschaften wie Helligkeit, Kontrast oder Form, noch nicht „Gesicht“ als Kategorie. Wahrscheinlich unterscheiden sich deine Gesichts- und Objektbilder also schon in solchen Eigenschaften.
- **Gipfel bei ca. 150–170 ms, AUC ≈ 0,93:** Hier lassen sich die beiden Kategorien fast perfekt trennen. Das ist das Zeitfenster der N170.
- **Danach bleibt die Kurve bei 0,6–0,75** bis zum Ende. Die Information verschwindet also nicht, sie wird nur schwächer und verteilter.

Zur y-Achse: Das ist **AUC**, nicht Genauigkeit. Die Ratequote liegt trotzdem bei 0,5, und 1,0 wäre eine perfekte Trennung.

## Die Patterns-Grafik (Kopfkarten)

Sie zeigt, **welche Elektroden** das Modell zu jedem Zeitpunkt nutzt.
- **0,1 s und 0,2 s:** Zwei rote Flecken hinten links und rechts am Kopf, über dem Bereich seitlich hinter den Ohren (etwa P7/P8). Genau dort misst man die N170. Das Modell lernt also etwas Neurophysiologisches und keinen Zufallseffekt.
- Die **Kurven darunter** haben ihren Gipfel ebenfalls bei ca. 150 ms, passend zur Decoding-Kurve.
- Ignoriere die Einheit (1e6 µV). Weil du die Daten vorher skaliert hast, sind die Werte willkürlich. Nur die Form und die Verteilung auf dem Kopf zählen.

## Warum CSP bei 50 % liegt

CSP sucht Unterschiede in der **Varianz**, also in der Stärke von Schwingungen. Das funktioniert gut, wenn sich die Gehirnschwingungen verändern, zum Beispiel wenn sich jemand eine Bewegung nur vorstellt. Bei dir steckt der Unterschied aber im **Verlauf der Spannung über die Zeit** (der Ausschlag der N170), und den sieht CSP kaum. Das Ergebnis ist also kein Fehler in deinem Code, die Methode passt nur nicht zu deiner Frage. Die CSP-Grafiken (Scree plot, Pattern0–2) kannst du deshalb ignorieren.

## Zwei Punkte zu deinem Code

1. **Doppelte Skalierung mit Datenleck:** In Zelle 6 machst du `scaler.fit_transform` auf **allen** Gesichts- und Objekt-Epochs, also bevor die Kreuzvalidierung die Daten aufteilt. Dadurch sieht der Scaler auch die späteren Testdaten. Die Pipeline enthält außerdem schon einen `StandardScaler`. Frage an dich: Was passiert, wenn du die äußere Skalierung einfach weglässt?
2. **Dieselben Bilder in Training und Test:** Jedes der 32 Bilder kommt 32-mal vor. Das Modell könnte deshalb einzelne Bilder wiedererkennen statt die Kategorie. Das passt zu dem frühen Anstieg bei 80 ms. Für die Arbeit solltest du dir das merken.

## Nächster Schritt (falls du willst)

Bisher sagst du nur mit dem Auge, dass die Kurve „über dem Zufall“ liegt. Für die Bachelorarbeit brauchst du eine Zahl dafür. Deine Aufgabe: Finde heraus, **wie du prüfen kannst, ab welchem Zeitpunkt die AUC signifikant über 0,5 liegt.** Ein Hinweis: Denk daran, wie die Kurve aussähe, wenn du die Labels zufällig vertauschst.

Sag „Hinweis“, wenn du einen Startpunkt brauchst.