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