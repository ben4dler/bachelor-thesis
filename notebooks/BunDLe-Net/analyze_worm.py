"""Beschreibende Analyse der Wurmdaten, die functions.py verwendet.

Aufruf aus dem Projektverzeichnis mit der vorhandenen Python-Umgebung:
    .venv/bin/python notebooks/BunDLe-Net/analyze_worm.py
    .venv/bin/python notebooks/BunDLe-Net/analyze_worm.py --all
    .venv/bin/python notebooks/BunDLe-Net/analyze_worm.py --worm 2 --duration 120

Voraussetzungen: numpy, matplotlib, mat73.
Ergebnisse: results/worm_analysis/worm_<Nummer>/

Die Kennwerte beziehen sich auf die ganze Aufnahme; --duration begrenzt nur
die Zeitachse der Aktivitätsgrafik. Es wird kein Modell trainiert. Die Daten
werden wie im ursprünglichen Database-Loader geladen, ohne functions.py zu
importieren (und damit ohne TensorFlow oder MNE zu benötigen).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import mat73
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data" / "raw" / "NoStim_Data.mat"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "worm_analysis"


def load_worm(data: dict, worm: int) -> tuple:
    """Erstellt dieselben X- und B-Arrays wie Database und main.py.

    deltaFOverF_bc liegt in der Datei bereits als (Zeitpunkte, Neuronen) vor.
    Database transponiert es; main.py transponiert es wieder zurück.
    """
    X = np.asarray(data["deltaFOverF_bc"][worm], dtype=float)
    states = data["States"][worm]
    state_names = list(states)
    masks = np.stack([np.asarray(states[name]).reshape(-1) for name in state_names])

    if X.ndim != 2 or masks.shape[1] != X.shape[0]:
        raise ValueError(f"Wurm {worm}: X und States haben unpassende Formen.")
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError(f"Wurm {worm}: States enthält andere Werte als 0 und 1.")
    if not np.all(masks.sum(axis=0) == 1):
        raise ValueError(
            f"Wurm {worm}: Nicht jeder Zeitpunkt hat genau eine Klasse. "
            "Die Labelberechnung aus functions.py wäre hier mehrdeutig."
        )

    # Genau die Zuordnung aus functions.py: Position des State-Namens = Label.
    B = np.sum([label * masks[label] for label in range(len(state_names))], axis=0).astype(int)
    fps = float(data["fps"][worm])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Wurm {worm}: Ungültige Abtastrate {fps}.")

    # In Datensatz 3 steckt der letzte Neuronenname in einer zusätzlichen Liste.
    neuron_names = []
    for name in data["NeuronNames"][worm]:
        while isinstance(name, (list, tuple, np.ndarray)):
            name = name[0]
        neuron_names.append(str(name))
    if len(neuron_names) != X.shape[1]:
        raise ValueError(f"Wurm {worm}: Zahl der Neuronennamen passt nicht zu X.")
    return X, B, fps, neuron_names, state_names


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def neuron_statistics(X: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for index, name in enumerate(names):
        values = X[:, index]
        finite = values[np.isfinite(values)]
        rows.append({
            "neuron_index": index,
            "neuron_name": name,
            "finite_samples": finite.size,
            "nan_samples": int(np.isnan(values).sum()),
            "inf_samples": int(np.isinf(values).sum()),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
        })
    return rows


def behavior_statistics(B: np.ndarray, names: list[str], fps: float) -> list[dict]:
    return [{
        "label": label,
        "state_name": name,
        "samples": int(np.count_nonzero(B == label)),
        "seconds": float(np.count_nonzero(B == label) / fps),
        "percent": float(100 * np.count_nonzero(B == label) / B.size),
    } for label, name in enumerate(names)]


def plot_activity(X, B, fps, neuron_names, state_names, worm, duration, output):
    """Zeigt ausgewählte Spuren, alle Neuronen und zeitlich passende Labels."""
    count = min(X.shape[0], max(1, int(duration * fps)))
    time = np.arange(count) / fps
    colors = ListedColormap(plt.get_cmap("tab10").colors[:len(state_names)])
    norm = BoundaryNorm(np.arange(len(state_names) + 1) - 0.5, colors.N)
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 3, 1]}, layout="constrained",
    )
    for index in range(min(6, X.shape[1])):
        # Standardisierung nur zur Darstellung. Rohdaten und CSV bleiben unverändert.
        values = X[:count, index]
        finite = np.isfinite(values)
        normalized = np.full(values.shape, np.nan)
        if finite.any():
            std = values[finite].std()
            normalized[finite] = (values[finite] - values[finite].mean()) / (std or 1)
        axes[0].plot(time, normalized + index * 5, linewidth=0.8)
    axes[0].set_yticks(np.arange(min(6, X.shape[1])) * 5)
    axes[0].set_yticklabels(neuron_names[:6])
    axes[0].set_title("Erste sechs Neuronen: pro Neuron standardisiert und vertikal versetzt")
    axes[0].set_ylabel("Neuron")

    # Binränder: jede Messung belegt ein Intervall von 1/fps Sekunden.
    extent = (0, count / fps, X.shape[1] - 0.5, -0.5)
    image = axes[1].imshow(
        np.ma.masked_invalid(X[:count].T), aspect="auto", interpolation="nearest",
        extent=extent, cmap="viridis",
    )
    axes[1].set_title("Alle Neuronen: unveränderte deltaFOverF_bc-Werte")
    axes[1].set_ylabel("Neuronenindex")
    fig.colorbar(image, ax=axes[1], label="deltaFOverF_bc")
    image = axes[2].imshow(
        B[:count][None, :], aspect="auto", interpolation="nearest",
        extent=(0, count / fps, -0.5, 0.5), cmap=colors, norm=norm,
    )
    axes[2].set_yticks([])
    axes[2].set_ylabel("B")
    axes[2].set_xlabel("Zeit seit Aufnahmebeginn [s]")
    bar = fig.colorbar(image, ax=axes[2], ticks=np.arange(len(state_names)))
    bar.ax.set_yticklabels(state_names)
    fig.suptitle(f"Wurm {worm} · erste {count / fps:.1f} s · {fps:.4f} Hz")
    fig.savefig(output / "activity.png", dpi=150)
    plt.close(fig)


def plot_behavior(rows: list[dict], worm: int, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    bars = ax.bar([row["state_name"] for row in rows], [row["percent"] for row in rows])
    ax.bar_label(bars, fmt="%.1f %%", padding=4)
    ax.set_ylim(0, max(row["percent"] for row in rows) * 1.2 + 1)
    ax.set_ylabel("Anteil an allen Zeitpunkten [%]")
    ax.set_xlabel("Verhaltensklasse (Originalnamen aus States)")
    ax.set_title(f"Wurm {worm}: Klassenverteilung der gesamten Aufnahme")
    fig.savefig(output / "behavior.png", dpi=150)
    plt.close(fig)


def write_pair_example(X, B, state_names, win, output):
    """Zeigt die ersten drei Paare; legt kein großes Array für alle Paare an."""
    lines = [
        "prep_data(): Beispiel mit echten Messwerten",
        f"Gewünschte Fensterlänge: {win}",
        f"X.shape = {X.shape}",
        f"B.shape = {B.shape}",
        f"X_paired.shape = ({X.shape[0] - win}, 2, {win}, {X.shape[1]})",
        f"B_1.shape = ({X.shape[0] - win},)",
        "",
        "Pro Paar werden nur die ersten drei Neuronen angezeigt.",
        "Jede Zeile eines Fensters ist ein Zeitpunkt; jede Spalte ein Neuron.",
    ]
    for index in range(min(3, X.shape[0] - win)):
        label = int(B[index + win])
        lines.extend([
            "", f"Paar {index}:",
            f"  Erstes Fenster: X[{index}:{index + win}, :3]",
            np.array2string(X[index:index + win, :3], precision=4),
            f"  Zweites Fenster: X[{index + 1}:{index + win + 1}, :3]",
            np.array2string(X[index + 1:index + win + 1, :3], precision=4),
            f"  B_1[{index}] = B[{index + win}] = {label} ({state_names[label]})",
        ])
    (output / "prep_data_example.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(data, worm, args):
    X, B, fps, neuron_names, state_names = load_worm(data, worm)
    if args.win >= X.shape[0]:
        raise ValueError(f"Wurm {worm}: --win muss kleiner als {X.shape[0]} sein.")
    output = args.output / f"worm_{worm}"
    output.mkdir(parents=True, exist_ok=True)
    neurons = neuron_statistics(X, neuron_names)
    behaviors = behavior_statistics(B, state_names, fps)
    write_csv(output / "neuron_stats.csv", neurons)
    write_csv(output / "behavior_stats.csv", behaviors)
    plot_activity(X, B, fps, neuron_names, state_names, worm, args.duration, output)
    plot_behavior(behaviors, worm, output)
    write_pair_example(X, B, state_names, args.win, output)

    report = [
        f"Wurmdaten: Datensatz {worm} (Nummerierung beginnt bei 0)",
        f"Quelldatei: {args.data.resolve()}",
        "Quelle der Messwerte: NoStim_Data['deltaFOverF_bc']",
        "Messart: Calcium-/Fluoreszenzsignale von Neuronen; keine EEG-Kanäle.",
        "",
        f"Zeitpunkte: {X.shape[0]}", f"Neuronen: {X.shape[1]}",
        f"Abtastrate: {fps:.9f} Hz", f"Abstand zwischen Messungen: {1 / fps:.6f} s",
        f"Dauer (Zeitpunkte / fps): {X.shape[0] / fps:.3f} s",
        f"Zeitstempel der letzten Messung: {(X.shape[0] - 1) / fps:.3f} s",
        f"Nicht endliche Messwerte (NaN/Inf): {int((~np.isfinite(X)).sum())}",
        f"Konstante Neuronen (std = 0): {sum(row['std'] == 0 for row in neurons)}",
        "",
        "Zuordnung zu functions.py / main.py:",
        f"  Database.neuron_traces.shape = {X.T.shape} (Neuronen, Zeitpunkte)",
        f"  X = data.neuron_traces.T: {X.shape} (Zeitpunkte, Neuronen)",
        f"  B = data.states: {B.shape} (ein Klassenlabel pro Zeitpunkt)",
        "  B entsteht aus den binären Masken in States, in deren vorhandener Reihenfolge.",
        "  Geprüft: an jedem Zeitpunkt ist genau eine Maske aktiv.",
        "",
        "Klassen der gesamten Aufnahme:",
    ]
    report.extend(
        f"  {row['label']}: {row['state_name']:<8} {row['samples']:>5} Zeitpunkte, "
        f"{row['seconds']:>8.2f} s, {row['percent']:>5.1f} %" for row in behaviors
    )
    report.extend([
        "", f"prep_data(X, B, win={args.win}):",
        f"  X_paired.shape = ({X.shape[0] - args.win}, 2, {args.win}, {X.shape[1]})",
        f"  B_1.shape = ({X.shape[0] - args.win},)",
        f"  Fensterlänge: {args.win} Samples; Verschiebung: 1 Sample = {1 / fps:.6f} s.",
        "  Details und echte Beispielwerte: prep_data_example.txt",
        "", "Darstellung:",
        f"  activity.png zeigt die ersten {args.duration:g} Sekunden (auf ganze Samples gerundet).",
        "  Nur die ausgewählten Kurven sind für die Darstellung standardisiert.",
        "  Alle Kennwerte und die Klassenverteilung verwenden die gesamte Aufnahme.",
        "  Es werden keine Neuronen ausgeschlossen, Filter angewendet oder Modelle trainiert.",
    ])
    text = "\n".join(report) + "\n"
    (output / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Ergebnisse: {output.resolve()}\n")
    return {"worm": worm, "samples": X.shape[0], "neurons": X.shape[1],
            "fps": fps, "duration_seconds": X.shape[0] / fps,
            "classes": len(state_names), "nonfinite_values": int((~np.isfinite(X)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--worm", type=int, default=0, help="Wurmindex, Standard: 0")
    selection.add_argument("--all", action="store_true", help="Alle Würmer analysieren")
    parser.add_argument("--duration", type=float, default=60, help="Sekunden in activity.png, Standard: 60")
    parser.add_argument("--win", type=int, default=15, help="Fensterlänge für das prep_data-Beispiel")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Pfad zu NoStim_Data.mat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ausgabeverzeichnis")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0 or args.win < 1:
        parser.error("--duration muss endlich und positiv sein; --win muss mindestens 1 sein.")
    if not args.data.is_file():
        parser.error(f"MAT-Datei nicht gefunden: {args.data}")

    print(f"Lade {args.data} ...", flush=True)
    data = mat73.loadmat(str(args.data))["NoStim_Data"]
    count = len(data["deltaFOverF_bc"])
    if not args.all and not 0 <= args.worm < count:
        parser.error(f"--worm muss zwischen 0 und {count - 1} liegen.")
    args.output.mkdir(parents=True, exist_ok=True)
    # Separate Übersicht für Einzelauswertung und Gesamtauswertung.
    worms = range(count) if args.all else [args.worm]
    rows = [analyze(data, worm, args) for worm in worms]
    name = "overview.csv" if args.all else f"overview_worm_{args.worm}.csv"
    write_csv(args.output / name, rows)


if __name__ == "__main__":
    main()
