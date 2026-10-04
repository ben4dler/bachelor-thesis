"""Beschreibende Analyse der EEG-Daten aus ds005363.

Das Skript liest eine BrainVision-Aufnahme ein und beschreibt sowohl den
Dateiaufbau als auch die wichtigsten Eigenschaften der Messung. Es erzeugt:

* eine Textzusammenfassung der BIDS-/BrainVision-Struktur,
* Kanalstatistiken als CSV-Datei,
* relative und absolute Leistung in üblichen Frequenzbändern als CSV-Datei,
* Grafiken für EEG-Signale, Leistungsdichtespektrum und Events.

Die Analyse ist bewusst deskriptiv. Es werden keine Filterung, keine
Artefaktkorrektur und keine klinischen Aussagen vorgenommen.

Voraussetzungen:
    pip install mne numpy matplotlib

Beispiele (aus dem Projekt-Root):
    python3 "notebooks/EEG Visualizer/analyze_eeg.py"
    python3 "notebooks/EEG Visualizer/analyze_eeg.py" --subject 101 --duration 60

Die Rohdaten werden standardmäßig nicht dauerhaft in den Arbeitsspeicher
geladen. Für die Kennwerte und Grafiken wird nur der Anfang der Aufnahme
(`--duration`, standardmäßig 60 Sekunden) eingelesen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on local environment
    missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
    raise SystemExit(
        f"Fehlendes Paket: {missing}. Installiere die Voraussetzungen mit "
        "'pip install mne numpy matplotlib'."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "ds005363"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "analysis_output"

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Liest eine TSV-Datei robust gegen ein UTF-8-BOM ein."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Schreibt Ergebnisdaten als TSV, damit auch Excel sie leicht öffnen kann."""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def first_matching(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Keine Datei mit Muster {pattern!r} in {directory}")
    return matches[0]


def require_materialized(path: Path) -> Path:
    """Prüft, ob ein Git-Annex-Link tatsächlich auf lokale Daten zeigt."""

    if path.is_file():
        return path
    if path.is_symlink():
        raise FileNotFoundError(
            f"{path} ist ein Git-Annex-Symlink, aber die Datei ist lokal nicht vorhanden.\n"
            "Lade zuerst die Annex-Inhalte herunter, zum Beispiel mit:\n"
            f"  git annex get -- {path}\n"
            "Falls kein Annex-Remote konfiguriert ist, muss der Datensatz "
            "ds005363 erneut über seine Datenquelle heruntergeladen werden."
        )
    raise FileNotFoundError(f"Datei ist nicht vorhanden: {path}")


def subject_directory(data_root: Path, subject: str) -> Path:
    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    directory = data_root / subject_label / "ses-01" / "eeg"
    if not directory.is_dir():
        raise FileNotFoundError(f"EEG-Verzeichnis nicht gefunden: {directory}")
    return directory


def finite_stats(values: np.ndarray) -> dict[str, float]:
    """Berechnet Kennwerte ohne NaN-/Inf-Werte."""

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {key: float("nan") for key in ("mean", "std", "min", "max", "peak_to_peak")}
    return {
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "peak_to_peak": float(np.ptp(finite_values)),
    }


def integrate_power(frequencies: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    # np.trapezoid existiert erst in neueren NumPy-Versionen.
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(spectrum[mask], frequencies[mask]))


def make_plots(
    data_uv: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    events: list[dict[str, str]],
    output_dir: Path,
) -> None:
    """Erstellt die drei Übersichtsabbildungen."""

    time = np.arange(data_uv.shape[1]) / sfreq

    # Signale: maximal zwölf Kanäle, zur besseren Lesbarkeit vertikal versetzt.
    number_to_plot = min(12, data_uv.shape[0])
    fig, ax = plt.subplots(figsize=(15, 8))
    visible = data_uv[:number_to_plot]
    scale = max(float(np.nanmedian(np.nanstd(visible, axis=1))), 1.0)
    offsets = np.arange(number_to_plot)[::-1] * 5.0 * scale
    for index in range(number_to_plot):
        ax.plot(time, data_uv[index] + offsets[index], linewidth=0.45)
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names[:number_to_plot])
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Kanal")
    ax.set_title("EEG-Signale (erste Kanäle, Rohdaten in µV)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "eeg_signals.png", dpi=150)
    plt.close(fig)

    # Mittleres Leistungsdichtespektrum über die EEG-Kanäle.
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_spectrum = np.nanmean(spectrum, axis=0)
    ax.semilogy(frequencies, np.maximum(mean_spectrum, np.finfo(float).tiny))
    for name, (low, high) in BANDS.items():
        ax.axvspan(low, high, alpha=0.12, label=name)
    ax.set_xlim(1, 45)
    ax.set_xlabel("Frequenz (Hz)")
    ax.set_ylabel("Leistungsdichte (V²/Hz)")
    ax.set_title("Mittleres EEG-Leistungsdichtespektrum")
    ax.legend(ncol=len(BANDS), loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "eeg_power_spectrum.png", dpi=150)
    plt.close(fig)

    # Häufigkeit der im Event-TSV vorhandenen Trigger.
    if events:
        labels = [row.get("trial_type", "unbekannt") or "unbekannt" for row in events]
        counts = Counter(labels)
        labels, values = zip(*counts.most_common(25))
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.set_ylabel("Anzahl")
        ax.set_title("Häufigste Event-/Trigger-Typen")
        fig.tight_layout()
        fig.savefig(output_dir / "event_counts.png", dpi=150)
        plt.close(fig)


def analyze(subject: str, duration: float, data_root: Path, output_root: Path) -> Path:
    eeg_dir = subject_directory(data_root, subject)
    output_dir = output_root / (subject if subject.startswith("sub-") else f"sub-{subject}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vhdr_path = require_materialized(first_matching(eeg_dir, "*_eeg.vhdr"))
    json_path = require_materialized(first_matching(eeg_dir, "*_eeg.json"))
    channels_path = require_materialized(first_matching(eeg_dir, "*_channels.tsv"))
    events_path = require_materialized(first_matching(eeg_dir, "*_events.tsv"))

    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    channels = read_tsv(channels_path)
    events = read_tsv(events_path)

    print(f"Lese BrainVision-Aufnahme ein: {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise RuntimeError("Die Aufnahme enthält keine als EEG markierten Kanäle.")

    channel_names = [raw.ch_names[index] for index in eeg_picks]
    sfreq = float(raw.info["sfreq"])
    total_duration = float(raw.n_times / sfreq)
    read_duration = min(max(duration, 1.0), total_duration)
    sample_count = max(1, int(read_duration * sfreq))

    # MNE liefert Volt; für Tabellen und Signalstatistiken werden µV verwendet.
    data_uv = raw.get_data(picks=eeg_picks, start=0, stop=sample_count) * 1e6
    psd, frequencies = mne.time_frequency.psd_array_welch(
        data_uv / 1e6,
        sfreq=sfreq,
        fmin=1.0,
        fmax=45.0,
        n_fft=min(2048, data_uv.shape[1]),
        verbose=False,
    )

    channel_rows: list[dict[str, Any]] = []
    channel_metadata = {row.get("name", ""): row for row in channels}
    for index, name in enumerate(channel_names):
        stats = finite_stats(data_uv[index])
        info = channel_metadata.get(name, {})
        channel_rows.append(
            {
                "channel": name,
                "type": info.get("type", "EEG"),
                "status": info.get("status", ""),
                "mean_uv": f"{stats['mean']:.6g}",
                "std_uv": f"{stats['std']:.6g}",
                "min_uv": f"{stats['min']:.6g}",
                "max_uv": f"{stats['max']:.6g}",
                "peak_to_peak_uv": f"{stats['peak_to_peak']:.6g}",
            }
        )
    write_tsv(
        output_dir / "channel_statistics.tsv",
        channel_rows,
        [
            "channel",
            "type",
            "status",
            "mean_uv",
            "std_uv",
            "min_uv",
            "max_uv",
            "peak_to_peak_uv",
        ],
    )

    band_rows: list[dict[str, Any]] = []
    channel_band_power: dict[str, dict[str, float]] = {}
    for index, name in enumerate(channel_names):
        channel_band_power[name] = {
            band: integrate_power(frequencies, psd[index], low, high)
            for band, (low, high) in BANDS.items()
        }
    total_band_power = {
        band: float(np.nanmean([values[band] for values in channel_band_power.values()]))
        for band in BANDS
    }
    total_power = sum(value for value in total_band_power.values() if np.isfinite(value))
    for band, value in total_band_power.items():
        band_rows.append(
            {
                "band": band,
                "frequency_range_hz": f"{BANDS[band][0]}-{BANDS[band][1]}",
                "mean_absolute_power_v2": f"{value:.6g}",
                "mean_relative_power": f"{value / total_power:.6g}" if total_power else "nan",
            }
        )
    write_tsv(
        output_dir / "band_power.tsv",
        band_rows,
        ["band", "frequency_range_hz", "mean_absolute_power_v2", "mean_relative_power"],
    )

    make_plots(data_uv, channel_names, sfreq, frequencies, psd, events, output_dir)

    event_types = Counter(row.get("trial_type", "unbekannt") or "unbekannt" for row in events)
    channel_type_counts = Counter(raw.get_channel_types(picks=eeg_picks))
    report_lines = [
        "EEG-ANALYSEBERICHT",
        "==================",
        "",
        "1. Dateiaufbau",
        f"BIDS-Datensatz: {data_root}",
        f"Subjekt/Sitzung: {eeg_dir.parent.parent.name}/{eeg_dir.parent.name}",
        f"Aufgabe: {metadata.get('TaskName', 'unbekannt')}",
        "BrainVision-Dateien: .vhdr (Header), .eeg (Messwerte), .vmrk (Marker)",
        "BIDS-Sidecars: *_eeg.json (Messmetadaten), *_channels.tsv (Kanäle), *_events.tsv (Events)",
        "",
        "2. Aufnahme",
        f"Kanäle insgesamt: {raw.info['nchan']}",
        f"EEG-Kanäle: {len(eeg_picks)}",
        f"Kanaltypen: {dict(channel_type_counts)}",
        f"Abtastrate: {sfreq:.3f} Hz",
        f"Anzahl Samples: {raw.n_times}",
        f"Aufnahmedauer: {total_duration:.3f} s ({total_duration / 60:.2f} min)",
        f"Analysefenster: erste {read_duration:.3f} s",
        f"Referenz laut JSON: {metadata.get('EEGReference', 'unbekannt')}",
        f"Elektrodenlayout: {metadata.get('EEGPlacementScheme', 'unbekannt')}",
        "",
        "3. Kanal- und Signalbeschreibung",
        "Die vollständigen Kennwerte stehen in channel_statistics.tsv.",
        "Die Spalten mean/std/min/max/peak_to_peak sind in µV angegeben.",
        "",
        "4. Frequenzanalyse",
        "Die Band-Leistungen stehen in band_power.tsv.",
        "Berechnet wurden Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz),",
        "Beta (13–30 Hz) und Gamma (30–45 Hz) aus dem Analysefenster.",
        "",
        "5. Events/Trigger",
        f"Anzahl Event-Zeilen: {len(events)}",
        f"Event-Typen: {dict(event_types)}",
        "Die Trigger werden aus *_events.tsv übernommen; ihre Codes werden nicht interpretiert.",
        "",
        "Hinweis: Die Auswertung ist deskriptiv und ersetzt keine Vorverarbeitung.",
        "Insbesondere wurden keine Filterung und keine Artefaktkorrektur durchgeführt.",
    ]
    (output_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nErgebnisse gespeichert in: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="101", help="Subjekt-ID, z. B. 101 oder sub-101")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Dauer des Analysefensters ab Beginn der Aufnahme in Sekunden (Standard: 60)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"BIDS-Datenverzeichnis (Standard: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Ausgabeverzeichnis (Standard: {DEFAULT_OUTPUT_ROOT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        analyze(args.subject, args.duration, args.data_root, args.output_root)
    except FileNotFoundError as exc:
        raise SystemExit(f"Datei nicht verfügbar: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"Analyse konnte nicht durchgeführt werden: {exc}") from exc
