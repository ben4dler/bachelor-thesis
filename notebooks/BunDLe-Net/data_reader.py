import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
import mne
import csv
import json
import sys

print(__doc__)


class DataReader:
    def __init__(self):
        self.PROJECT_ROOT = Path.cwd().resolve()
        # Nach oben gehen, bis das Datenverzeichnis gefunden wird
        while not (self.PROJECT_ROOT / "data" / "raw" / "ds005363-download").is_dir():
            if self.PROJECT_ROOT == self.PROJECT_ROOT.parent:
                raise FileNotFoundError("Projektverzeichnis nicht gefunden.")

            self.PROJECT_ROOT = self.PROJECT_ROOT.parent

        self.DATA_ROOT = self.PROJECT_ROOT / "data" / "raw" / "ds005363-download"

        self.BUNDLE_DIR = self.PROJECT_ROOT / "notebooks" / "BunDLe-Net"
        if str(self.BUNDLE_DIR) not in sys.path:
            sys.path.insert(0, str(self.BUNDLE_DIR))

        print("Arbeitsverzeichnis:", Path.cwd())
        print("Datenverzeichnis:", self.DATA_ROOT)

    def first_matching(self, directory: Path, pattern: str) -> Path:
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Keine Datei mit Muster {pattern!r} in {directory}"
            )
        return matches[0]

    def read_tsv(self, path: Path) -> list[dict[str, str]]:
        """Liest eine TSV-Datei robust gegen ein UTF-8-BOM ein."""

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file, delimiter="\t"))

    def subject_directory(self, data_root: Path, subject: str) -> Path:
        subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
        directory = data_root / subject_label / "ses-01" / "eeg"
        if not directory.is_dir():
            raise FileNotFoundError(f"EEG-Verzeichnis nicht gefunden: {directory}")
        return directory

    # subjects_dir = subject_directory(DATA_ROOT, "102")

    def read_data(self, subject: str, duration: float):
        eeg_dir = self.subject_directory(self.DATA_ROOT, subject)

        vhdr_path = self.first_matching(eeg_dir, "*_eeg.vhdr")
        json_path = self.first_matching(eeg_dir, "*_eeg.json")
        channels_path = self.first_matching(eeg_dir, "*_channels.tsv")
        events_path = self.first_matching(eeg_dir, "*_events.tsv")

        with json_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        channels = self.read_tsv(channels_path)
        events = self.read_tsv(events_path)

        print(f"Lese BrainVision-Aufnahme ein: {vhdr_path}")
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise RuntimeError("Die Aufnahme enthält keine als EEG markierten Kanäle.")

        raw.set_montage("standard_1020")

        return raw, channels, events
