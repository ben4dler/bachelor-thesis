import numpy as np
import os
from pathlib import Path

class ProcessedDataStroage:
    def __init__(self):
        self.PROJECT_ROOT = Path.cwd().resolve()
        # Nach oben gehen, bis das Datenverzeichnis gefunden wird
        while not (self.PROJECT_ROOT / "data" / "raw" / "ds005363-download").is_dir():
            if self.PROJECT_ROOT == self.PROJECT_ROOT.parent:
                raise FileNotFoundError("Projektverzeichnis nicht gefunden.")

            self.PROJECT_ROOT = self.PROJECT_ROOT.parent

        self.DATA_ROOT = self.PROJECT_ROOT / "data" / "raw" / "ds005363-download"
        self.DATA = self.PROJECT_ROOT / "data"
        self.DIRECTORY = self.DATA/"processed"
        os.makedirs(self.DIRECTORY, exist_ok=True)

    def save_subject(self, subject, arrays):
        subject = f"sub-{subject}"
        
        filename = f"{subject}_X_train"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        np.save(completeName, arrays.get("X_train"))

        filename = f"{subject}_X_test"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        np.save(completeName, arrays.get("X_test"))

        filename = f"{str(subject)}_labels_train"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        np.save(completeName, arrays.get("labels_train"))

        filename = f"{str(subject)}_labels_test"
        completeName = os.path.join(self.DIRECTORY, filename+".npy") 
        np.save(completeName, arrays.get("labels_test"))

    def load_subject(self, subject): 
        subject = f"sub-{subject}"

        filename = f"{subject}_X_train"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        X_train = np.load(completeName, "r")

        filename = f"{subject}_X_test"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        X_test = np.load(completeName, "r")

        filename = f"{subject}_labels_train"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        labels_train = np.load(completeName, "r")

        filename = f"{subject}_labels_test"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        labels_test = np.load(completeName, "r")

        return X_train, X_test, labels_train, labels_test