import numpy as np
import os
from pathlib import Path
from preprocess import Preprocessor
from data_reader import DataReader


class ProcessedDataStroage:
    def __init__(self):
        self.PROJECT_ROOT = Path.cwd().resolve()
        while not (self.PROJECT_ROOT / "data" / "raw" / "ds005363-download").is_dir():
            if self.PROJECT_ROOT == self.PROJECT_ROOT.parent:
                raise FileNotFoundError("Projektverzeichnis nicht gefunden.")

            self.PROJECT_ROOT = self.PROJECT_ROOT.parent

        self.DATA_ROOT = self.PROJECT_ROOT / "data" / "raw" / "ds005363-download"
        self.DATA = self.PROJECT_ROOT / "data"
        self.DIRECTORY = self.DATA/"processed"
        os.makedirs(self.DIRECTORY, exist_ok=True)
        self.data_reader = DataReader()
        self.preprocessor = Preprocessor()

    def preprocess_and_save_subjects_from_range(self, subjects_range):
        for sub in subjects_range:
            subject = sub + 100
            if subject == 106:
                continue
            subject = str(subject)
            raw, _, _ = self.data_reader.read_data(
                subject=str(subject),
            )
        
            self.preprocessor.run_pipeline(raw, use_filter=True, use_ica=True, use_autoreject=True)

            del raw

            epochs_train, epochs_test = self.preprocessor.split_train_test()

            X_train, X_test = self.preprocessor.scale(epochs_train.get_data(), epochs_test.get_data())
            
            labels_train, category_per_epoch_train = self.preprocessor.create_labels_for_epochs(epochs_train)    # Form (1638, 120)
            
            labels_test,category_per_epoch_test = self.preprocessor.create_labels_for_epochs(epochs_test)      # Form (410, 120)
            
            self.save_subject(subject, {"X_train": X_train, "X_test": X_test, "labels_train": labels_train, "labels_test": labels_test, "category_per_epoch_train": category_per_epoch_train, "category_per_epoch_test": category_per_epoch_test})

    def read_subjects_from_range(self, subjects_range):
        X_train = []
        X_test = []
        labels_train = []
        labels_test = []
        category_per_epoch_all_subs_train = []
        category_per_epoch_all_subs_test = []

        for sub in subjects_range:
            subject = sub + 100
            if subject == 106:
                continue
            values = self.load_subject(subject)
            X_train.append(values[0])
            X_test.append(values[1])
            labels_train.append(values[2])
            labels_test.append(values[3])
            category_per_epoch_all_subs_train.append(values[4])
            category_per_epoch_all_subs_test.append(values[5])

        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        labels_test = np.concatenate(labels_test, axis=0)
        category_per_epoch_all_subs_train = np.concatenate(category_per_epoch_all_subs_train, axis=0)
        category_per_epoch_all_subs_test = np.concatenate(category_per_epoch_all_subs_test, axis=0)

        return X_train, X_test, labels_train, labels_test, category_per_epoch_all_subs_train, category_per_epoch_all_subs_test

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

        filename = f"{str(subject)}_category_per_epoch_train"
        completeName = os.path.join(self.DIRECTORY, filename+".npy") 
        np.save(completeName, arrays.get("category_per_epoch_train"))

        filename = f"{str(subject)}_category_per_epoch_test"
        completeName = os.path.join(self.DIRECTORY, filename+".npy") 
        np.save(completeName, arrays.get("category_per_epoch_test"))

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

        filename = f"{subject}_category_per_epoch_train"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        category_per_epoch_train = np.load(completeName, "r")

        filename = f"{subject}_category_per_epoch_test"
        completeName = os.path.join(self.DIRECTORY, filename+".npy")
        category_per_epoch_test = np.load(completeName, "r")


        return X_train, X_test, labels_train, labels_test, category_per_epoch_train, category_per_epoch_test