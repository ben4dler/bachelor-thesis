import mne
import autoreject
from epoch_creator import get_categories, get_events, create_event_ids, create_epochs

# Filter - ICA - Epoch erstellen - Autoreject


class Preprocessor:
    def __init__(self, raw):
        self.raw = raw

    def run_pipeline(self):
        self.filter()
        self.ica()
        self.create_epochs()
        self.autoreject()

    def filter(self):
        self.raw.load_data()
        self.raw.filter(l_freq=0.5, h_freq=None)
        print(f"Successfully filtered the data.")
        return self.raw

    def ica(self):
        ica = mne.preprocessing.ICA(n_components=20)
        ica.fit(self.raw)
        self.raw.load_data()
        ica.apply(self.raw)
        print(f"Successfully ran ICA on the data.")
        return self.raw

    def create_epochs(self):

        self.events_arr, self.event_dict = get_events(self.raw)

        self.categories = get_categories()

        self.event_ids = create_event_ids(self.event_dict, self.categories)

        self.epochs = create_epochs(self.raw, self.events_arr, self.event_ids)

    def autoreject(self):

        self.epochs.resample(250)

        """ar = autoreject.AutoReject(n_interpolate=[1, 4], random_state=11,
                                n_jobs=-1, verbose=True)     
        ar.fit(epochs[fit_idx])
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)"""

        reject = autoreject.get_rejection_threshold(
            self.epochs
        )  # z. B. {'eeg': 1.2e-4}
        self.epochs_clean = self.epochs.copy().drop_bad(reject=reject)
        print("Autoreject complete.")
