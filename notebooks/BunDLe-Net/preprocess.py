import mne
import autoreject
import numpy as np
from epoch_creator import get_categories, get_events, create_event_ids, create_epochs

# Filter - ICA - Epoch erstellen - Autoreject


class Preprocessor:

    def __init__(self, raw, number_of_categories=None):
        self.raw = raw
        self.NUMBER_OF_IMAGES_PER_CATEGORY = 16
        self.LABEL_BEFORE_IMAGE_ONSET = 0
        self.BATCH_SIZE = 1024
        self.GAMMA = 0.9
        self.WINDOW_LENGTH = 25
        self.number_of_categories = number_of_categories

    # TODO: Implement PCA and compare before and after; check creations of epochs with official mne doc. https://mne.tools/stable/auto_examples/decoding/decoding_unsupervised_spatial_filter.html

    def run_pipeline(self, use_filter=True, use_ica=True, use_autoreject=True):
        if use_filter:
            self.filter()
        if use_ica:
            self.ica()
        self.create_epochs()
        if use_autoreject:
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

        self.categories = get_categories(self.number_of_categories)

        self.event_ids = create_event_ids(self.event_dict, self.categories)

        self.epochs = create_epochs(self.raw, self.events_arr, self.event_ids)

    def autoreject(self):

        self.epochs.resample(250)

        """ar = autoreject.AutoReject(n_interpolate=[1, 4], random_state=11,
                                n_jobs=-1, verbose=True)     
        ar.fit(epochs[fit_idx])
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)"""

        reject = autoreject.get_rejection_threshold(self.epochs)

        self.epochs = self.epochs.drop_bad(reject=reject)

        print("Autoreject complete.")

    def make_B(self, epochs):
        codes = epochs.events[:, 2]
        cat = (codes - 1) // 16 + 1  # 1=face, 2=animal, 3=place, 4=object
        B = np.zeros((len(epochs), len(epochs.times)), dtype=int)
        B[:, epochs.times >= 0] = cat[:, None]
        return B

    def split_train_test(self):

        n_train = int(0.8 * len(self.epochs))
        rng = np.random.default_rng(11)
        fit_idx = rng.choice(len(self.epochs), 200, replace=False)
        idx = rng.permutation(len(self.epochs))

        epochs_train = self.epochs[idx[:n_train]]
        epochs_test = self.epochs[idx[n_train:]]

        # X_train = epochs_train.get_data()
        # X_test = epochs_test.get_data()

        return epochs_train, epochs_test

    def scale(self, X_train, X_test):
        scaler = mne.decoding.Scaler(scalings="mean")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def create_labels_for_epochs(self, epochs):
        """
        Gibt eine Matrix mit der Form (Anzahl Epochs, Anzahl Zeitpunkte) zurück.
        Vor dem Bildbeginn steht 0, ab dem Bildbeginn die Kategorie:
        1 = Gesicht, 2 = Tier, 3 = Ort, 4 = Objekt.
        """
        image_numbers = epochs.events[:, 2]

        category_per_epoch = (
            image_numbers - 1
        ) // self.NUMBER_OF_IMAGES_PER_CATEGORY + 1

        number_of_epochs = len(epochs)
        number_of_time_points = len(epochs.times)
        labels = np.full(
            (number_of_epochs, number_of_time_points),
            self.LABEL_BEFORE_IMAGE_ONSET,
            dtype=int,
        )

        time_points_after_image_onset = epochs.times >= 0
        for epoch_number in range(number_of_epochs):
            labels[epoch_number, time_points_after_image_onset] = category_per_epoch[
                epoch_number
            ]

        return labels, category_per_epoch

    def list_all_window_pairs(
        self, number_of_epochs, number_of_time_points, window_length
    ):
        """
        Jedes Fensterpaar ist eindeutig durch zwei Zahlen beschrieben:
        aus welcher Epoch es stammt und an welchem Zeitpunkt das erste Fenster beginnt.
        Das zweite Fenster ist um einen Zeitpunkt verschoben, deshalb braucht ein Paar
        window_length + 1 Zeitpunkte. Paare über Epoch-Grenzen hinweg gibt es nicht.
        """

        number_of_start_points_per_epoch = number_of_time_points - window_length

        epoch_numbers = []
        start_time_points = []
        for epoch_number in range(number_of_epochs):
            for start_time_point in range(number_of_start_points_per_epoch):
                epoch_numbers.append(epoch_number)
                start_time_points.append(start_time_point)

        return np.array(epoch_numbers), np.array(start_time_points)

    def build_training_batch(
        self, eeg_data, labels, epoch_numbers, start_time_points, window_length
    ):
        """
        eeg_data:  Form (Anzahl Epochs, Anzahl Kanäle, Anzahl Zeitpunkte)
        labels:    Form (Anzahl Epochs, Anzahl Zeitpunkte)

        Rückgabe wie bei prep_data:
        window_pairs:  Form (Anzahl Paare, 2, window_length, Anzahl Kanäle)
        target_labels: Form (Anzahl Paare,)
        """
        window_pairs = []
        target_labels = []

        for epoch_number, start_time_point in zip(epoch_numbers, start_time_points):
            one_epoch = eeg_data[epoch_number].T  # Form (Zeitpunkte, Kanäle)

            first_window = one_epoch[
                start_time_point : start_time_point + window_length
            ]
            second_window = one_epoch[
                start_time_point + 1 : start_time_point + window_length + 1
            ]

            window_pairs.append(np.stack([first_window, second_window]))

            last_time_point_of_second_window = start_time_point + window_length
            target_labels.append(labels[epoch_number, last_time_point_of_second_window])

        return np.array(window_pairs, dtype=np.float32), np.array(target_labels)
