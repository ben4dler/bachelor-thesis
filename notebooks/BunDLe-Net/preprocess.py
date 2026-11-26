import mne
import autoreject
import numpy as np
from epoch_creator import get_categories, get_events, create_event_ids, create_epochs

# Filter - ICA - Epoch erstellen - Autoreject


class Preprocessor:

    def __init__(self, number_of_categories=None):
        self.NUMBER_OF_IMAGES_PER_CATEGORY = 16
        self.LABEL_BEFORE_IMAGE_ONSET = 0
        self.BATCH_SIZE = 1024
        self.GAMMA = 0.7
        self.WINDOW_LENGTH = 15
        self.number_of_categories = number_of_categories

    # TODO: Implement PCA and compare before and after; check creations of epochs with official mne doc. https://mne.tools/stable/auto_examples/decoding/decoding_unsupervised_spatial_filter.html
    # CAR - Spatial filtering 
    # Normalize
    
    def run_pipeline(self, raw, use_filter=True, use_ica=True, use_autoreject=True):
        if use_filter:
            self.filter(raw)
        if use_ica:
            self.ica(raw)
        self.create_epochs(raw)
        if use_autoreject:
           self.autoreject(raw)

    def filter(self, raw):
        raw.load_data()
        raw.filter(l_freq=1, h_freq=None)
        raw.notch_filter(np.arange(50,251,50))
        print(f"Successfully filtered the data.")
        return raw

    def ica(self, raw):
        ica = mne.preprocessing.ICA(n_components=20)
        ica.fit(raw)
        raw.load_data()
        ica.apply(raw)
        print(f"Successfully ran ICA on the data.")
        return raw

    def create_epochs(self, raw):

        self.events_arr, self.event_dict = get_events(raw)

        self.categories = get_categories(self.number_of_categories)

        self.event_ids = create_event_ids(self.event_dict, self.categories)

        self.epochs = create_epochs(raw, self.events_arr, self.event_ids)
        self.epochs.resample(100)

    def autoreject(self, raw):

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
        return X_train.astype(np.float32), X_test.astype(np.float32)

    def create_labels_for_epochs(self, epochs):

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

        return labels.astype(np.float32), category_per_epoch

    def list_all_window_pairs(
        self, number_of_epochs, number_of_time_points, window_length
    ):

        number_of_start_points_per_epoch = number_of_time_points - window_length

        epoch_numbers = []
        start_time_points = []
        for epoch_number in range(number_of_epochs):
            for start_time_point in range(number_of_start_points_per_epoch):
                epoch_numbers.append(epoch_number)
                start_time_points.append(start_time_point)

        return np.array(epoch_numbers), np.array(start_time_points)

    def build_window_features(self, eeg_data, window_length):
        """Rohfenster als Features für einen SlidingEstimator.

        Nimmt pro Startpunkt dasselbe Fenster, das tau sieht (das zweite Fenster
        aus build_training_batch: start + 1 ... start + window_length) und legt
        Kanäle und Zeit zu einem Feature-Vektor zusammen.

        Eingabe:  (Epochen, Kanäle, Zeitpunkte)
        Rückgabe: (Epochen, Kanäle * window_length, Zeitpunkte - window_length)
        Die letzte Achse hat damit genau so viele Positionen wie die tau-Werte,
        und Position p gehört zum Zeitpunkt times[window_length + p].
        """
        number_of_epochs, number_of_channels, number_of_time_points = eeg_data.shape
        number_of_start_points = number_of_time_points - window_length

        windows = []
        for start_time_point in range(number_of_start_points):
            second_window = eeg_data[
                :, :, start_time_point + 1 : start_time_point + window_length + 1
            ]
            windows.append(
                second_window.reshape(number_of_epochs, number_of_channels * window_length)
            )

        return np.stack(windows, axis=2)

    def build_training_batch(
        self, eeg_data, labels, epoch_numbers, start_time_points, window_length
    ):

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
