import sys
from data_reader import DataReader

sys.path.append(r"../")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

#########################################
############## Loading Data #############
#########################################


class Database:
    """
    Loading neuronal and behavioural data from matlab files

    Attributes:
        data_set_no (int): The number of the data set.
        states (numpy.ndarray): A single array of states, where each number corresponds to a behaviour.
        state_names (list): List of state names.
        neuron_traces (numpy.ndarray): Array of neuron traces.
        neuron_names (numpy.ndarray): Array of neuron names.
        fps (float): Frames per second.

    Methods:
        exclude_neurons: Excludes specified neurons from the database.
        categorise_neurons: Categorises neurons based on whether it is sensory,
                            inter or motor neuron.

    """

    def __init__(self):
        data_reader = DataReader()

        self.raw, self.channels, self.events = data_reader.read_data(
            subject="101",
            duration=60.0,
        )

        # EEG-Kanäle auswählen
        eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=[])

        # Messwerte: (Kanäle, Zeitpunkte)
        self.neuron_traces = self.raw.get_data(picks=eeg_picks)

        # Kanalnamen in derselben Reihenfolge wie die Messwerte
        self.channel_names = np.array(
            [self.raw.ch_names[i] for i in eeg_picks], dtype=object
        )

        # Abtastrate der Aufnahme: bei dir 1000 Hz
        self.fps = self.raw.info["sfreq"]

        self.state_names = [
            "keine Kategorie",
            "Gesicht",
            "Tier",
            "Ort",
            "Objekt",
            "Catch",
        ]

        # images of faces = S1-16 images of animals = S17-32 images of places = S33-48 images of objects = S49-64 catch trials = S65-69
        # button_press = S99 run_onset = S100+run_number (8 runs in total) run_end = S199

        self.stimulus_to_state = {}

        for code in range(1, 17):
            self.stimulus_to_state[code] = 1  # Gesicht

        for code in range(17, 33):
            self.stimulus_to_state[code] = 2  # Tier

        for code in range(33, 49):
            self.stimulus_to_state[code] = 3  # Ort

        for code in range(49, 65):
            self.stimulus_to_state[code] = 4  # Objekt

        for code in range(65, 70):
            self.stimulus_to_state[code] = 5

        labels = []
        self.B = np.zeros(self.neuron_traces.shape[1], dtype=int)
        for event in self.events:
            code = self.extract_s(event["trial_type"])

            if code in self.stimulus_to_state:
                label = self.stimulus_to_state[code]
            elif code == 199 or code in range(101, 109):
                label = 0
            else:
                continue

            sample = int(event["sample"])
            labels.append((sample, label))

        l = len(labels)

        for e in range(l):
            current_sample, label = labels[e]
            if e + 1 < l:
                next_sample = labels[e + 1][0]
            else:
                next_sample = l

            self.B[current_sample:next_sample] = label

    def extract_s(self, trial_type):
        if not trial_type.startswith("Stimulus/S"):
            return None

        number = trial_type.removeprefix("Stimulus/S").strip()
        return int(number)


########################################################
##########  Preparing the data for BunDLe Net ##########
########################################################


def prep_data(X, B, win=15):
    """
    Prepares the data for the BundleNet algorithm by formatting the input neuronal and behavioral traces.

    Parameters:
        X : np.ndarray
            Raw neuronal traces of shape (t, n), where n is the number of neurons and t is the number of time steps.
        B : np.ndarray
            Raw behavioral traces of shape (t,), representing the behavioral data corresponding to the neuronal
            traces.
        win : int, optional
            Length of the window to feed as input to the algorithm. If win > 1, a slice of the time series is used
            as input.

    Returns:
        X_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win is the length of each window,
            and n is the number of neurons.
        B_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents
            the behavioral data corresponding to the next time step in the paired neuronal traces.

    """
    if win < 1 or len(X) <= win:
        raise ValueError("X must contain more samples than the positive window length.")
    if len(X) != len(B):
        raise ValueError("X and B must have the same number of samples.")
    # Read-only views share the original recording, including overlapping pairs.
    windows = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=win + 1, axis=0
    ).transpose(0, 2, 1)
    X_paired = np.lib.stride_tricks.sliding_window_view(
        windows, window_shape=win, axis=1
    ).transpose(0, 1, 3, 2)
    B_1 = B[win:]

    return X_paired, B_1


def timeseries_train_test_split(X_paired, B_1):
    """
    Perform a train-test split for time series data without shuffling, based on a specific fold.

    Parameters:
        X_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win-1 is the length of each window excluding the last time
            step,and n is the number of neurons.
        B_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents the behavioral
            data corresponding to the next time step in the paired neuronal traces.

    Returns:
        X_train : np.ndarray
            Training set of paired neuronal traces, of shape (m_train, 2, win, n), where m_train is the number of
            paired windows in the training set.
        X_test : np.ndarray
            Test set of paired neuronal traces, of shape (m_test, 2, win, n), where m_test is the number of paired
            windows in the test set.
        B_train_1 : np.ndarray
            Behavioral traces corresponding to the next time step in the training set, of shape (m_train,).
        B_test_1 : np.ndarray
            Behavioral traces corresponding to the next time step in the test set, of shape (m_test,).

    """
    # Train test split
    kf = KFold(n_splits=7)
    for i, (train_index, test_index) in enumerate(kf.split(X_paired)):
        if i == 4:
            # Train test split based on a fold
            X_train, X_test = X_paired[train_index], X_paired[test_index]
            B_train_1, B_test_1 = B_1[train_index], B_1[test_index]

            return X_train, X_test, B_train_1, B_test_1


#################################################################
##### BunDLe Net --- Architecture and functions for training ####
#################################################################


class BunDLeNet(Model):
    """Behaviour and Dynamical Learning Network (BunDLeNet) model.

    This model represents the BunDLe Net's architecture for deep learning and is based on the commutativity
    diagrams. The resulting model is dynamically consistent (DC) and behaviourally consistent (BC) as per
    the notion described in the paper.

    Args:
        latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, latent_dim):
        super(BunDLeNet, self).__init__()
        self.latent_dim = latent_dim
        self.tau = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(50, activation="relu"),
                layers.Dense(30, activation="relu"),
                layers.Dense(25, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Dense(latent_dim, activation="linear"),
                layers.GaussianNoise(0.05),
            ]
        )
        self.T_Y = tf.keras.Sequential(
            [
                layers.Dense(latent_dim, activation="linear"),
            ]
        )
        self.predictor = tf.keras.Sequential([layers.Dense(5, activation="linear")])

    def call(self, X):
        # Upper arm of commutativity diagram
        Yt1_upper = self.tau(X[:, 1])
        Bt1_upper = self.predictor(Yt1_upper)

        # Lower arm of commutativity diagram
        Yt_lower = self.tau(X[:, 0])
        Yt1_lower = Yt_lower + self.T_Y(Yt_lower)

        return Yt1_upper, Yt1_lower, Bt1_upper


####################################################
#################### Evaluation ####################
####################################################

flat_partial = lambda x: x.reshape(x.shape[0], -1)


def r2_single(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return 1 - mse(y_pred, y_true) / tf.math.reduce_variance(y_true)


def r2(Y_true, Y_pred):
    r2_list = []
    for i in range(Y_true.shape[-1]):
        R2 = r2_single(Y_true[:, i], Y_pred[:, i])
        r2_list.append(R2)
    r2_list = tf.stack(r2_list)
    return tf.math.reduce_mean(r2_list)


def hits_at_rank(rank, Y_test, Y_pred):
    nbrs = NearestNeighbors(n_neighbors=rank, algorithm="ball_tree").fit(Y_test)
    distances, indices = nbrs.kneighbors(Y_test)
    return np.mean(np.linalg.norm(Y_pred - Y_test, axis=1) < distances[:, -1])



# def exclude_neurons(self, exclude_neurons):
#     """
#     Excludes specified neurons from the database.

#     Args:
#         exclude_neurons (list): List of neuron names to exclude.

#     Returns:
#         None

#     """
#     neuron_names = self.neuron_names
#     mask = np.zeros_like(self.neuron_names, dtype='bool')
#     for exclude_neuron in exclude_neurons:
#         mask = np.logical_or(mask, neuron_names==exclude_neuron)
#     mask = ~mask
#     self.neuron_traces = self.neuron_traces[mask]
#     #self.derivative_traces = self.derivative_traces[mask]
#     self.neuron_names = self.neuron_names[mask]

# def _only_identified_neurons(self):
#     mask = np.logical_not([x.isnumeric() for x in self.neuron_names])
#     self.neuron_traces = self.neuron_traces[mask]
#     #self.derivative_traces = self.derivative_traces[mask]
#     self.neuron_names = self.neuron_names[mask]

# def categorise_neurons(self):
#     self._only_identified_neurons()
#     neuron_list = mat73.loadmat('data/raw/Order279.mat')['Order279']
#     neuron_category = mat73.loadmat('data/raw/ClassIDs_279.mat')['ClassIDs_279']
#     category_dict = {neuron: int(category) for neuron, category in zip(neuron_list, neuron_category)}

#     mask = np.array([category_dict[neuron] for neuron in self.neuron_names])
#     mask_s = mask == 1
#     mask_i = mask == 2
#     mask_m = mask == 3

#     self.neuron_names_s = self.neuron_names[mask_s]
#     self.neuron_names_i = self.neuron_names[mask_i]
#     self.neuron_names_m = self.neuron_names[mask_m]

#     self.neuron_traces_s = self.neuron_traces[mask_s]
#     self.neuron_traces_i = self.neuron_traces[mask_i]
#     self.neuron_traces_m = self.neuron_traces[mask_m]

#     return mask

# flat_partial = lambda x: x.reshape(x.shape[0],-1)

############################################
####### Data preprocessing functions #######
############################################

# def bandpass(traces, f_l, f_h, sampling_freq):
#     """
#     Apply a bandpass filter to the input traces.

#     Parameters:
#         traces (np.ndarray): Input traces to be filtered.
#         f_l (float): Lower cutoff frequency in Hz.
#         f_h (float): Upper cutoff frequency in Hz.
#         sampling_freq (float): Sampling frequency in Hz.

#     Returns:
#         filtered (np.ndarray): Filtered traces.

#     """
#     cut_off_h = f_h*sampling_freq/2 ## in units of sampling_freq/2
#     cut_off_l= f_l*sampling_freq/2 ## in units of sampling_freq/2
#     #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.

#     sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
#     ### filtering the traces forward and backwards
#     filtered = signal.sosfilt(sos, traces)
#     filtered = np.flip(filtered, axis=1)
#     filtered = signal.sosfilt(sos, filtered)
#     filtered = np.flip(filtered, axis=1)
#     return filtered

# def preprocess_data(X, fps):
#     """Preprocess the input data by applying bandpass filtering.

#     Args:
#         X: Input data.
#         fps (float): Frames per second.

#     Returns:
#         numpy.ndarray: Preprocessed data after bandpass filtering.
#     """
#     time = 1 / fps * np.arange(0, X.shape[0])
#     filtered = (X.T, f_l=1e-10, f_h=0.05, sampling_freq=fps).T

#     return time, filtered
