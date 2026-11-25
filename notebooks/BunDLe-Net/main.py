import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from functions import (
    BunDLeNet
)
from bundle_trainer import BunDLeTrainer
from plot import (
    plot_loss,
    plot_latent_timeseries,
    plot_phase_space,
    plot_mean_trajectories,
    plot_recurrence,
    plotting_neuronal_behavioural,
)
from processed_data_storage import ProcessedDataStroage
from preprocess import Preprocessor

state_names = [
    "keine Kategorie",
    "Gesicht",
    "Tier",
    "Ort",
    "Objekt",
    "Catch",
]

n_epochs = 5
subjects_range = range(1, 2)

processed_data_storage = ProcessedDataStroage()
preprocessor = Preprocessor()


processed_data_storage.preprocess_and_save_subjects_from_range(subjects_range)
X_train, X_test, labels_train, labels_test, category_per_epoch_all_subs_train, category_per_epoch_all_subs_test = processed_data_storage.read_subjects_from_range(subjects_range)

number_of_epochs, number_of_channels, number_of_time_points = X_train.shape
print(f"X_train shape: {X_train.shape}")
print(f"Number of epochs: {number_of_epochs}")
print(f"Number of channels: {number_of_channels}")

# Alle Fensterpaare der Trainingsdaten (darauf wird trainiert)
epoch_numbers, start_time_points = preprocessor.list_all_window_pairs(
    number_of_epochs, number_of_time_points, preprocessor.WINDOW_LENGTH
)

# Alle Fensterpaare der Testdaten (werden nur ausgewertet, nie trainiert)
number_of_test_epochs = X_test.shape[0]
number_of_test_time_points = X_test.shape[2]
test_epoch_numbers, test_start_time_points = preprocessor.list_all_window_pairs(
    number_of_test_epochs, number_of_test_time_points, preprocessor.WINDOW_LENGTH
)

# Feste Zufallsauswahl an Trainingsfenstern, genauso viele wie Testfenster.
# Auf dieser Auswahl messen wir nach jeder Trainingsepoche Loss und Genauigkeit
# auf genau dieselbe Art wie auf den Testdaten.
random_generator = np.random.default_rng(11)
train_subset_indices = random_generator.choice(
    len(epoch_numbers), size=len(test_epoch_numbers), replace=False
)
train_subset_epoch_numbers = epoch_numbers[train_subset_indices]
train_subset_start_time_points = start_time_points[train_subset_indices]

print(f"Trainingsfenster: {len(epoch_numbers)}, davon ausgewertet: {len(train_subset_epoch_numbers)}")
print(f"Testfenster: {len(test_epoch_numbers)}")

# Zufallswert (reines Raten) als Vergleichslinie.
# Das Ziel eines Fensterpaares ist das Label am Index start + WINDOW_LENGTH,
# also frühestens am Index WINDOW_LENGTH. Davor liegende Labels kommen als Ziel nie vor.
target_labels_that_can_occur = labels_train[:, preprocessor.WINDOW_LENGTH:]
target_categories = np.unique(target_labels_that_can_occur)
number_of_target_categories = len(target_categories)
chance_loss = np.log(number_of_target_categories)
chance_accuracy = 1 / number_of_target_categories
print(f"Kategorien, die als Ziel vorkommen: {target_categories}")
print(f"Raten ergibt Loss ln({number_of_target_categories}) = {chance_loss:.3f} und Genauigkeit {chance_accuracy:.2f}")

start_time_in_seconds = -0.2                      
sampling_rate_in_hertz = 100                
times = start_time_in_seconds + np.arange(number_of_time_points) / sampling_rate_in_hertz

plotting_neuronal_behavioural(
    X_train, labels_train, state_names=state_names, max_epochs=20
)
plt.close("all")

model = BunDLeNet(3)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
trainer = BunDLeTrainer(model, optimizer)
random = np.random.default_rng(11)

bundle_trainer = BunDLeTrainer(model, optimizer)
history_per_epoch, all_losses = bundle_trainer.train_model(X_train, labels_train, X_test, labels_test, epoch_numbers, start_time_points, random, n_epochs)

plot_loss(history_per_epoch, all_losses, chance_loss, number_of_target_categories, chance_accuracy, preprocessor.GAMMA)

### Projecting into latent space
# tau wurde auf Fenstern der Form (WINDOW_LENGTH, Kanäle) trainiert, die innerhalb
# einer Epoche entlang der Zeitachse liegen (siehe Preprocessor.build_training_batch).
# Genauso projizieren wir hier: für jede Epoche jedes Fenster, das bei s beginnt.
# Ergebnis pro Epoche: (Anzahl Fenster, latent_dim), Label = Label am letzten Zeitpunkt des Fensters.
def project_epochs(model, eeg_data, window_length, epochs_per_batch=64):
    number_of_epochs, number_of_channels, number_of_time_points = eeg_data.shape
    number_of_windows = number_of_time_points - window_length + 1
    Y = np.empty((number_of_epochs, number_of_windows, model.latent_dim), dtype=np.float32)

    for start in range(0, number_of_epochs, epochs_per_batch):
        stop = min(start + epochs_per_batch, number_of_epochs)
        # (E, Kanäle, Zeit) -> (E, Kanäle, Fenster, Fensterlänge)
        windows = sliding_window_view(eeg_data[start:stop], window_length, axis=2)
        # -> (E, Fenster, Fensterlänge, Kanäle) -> (E*Fenster, Fensterlänge, Kanäle)
        windows = windows.transpose(0, 2, 3, 1).reshape(-1, window_length, number_of_channels)
        latent = model.tau(tf.convert_to_tensor(windows, dtype=tf.float32), training=False).numpy()
        Y[start:stop] = latent.reshape(stop - start, number_of_windows, model.latent_dim)
    return Y


window_length = preprocessor.WINDOW_LENGTH
Y_train = project_epochs(model, X_train, window_length)
Y_test = project_epochs(model, X_test, window_length)
B_train = labels_train[:, window_length - 1:].astype(int)
B_test = labels_test[:, window_length - 1:].astype(int)
print("Latent shape (Training):", Y_train.shape, "Labels:", B_train.shape)
print("Latent shape (Test):", Y_test.shape, "Labels:", B_test.shape)

algorithm = "BunDLeNet"
# ### Save the weights (Uncomment to save and load for for later use)
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plot_latent_timeseries(Y_train, B_train, state_names, max_epochs=20, title="Latenter Raum (Training)")
plot_phase_space(Y_train, B_train, state_names=state_names, max_epochs=200, title="Trajektorien (Training)")
plot_phase_space(Y_test, B_test, state_names=state_names, max_epochs=200, title="Trajektorien (Test)")
plot_mean_trajectories(Y_train, category_per_epoch_all_subs_train, state_names, title="Mittlere Trajektorie pro Kategorie (Training)")
plot_mean_trajectories(Y_test, category_per_epoch_all_subs_test, state_names, title="Mittlere Trajektorie pro Kategorie (Test)")
# rotating_plot(Y_train, B_train, filename='figures/rotation_' + algorithm + '.gif', state_names=state_names, legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y_train.reshape(-1, model.latent_dim)).reshape(Y_train.shape)
print("Erklärte Varianz der PCA-Komponenten:", np.round(pca.explained_variance_ratio_, 3))
plot_latent_timeseries(Y_pca, B_train, state_names, max_epochs=20, title="PCA des latenten Raums (Training)")

### Recurrence plot analysis of embedding
# Nur eine Stichprobe: pro Kategorie 10 Epochen, jeder 4. Zeitpunkt
# -> 4 * 10 * 27 = 1080 Zustände, Distanzmatrix ca. 9 MB statt (Epochen*Zeit)^2.
recurrence_train, category_recurrence_train, epsilon_train = plot_recurrence(
    Y_train, B_train, category_per_epoch_all_subs_train, state_names,
    epochs_per_category=10, time_step=4, recurrence_rate=0.1,
    title="Rekurrenzanalyse des latenten Raums (Training)",
)
recurrence_test, category_recurrence_test, epsilon_test = plot_recurrence(
    Y_test, B_test, category_per_epoch_all_subs_test, state_names,
    epochs_per_category=10, time_step=4, recurrence_rate=0.1,
    title="Rekurrenzanalyse des latenten Raums (Test)",
)
