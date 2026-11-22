import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from functions import (
    Database,
    BunDLeNet
)
from bundle_trainer import BunDLeTrainer
from plot import plot_loss, plot_latent_timeseries, plot_phase_space, plot_ps_, plotting_neuronal_behavioural
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

n_epochs = 20
subjects_range = range(1, 8)

processed_data_storage = ProcessedDataStroage()
preprocessor = Preprocessor()


# processed_data_storage.preprocess_and_save_subjects_from_range(subjects_range)
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

preview_step = max(1, (len(X_train) + 4999) // 5000)

"""
plotting_neuronal_behavioural(
    X_train[::preview_step], labels_train[::preview_step], state_names=state_names
)
plt.close("all")
"""
### Preprocess and prepare data for BundLe Net

# Modell hier neu anlegen: Wenn man nur diese Zelle erneut ausführt,
# startet das Training von vorne statt mit dem alten Modell weiterzumachen.
model = BunDLeNet(3)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
trainer = BunDLeTrainer(model, optimizer)
random = np.random.default_rng(11)

bundle_trainer = BunDLeTrainer(model, optimizer)
history_per_epoch, all_losses = bundle_trainer.train_model(X_train, labels_train, X_test, labels_test, epoch_numbers, start_time_points, random, n_epochs)

plot_loss(history_per_epoch, all_losses, chance_loss, number_of_target_categories, chance_accuracy, preprocessor.GAMMA)

### Projecting into latent space
# Project the complete recording below in bounded batches.
"""
# Apply the trained model to the complete recording in bounded batches.
X_windows = sliding_window_view(
    X_train,
    window_shape=15,
    axis=0,
).transpose(
    0, 2, 1
)[:-1]
B_full = labels_train[15:]

Y0_full = np.empty((len(X_windows), model.latent_dim), dtype=np.float32)
batch_size = 4096

for batch_number, start in enumerate(range(0, len(X_windows), batch_size), start=1):
    stop = min(start + batch_size, len(X_windows))
    batch = tf.convert_to_tensor(X_windows[start:stop], dtype=tf.float32)
    Y0_full[start:stop] = model.tau(batch, training=False).numpy()

    if batch_number == 1 or batch_number % 100 == 0 or stop == len(X_windows):
        print(f"Full-data inference: {stop}/{len(X_windows)} samples", flush=True)

Y0_ = Y0_full
print("Latent shape:", Y0_full.shape)
print("Labels shape:", B_full.shape)

algorithm = "BunDLeNet"
# ### Save the weights (Uncomment to save and load for for later use)
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plot_step = max(1, len(Y0_full) // 5000)
plot_latent_timeseries(Y0_[::plot_step], B_1[::plot_step], state_names)
plot_phase_space(
    Y0_full[::plot_step],
    B_full[::plot_step],
    state_names=state_names,
)
# rotating_plot(Y0_, B_,filename='figures/rotation_'+ algorithm + '_worm_' +'.gif', state_names=state_names, legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca[::plot_step], B_1[::plot_step], state_names)

### Recurrence plot analysis of embedding
# pd_Y = np.linalg.norm(Y0_[:, np.newaxis] - Y0_, axis=-1) < 0.8
# plt.matshow(pd_Y, cmap='Greys')
# plot_latent_timeseries(Y0_[::plot_step], B_[::plot_step], state_names)
# plt.show()
"""