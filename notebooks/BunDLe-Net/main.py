import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from functions import Database, prep_data, BunDLeNet, train_model, plotting_neuronal_behavioural, plot_latent_timeseries, plot_phase_space

sys.path.append(r'../')

### Load Data (excluding behavioural neurons) and plot

data = Database()

X_full = data.neuron_traces.T.astype(np.float32, copy=False)
del data.neuron_traces

training_samples = min(len(X_full), int(60 * data.fps))
X = X_full[:training_samples]
B = data.B[:training_samples]
state_names = data.state_names
plotting_neuronal_behavioural(X, B, state_names=state_names)

### Preprocess and prepare data for BundLe Net
time = data.fps
X_, B_ = prep_data(X, B, win=15)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
# Erzeugt zunächst nur eine speichersparende Sicht auf die Daten.

_ = model(tf.convert_to_tensor(X_[:1], dtype=tf.float32))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_array = train_model(
	X_,
	B_,
	model,
	optimizer,
	gamma=0.9, 
	n_epochs=10,
	pca_init=False,
	best_of_5_init=False
)

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
	plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0], training=False).numpy()

# Apply the trained model to the complete recording in bounded batches.
X_windows = sliding_window_view(
	X_full,
	window_shape=15,
	axis=0,
).transpose(0, 2, 1)[:-1]
B_full = data.B[15:]

Y0_full = np.empty((len(X_windows), model.latent_dim), dtype=np.float32)
batch_size = 4096

for batch_number, start in enumerate(range(0, len(X_windows), batch_size), start=1):
	stop = min(start + batch_size, len(X_windows))
	batch = tf.convert_to_tensor(X_windows[start:stop], dtype=tf.float32)
	Y0_full[start:stop] = model.tau(batch, training=False).numpy()

	if batch_number == 1 or batch_number % 100 == 0 or stop == len(X_windows):
		print(f"Full-data inference: {stop}/{len(X_windows)} samples", flush=True)

print("Latent shape:", Y0_full.shape)
print("Labels shape:", B_full.shape)

algorithm = 'BunDLeNet'
# ### Save the weights (Uncomment to save and load for for later use)
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plot_latent_timeseries(Y0_, B_, state_names)
plot_step = max(1, len(Y0_full) // 5000)
plot_phase_space(
	Y0_full[::plot_step],
	B_full[::plot_step],
	state_names=state_names,
)
# rotating_plot(Y0_, B_,filename='figures/rotation_'+ algorithm + '_worm_' +'.gif', state_names=state_names, legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca, B_, state_names)

### Recurrence plot analysis of embedding
# pd_Y = np.linalg.norm(Y0_[:, np.newaxis] - Y0_, axis=-1) < 0.8
# plt.matshow(pd_Y, cmap='Greys')
# plot_latent_timeseries(Y0_, B_, state_names)
# plt.show()
