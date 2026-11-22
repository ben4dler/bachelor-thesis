import tensorflow as tf
from bundle_loss import bccdcc_loss
from preprocess import Preprocessor
import numpy as np

class BunDLeTrainer:
    """Trainer for the BunDLe Net model.

    This class handles the training process for the BunDLe Net model.

    Args:
        model: Instance of the BunDLeNet class.
        optimizer: Optimizer for model training.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.behaviour_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.preprocessor = Preprocessor()

    def train_model(self,
        X_train,
        labels_train,
        X_test,
        labels_test,
        epoch_numbers,
        start_time_points,
        random,
        n_epochs,
    ):

        all_losses = []           # pro Batch: [Markov, Behavior, Total]
        history_per_epoch = []    # pro Trainingsepoche: [Train-Loss, Train-Genauigkeit, Test-Loss, Test-Genauigkeit]

        for training_epoch in range(n_epochs):
            shuffled_order = random.permutation(len(epoch_numbers))
            for batch_start in range(0, len(shuffled_order), self.preprocessor.BATCH_SIZE):
                selected_pairs = shuffled_order[batch_start : batch_start + self.preprocessor.BATCH_SIZE]

                window_pairs, target_labels = self.preprocessor.build_training_batch(
                    X_train,
                    labels_train,
                    epoch_numbers[selected_pairs],
                    start_time_points[selected_pairs],
                    self.preprocessor.WINDOW_LENGTH,
                )

                markov_loss, behaviour_loss, total_loss = self._train_step(
                    tf.convert_to_tensor(window_pairs, dtype=tf.float32),
                    tf.convert_to_tensor(target_labels, dtype=tf.int64),
                    gamma=self.preprocessor.GAMMA,
                )
                all_losses.append([float(markov_loss), float(behaviour_loss), float(total_loss)])

            # Nach jeder Trainingsepoche: Training und Test auf dieselbe Art auswerten

            train_loss, train_accuracy, test_loss, test_accuracy = self.evaluate(X_test.shape[0], X_test.shape[2], 
                          random, epoch_numbers, start_time_points, self.model, X_train, labels_train, X_test, labels_test)
            
            history_per_epoch.append([train_loss, train_accuracy, test_loss, test_accuracy])

            print(
                f"Trainingsepoche {training_epoch + 1}/{n_epochs}: "
                f"Loss Training {train_loss:.3f} / Test {test_loss:.3f}, "
                f"Genauigkeit Training {train_accuracy:.3f} / Test {test_accuracy:.3f}"
            )
        return history_per_epoch, all_losses

    @tf.function
    def _train_step(self, x_train, b_train_1, gamma):
        with tf.GradientTape() as tape:
            yt1_upper, yt1_lower, bt1_upper = self.model(x_train, training=True)
            DCC_loss, behaviour_loss, total_loss = bccdcc_loss(
                yt1_upper, yt1_lower, bt1_upper, b_train_1, gamma
            )
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return DCC_loss, behaviour_loss, total_loss

    def evaluate(self, number_of_test_epochs, 
                number_of_test_time_points, 
                random, 
                epoch_numbers, 
                start_time_points, 
                model, 
                X_train, labels_train, 
                X_test, labels_test ):
        
        test_epoch_numbers, test_start_time_points = self.preprocessor.list_all_window_pairs(
            number_of_test_epochs, number_of_test_time_points, self.preprocessor.WINDOW_LENGTH
        )

        train_subset_indices = random.choice(
        len(epoch_numbers), size=len(test_epoch_numbers), replace=False
        )

        train_subset_epoch_numbers = epoch_numbers[train_subset_indices]
        train_subset_start_time_points = start_time_points[train_subset_indices]

        train_loss, train_accuracy = self._evaluate_windows(
            model, X_train, labels_train, train_subset_epoch_numbers, train_subset_start_time_points
        )

        test_loss, test_accuracy = self._evaluate_windows(
            model, X_test, labels_test, test_epoch_numbers, test_start_time_points
        )

        return train_loss, train_accuracy, test_loss, test_accuracy

    def _evaluate_windows(self, model, eeg_data, labels, epoch_numbers, start_time_points):
        """Berechnet Behavior-Loss und Genauigkeit auf den gegebenen Fensterpaaren.

        Das Modell wird dabei nicht verändert (kein Training).
        """
        sum_of_losses = 0.0
        number_of_correct_predictions = 0
        number_of_windows = len(epoch_numbers)

        for batch_start in range(0, number_of_windows, self.preprocessor.BATCH_SIZE):
            batch_end = batch_start + self.preprocessor.BATCH_SIZE

            window_pairs, target_labels = self.preprocessor.build_training_batch(
                eeg_data,
                labels,
                epoch_numbers[batch_start:batch_end],
                start_time_points[batch_start:batch_end],
                self.preprocessor.WINDOW_LENGTH,
            )
            target_labels = target_labels.astype(np.int64)

            # training=False schaltet die GaussianNoise-Schicht aus
            _, _, predicted_scores = model(
                tf.convert_to_tensor(window_pairs, dtype=tf.float32), training=False
            )

            # Der Loss ist ein Mittelwert über den Batch. Der letzte Batch ist kleiner,
            # deshalb gewichten wir mit der Batchgröße und teilen am Ende durch alle Fenster.
            batch_loss = float(self.behaviour_loss_function(target_labels, predicted_scores))
            number_of_windows_in_batch = len(target_labels)
            sum_of_losses = sum_of_losses + batch_loss * number_of_windows_in_batch

            # Vorhergesagte Kategorie = Ausgang mit dem höchsten Wert
            predicted_categories = np.argmax(predicted_scores.numpy(), axis=1)
            for window_number in range(number_of_windows_in_batch):
                if predicted_categories[window_number] == target_labels[window_number]:
                    number_of_correct_predictions = number_of_correct_predictions + 1

        mean_loss = sum_of_losses / number_of_windows
        accuracy = number_of_correct_predictions / number_of_windows
        return mean_loss, accuracy
"""
    def _pca_initialisation(self, X_, tau, latent_dim):
        
        Initialises BunDLe Net's tau such that its output is the PCA of the input traces.
        PCA initialisation may make the embeddings more reproduceable across runs.
        This function is called within the train_model() function and saves the learned tau weights
        in a .h5 file in the same repository.

        Parameters:
            X_ (np.ndarray): Input data.
            tau (object): BunDLe Net tau (tf sequential layer).
            latent_dim (int): Dimension of the latent space.


        
        ### Performing PCA on the time slice
        X0_ = X_[:, 0, :, :]
        X_pca = X_.reshape(X_.shape[0], 2, 1, -1)[:, 0, 0, :]
        pca = PCA(n_components=latent_dim, whiten=True)
        pca.fit(X_pca)
        Y0_ = pca.transform(X_pca)

        ### Training tau to reproduce the PCA
        class PCA_encoder(Model):
            def __init__(self, latent_dim):
                super(PCA_encoder, self).__init__()
                self.latent_dim = latent_dim
                self.encoder = tau

            def call(self, x):
                encoded = self.encoder(x)
                return encoded

        pcaencoder = PCA_encoder(latent_dim=latent_dim)
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
        pcaencoder.compile(optimizer=opt, loss="mse", metrics=["mse"])
        history = pcaencoder.fit(
            X0_,
            Y0_,
            epochs=10,
            batch_size=100,
            verbose=0,
        )
        Y0_pred = pcaencoder(X0_).numpy()
        ### Saving weights of this model
        pcaencoder.encoder.save_weights("data/generated/tau_pca_weights.h5")


    def _best_of_5_runs(self, X_train, B_train_1, model, optimizer, gamma):
        
        Initialises BunDLe net with the best of 5 runs

        Performs 200 epochs of training for 5 random model initialisations
        and picks the model with the lowest loss
        
        model_loss = []
        for i in range(5):
            model_ = keras.models.clone_model(model)
            model_.build(input_shape=X_train.shape)
            loss_array = self.train_model(
                X_train,
                B_train_1,
                model_,
                optimizer,
                gamma=gamma,
                n_epochs=200,
                pca_init=False,
                best_of_5_init=False,
            )
            model_.save_weights("data/generated/best_of_5_runs_models/model_" + str(i))
            model_loss.append(loss_array[-1, 2])

        for n, i in enumerate(model_loss):
            print("model:", n, "loss:", i)

        ### Load model with least loss
        model.load_weights(
            "data/generated/best_of_5_runs_models/model_" + str(np.argmin(model_loss))
        )
        return model
"""