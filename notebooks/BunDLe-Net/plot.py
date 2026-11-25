import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def category_colors(number_of_categories, light=False):
    """Feste Farbe pro Kategorie, in allen Plots gleich.

    "pastel" und "deep" haben in seaborn dieselbe Farbreihenfolge, so passen
    helle Hintergrundbänder und kräftige Linien zueinander.
    """
    return sns.color_palette("pastel" if light else "deep", number_of_categories)


def _concatenate_epochs(Y, B, max_epochs):
    """(Epochen, Zeit, ...) -> (Epochen*Zeit, ...) für die ersten max_epochs Epochen.

    Gibt zusätzlich die Positionen der Epochengrenzen zurück.
    2D-Eingaben (Zeit, ...) werden unverändert durchgereicht.
    """
    Y = np.asarray(Y)
    B = np.asarray(B)
    if B.ndim == 1:
        return Y, B.astype(int), []
    Y = Y[:max_epochs]
    B = B[:max_epochs]
    number_of_epochs, number_of_time_points = B.shape
    boundaries = np.arange(1, number_of_epochs) * number_of_time_points - 0.5
    return (
        Y.reshape(number_of_epochs * number_of_time_points, *Y.shape[2:]),
        B.reshape(-1).astype(int),
        boundaries,
    )


def _category_legend_handles(state_names, light=False):
    colors = category_colors(len(state_names), light=light)
    return [Patch(color=color, label=name) for color, name in zip(colors, state_names)]

def plot_loss(history_per_epoch, 
              all_losses, 
              chance_loss, 
              number_of_target_categories,
              chance_accuracy,
              gamma):
    # 1) Training und Test pro Trainingsepoche
    history = np.array(history_per_epoch)    # Form (Anzahl Trainingsepochen, 4)
    epoch_axis = np.arange(1, len(history) + 1)

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))

    loss_axis.plot(epoch_axis, history[:, 0], label="Training")
    loss_axis.plot(epoch_axis, history[:, 2], label="Test")
    loss_axis.axhline(chance_loss, color="gray", linestyle="--", label=f"Raten: ln({number_of_target_categories})")
    loss_axis.set_xlabel("Trainingsepoche")
    loss_axis.set_ylabel(r"$\mathcal{L}_{Behavior}$")
    loss_axis.legend()

    accuracy_axis.plot(epoch_axis, history[:, 1], label="Training")
    accuracy_axis.plot(epoch_axis, history[:, 3], label="Test")
    accuracy_axis.axhline(chance_accuracy, color="gray", linestyle="--", label=f"Raten: 1/{number_of_target_categories}")
    accuracy_axis.set_xlabel("Trainingsepoche")
    accuracy_axis.set_ylabel("Genauigkeit")
    accuracy_axis.legend()

    plt.tight_layout()
    plt.show()

    # 2) Verlauf pro Batch (mit Rauschen der GaussianNoise-Schicht)
    # total = GAMMA * Markov + (1 - GAMMA) * Behavior. Bei GAMMA = 0 ist der Markov-Anteil
    # immer 0 (auf einer log-Achse nicht darstellbar) und Total ist gleich Behavior.
    # Deshalb zeichnen wir nur die Kurven, die etwas Eigenes zeigen.
    loss_history = np.array(all_losses)      # Form (Anzahl Batches, 3)

    plt.figure()
    if gamma > 0:
        plt.semilogy(loss_history[:, 0], label=r"$\mathcal{L}_{Markov}$ (mal GAMMA)")
    if gamma < 1:
        plt.semilogy(loss_history[:, 1], label=r"$\mathcal{L}_{Behavior}$ (mal 1 - GAMMA)")
    if 0 < gamma < 1:
        plt.semilogy(loss_history[:, 2], label=r"Total loss $\mathcal{L}$")
    plt.xlabel("Batch")
    plt.ylabel("Loss (logarithmisch)")
    plt.legend()
    plt.show()

def plotting_neuronal_behavioural(X, B, state_names=None, vmin=None, vmax=None, max_epochs=20):
    """Zeigt EEG-Aktivität und Kategorie-Labels als zwei übereinanderliegende Heatmaps.

    X: (Epochen, Kanäle, Zeitpunkte) oder (Zeitpunkte, Kanäle)
    B: (Epochen, Zeitpunkte) oder (Zeitpunkte,)
    Bei 3D-Eingabe werden die ersten `max_epochs` Epochen hintereinander gehängt
    und die Epochengrenzen als senkrechte Linien eingezeichnet.
    """
    X = np.asarray(X)
    B = np.asarray(B)
    epoch_boundaries = []

    if X.ndim == 3:
        X = X[:max_epochs]
        B = B[:max_epochs]
        number_of_epochs, number_of_channels, number_of_time_points = X.shape
        # (Epochen, Kanäle, Zeit) -> (Epochen, Zeit, Kanäle) -> (Epochen*Zeit, Kanäle)
        X = X.transpose(0, 2, 1).reshape(-1, number_of_channels)
        B = B.reshape(-1)
        epoch_boundaries = np.arange(1, number_of_epochs) * number_of_time_points - 0.5

    B = B.astype(int)

    # Farbskala robust aus den Daten bestimmen (skalierte EEG-Daten liegen um 0)
    if vmin is None or vmax is None:
        limit = np.percentile(np.abs(X), 99)
        vmin, vmax = -limit, limit

    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                            gridspec_kw={"height_ratios": [4, 1]})

    im0 = axs[0].imshow(X.T, aspect="auto", vmin=vmin, vmax=vmax,
                        cmap="RdBu_r", interpolation="none")
    plt.colorbar(im0, ax=axs[0])
    axs[0].set_ylabel("Kanal")

    # Feste Farbzuordnung über alle Kategorien, auch wenn nicht alle vorkommen
    if state_names:
        number_of_categories = len(state_names)
    else:
        number_of_categories = int(B.max()) + 1
    cmap = ListedColormap(category_colors(number_of_categories, light=True))
    im1 = axs[1].imshow(
        B[np.newaxis, :], cmap=cmap, vmin=-0.5, vmax=number_of_categories - 0.5,
        aspect="auto", interpolation="none",
    )
    # Unsichtbare Farbleiste, damit beide Achsen gleich breit bleiben
    plt.colorbar(im1, ax=axs[1]).ax.set_visible(False)
    if state_names:
        axs[1].legend(handles=_category_legend_handles(state_names, light=True), loc="upper center", bbox_to_anchor=(0.5, -0.6),
                      ncol=len(state_names), frameon=False)
    axs[1].set_ylabel("Kategorie")
    axs[1].set_yticks([])
    axs[1].set_xlabel("Zeitpunkt (Epochen aneinandergereiht)")

    for boundary in epoch_boundaries:
        for ax in axs:
            ax.axvline(boundary, color="k", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_latent_timeseries(Y, B, state_names, max_epochs=20, title=None):
    """Latente Koordinaten über die Zeit, hinterlegt mit der Kategorie als Farbband.

    Y: (Epochen, Zeit, latent) oder (Zeit, latent)
    B: (Epochen, Zeit) oder (Zeit,)
    """
    Y, B, boundaries = _concatenate_epochs(Y, B, max_epochs)
    number_of_categories = len(state_names)
    y_limit = 1.1 * np.max(np.abs(Y))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(
        B[np.newaxis, :],
        cmap=ListedColormap(category_colors(number_of_categories, light=True)),
        vmin=-0.5, vmax=number_of_categories - 0.5,
        aspect="auto", interpolation="none",
        extent=(-0.5, len(B) - 0.5, -y_limit, y_limit),
    )
    # Farbige Hintergrundbänder -> Linien in Grautönen, unterschieden über Linienstil
    line_styles = [("black", "-"), ("dimgray", "--"), ("gray", ":")]
    for dimension in range(Y.shape[1]):
        color, style = line_styles[dimension % len(line_styles)]
        ax.plot(Y[:, dimension], color=color, linestyle=style,
                linewidth=1.1, label=f"$y_{dimension + 1}$")
    for boundary in boundaries:
        ax.axvline(boundary, color="k", linewidth=0.5, alpha=0.4)

    ax.set_xlim(-0.5, len(B) - 0.5)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Zeitpunkt (Epochen aneinandergereiht)")
    ax.set_ylabel("latente Koordinate")
    if title:
        ax.set_title(title)
    line_legend = ax.legend(loc="upper right", fontsize=8)
    ax.add_artist(line_legend)
    ax.legend(handles=_category_legend_handles(state_names, light=True),
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=number_of_categories, frameon=False)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_ps_(fig, ax, Y, B, state_names, show_points=False, legend=True,
             colors=None, max_epochs=200, linewidth=0.6, alpha=0.5):
    """Zeichnet Trajektorien im 3D-latenten Raum, Segmente gefärbt nach Kategorie.

    Y: (Epochen, Zeit, 3) oder (Zeit, 3); B passend dazu.
    Bei 3D-Eingabe ist jede Epoche eine eigene Trajektorie, damit keine
    Sprünge zwischen dem Ende einer Epoche und dem Anfang der nächsten entstehen.
    """
    if colors is None:
        colors = category_colors(len(state_names))
    colors = np.asarray(colors)

    Y = np.asarray(Y)
    B = np.asarray(B).astype(int)
    if Y.ndim == 2:
        Y, B = Y[np.newaxis], B[np.newaxis]
    Y, B = Y[:max_epochs], B[:max_epochs]

    # Segment i verbindet Y[t] mit Y[t+1] und bekommt die Farbe der Kategorie bei t+1
    segments = np.stack([Y[:, :-1, :3], Y[:, 1:, :3]], axis=2).reshape(-1, 2, 3)
    segment_colors = colors[B[:, 1:].reshape(-1)]
    ax.add_collection3d(Line3DCollection(
        segments, colors=segment_colors, linewidths=linewidth, alpha=alpha
    ))

    all_points = Y.reshape(-1, Y.shape[-1])
    for set_limit, dimension in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
        set_limit(all_points[:, dimension].min(), all_points[:, dimension].max())
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_zlabel("$y_3$")

    if legend:
        legend_elements = [
            Line2D([0], [0], color=color, lw=4, label=state)
            for color, state in zip(colors, state_names)
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    if show_points:
        ax.scatter(*all_points[:, :3].T, c="k", s=1)
    return fig, ax


def plot_phase_space(Y, B, state_names, show_points=False, legend=True, title=None, **kwargs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    plot_ps_(fig, ax, Y=Y, B=B, state_names=state_names,
             show_points=show_points, legend=legend, **kwargs)
    if title:
        ax.set_title(title)
    plt.show()
    return fig, ax


def plot_mean_trajectories(Y, category_per_epoch, state_names, title=None):
    """Mittlere latente Trajektorie pro Bildkategorie (Mittel über alle Epochen der Kategorie).

    Y: (Epochen, Zeit, 3), category_per_epoch: (Epochen,)
    Der Startpunkt jeder Trajektorie ist mit einem Kreis markiert.
    """
    colors = category_colors(len(state_names))
    category_per_epoch = np.asarray(category_per_epoch).astype(int)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    handles = []
    for category in np.unique(category_per_epoch):
        mean_trajectory = Y[category_per_epoch == category].mean(axis=0)
        ax.plot(*mean_trajectory[:, :3].T, color=colors[category], linewidth=2)
        ax.scatter(*mean_trajectory[0, :3], color=colors[category], s=40)
        handles.append(Line2D([0], [0], color=colors[category], lw=4,
                              label=f"{state_names[category]} (n={np.sum(category_per_epoch == category)})"))
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_zlabel("$y_3$")
    ax.legend(handles=handles, loc="upper left")
    if title:
        ax.set_title(title)
    plt.show()
    return fig, ax


def rotating_plot(Y, B, state_names, show_points=False, legend=True,
                  filename="rotation.gif", **kwargs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    def rotate(angle):
        ax.view_init(azim=angle)

    plot_ps_(fig, ax, Y=Y, B=B, state_names=state_names,
             show_points=show_points, legend=legend, **kwargs)
    rot_animation = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 362, 5), interval=150
    )
    rot_animation.save(filename, dpi=150, writer="pillow")
    plt.show()
    return ax


def select_recurrence_points(Y, B, category_per_epoch, epochs_per_category=10,
                             time_step=4, seed=0):
    """Wählt eine kleine, nach Kategorie sortierte Stichprobe latenter Zustände aus.

    Die volle Rekurrenzmatrix hätte (Epochen*Zeit)^2 Einträge, bei allen Probanden
    viele Terabyte. Deshalb: pro Kategorie nur `epochs_per_category` zufällige Epochen
    und innerhalb jeder Epoche nur jeden `time_step`-ten Zeitpunkt.

    Rückgabe:
        points: (N, latent_dim)
        point_labels: (N,) Label des Zeitpunkts (0 = vor Bildbeginn)
        point_categories: (N,) Bildkategorie der Epoche, aus der der Punkt stammt
        point_epochs: (N,) laufende Nummer der ausgewählten Epoche, aus der der Punkt stammt
        epoch_boundaries: Positionen zwischen den ausgewählten Epochen
    """
    random_generator = np.random.default_rng(seed)
    category_per_epoch = np.asarray(category_per_epoch).astype(int)

    selected_epochs = []
    for category in np.unique(category_per_epoch):
        epochs_of_category = np.flatnonzero(category_per_epoch == category)
        number_to_take = min(epochs_per_category, len(epochs_of_category))
        chosen = random_generator.choice(epochs_of_category, size=number_to_take, replace=False)
        selected_epochs.extend(np.sort(chosen))
    selected_epochs = np.array(selected_epochs)

    Y_selected = np.asarray(Y)[selected_epochs, ::time_step]
    B_selected = np.asarray(B)[selected_epochs, ::time_step].astype(int)
    number_of_epochs, points_per_epoch = B_selected.shape

    points = Y_selected.reshape(number_of_epochs * points_per_epoch, -1)
    point_labels = B_selected.reshape(-1)
    point_categories = np.repeat(category_per_epoch[selected_epochs], points_per_epoch)
    point_epochs = np.repeat(np.arange(number_of_epochs), points_per_epoch)
    epoch_boundaries = np.arange(1, number_of_epochs) * points_per_epoch - 0.5
    return points, point_labels, point_categories, point_epochs, epoch_boundaries


def plot_recurrence(Y, B, category_per_epoch, state_names, epochs_per_category=10,
                    time_step=4, recurrence_rate=0.1, max_points=5000, seed=0, title=None):
    """Rekurrenzplot des latenten Raums auf einer Stichprobe, sortiert nach Kategorie.

    R[i, j] = 1, wenn die Zustände i und j näher als epsilon beieinander liegen.
    epsilon wird so gewählt, dass `recurrence_rate` (z.B. 10 %) aller Paare rekurrent sind;
    ein fester Wert wie 0.8 hängt sonst von der (beliebigen) Skala des latenten Raums ab.

    Links: Rekurrenzmatrix mit Bildkategorie und Zeitpunkt-Label als Farbleisten.
    Rechts: mittlere Rekurrenzrate zwischen den Kategorien. Ist die Diagonale deutlich
    höher als der Rest, liegen Zustände derselben Kategorie näher beieinander.
    Gezählt werden nur Paare aus *verschiedenen* Epochen (Paare aus derselben Epoche
    sind wegen der zeitlichen Glätte fast immer rekurrent und kämen nur auf der
    Diagonale vor) und nur Zustände nach Bildbeginn (Label != 0), da vorher keine
    Kategorie-Information im EEG stecken kann.
    """
    from scipy.spatial.distance import cdist
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    points, point_labels, point_categories, point_epochs, epoch_boundaries = select_recurrence_points(
        Y, B, category_per_epoch, epochs_per_category, time_step, seed
    )
    number_of_points = len(points)
    memory_in_megabytes = number_of_points ** 2 * 8 / 1e6
    print(f"Rekurrenzplot: {number_of_points} Punkte, Distanzmatrix ca. {memory_in_megabytes:.0f} MB")
    if number_of_points > max_points:
        raise ValueError(
            f"{number_of_points} Punkte > max_points={max_points} "
            f"({memory_in_megabytes / 1e3:.1f} GB). epochs_per_category verkleinern "
            f"oder time_step vergrößern."
        )

    distances = cdist(points, points)
    upper_triangle = distances[np.triu_indices(number_of_points, k=1)]
    epsilon = np.quantile(upper_triangle, recurrence_rate)
    recurrence = distances < epsilon

    # Mittlere Rekurrenz zwischen den Kategorien:
    # nur Paare aus verschiedenen Epochen und nur Zustände nach Bildbeginn
    categories = np.unique(point_categories)
    after_onset = point_labels != 0
    different_epoch = point_epochs[:, np.newaxis] != point_epochs[np.newaxis, :]
    category_matrix = np.zeros((len(categories), len(categories)))
    for row, category_a in enumerate(categories):
        for column, category_b in enumerate(categories):
            rows = after_onset & (point_categories == category_a)
            columns = after_onset & (point_categories == category_b)
            block = np.ix_(rows, columns)
            category_matrix[row, column] = recurrence[block][different_epoch[block]].mean()

    number_of_categories = len(state_names)
    category_cmap = ListedColormap(category_colors(number_of_categories, light=True))
    strip_options = dict(cmap=category_cmap, vmin=-0.5, vmax=number_of_categories - 0.5,
                         aspect="auto", interpolation="none")

    fig, (matrix_axis, summary_axis) = plt.subplots(
        1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [2, 1]}
    )
    matrix_axis.imshow(recurrence, cmap="Greys", interpolation="none")
    for boundary in epoch_boundaries:
        matrix_axis.axhline(boundary, color="tab:blue", linewidth=0.2, alpha=0.3)
        matrix_axis.axvline(boundary, color="tab:blue", linewidth=0.2, alpha=0.3)
    matrix_axis.set_xticks([])
    matrix_axis.set_yticks([])
    matrix_axis.set_xlabel(f"Zustand $j$ (sortiert nach Kategorie, jeder {time_step}. Zeitpunkt)")
    matrix_axis.set_ylabel("Zustand $i$")

    # Farbleisten: oben Kategorie der Epoche, links Label des Zeitpunkts
    divider = make_axes_locatable(matrix_axis)
    top_strip = divider.append_axes("top", size="4%", pad=0.05)
    top_strip.imshow(point_categories[np.newaxis, :], **strip_options)
    top_strip.set_axis_off()
    top_strip.set_title(
        f"Rekurrenzmatrix, $\\varepsilon$ = {epsilon:.3f} (Rekurrenzrate {recurrence_rate:.0%})"
    )
    left_strip = divider.append_axes("left", size="4%", pad=0.05)
    left_strip.imshow(point_labels[:, np.newaxis], **strip_options)
    left_strip.set_axis_off()
    matrix_axis.legend(
        handles=_category_legend_handles(state_names, light=True),
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=8,
    )

    image = summary_axis.imshow(category_matrix, cmap="viridis")
    names = [state_names[category] for category in categories]
    summary_axis.set_xticks(range(len(categories)), names, rotation=45, ha="right")
    summary_axis.set_yticks(range(len(categories)), names)
    for row in range(len(categories)):
        for column in range(len(categories)):
            summary_axis.text(column, row, f"{category_matrix[row, column]:.2f}",
                              ha="center", va="center", color="w", fontsize=9)
    summary_axis.set_title("Mittlere Rekurrenzrate zwischen Kategorien\n(verschiedene Epochen, nach Bildbeginn)")
    plt.colorbar(image, ax=summary_axis, fraction=0.046)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return recurrence, category_matrix, epsilon
