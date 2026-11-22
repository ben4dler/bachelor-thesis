import matplotlib.pyplot as plt
import numpy as np

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

def plotting_neuronal_behavioural(X, B, state_names=[], vmin=0, vmax=2):
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    im0 = axs[0].imshow(X.T, aspect="auto", vmin=vmin, vmax=vmax, interpolation="None")
    # tell the colorbar to tick at integers
    cax0 = plt.colorbar(im0)
    axs[0].set_xlabel("time $t$")
    axs[0].set_ylabel("Neuronal activation")

    # get discrete colormap
    colors = sns.color_palette("pastel", len(state_names))
    cmap = cm.colors.ListedColormap(colors)
    cmap = plt.get_cmap("Pastel1", np.max(B) - np.min(B) + 1)
    im1 = axs[1].imshow(
        [B], cmap=cmap, vmin=np.min(B) - 0.5, vmax=np.max(B) + 0.5, aspect="auto"
    )
    # tell the colorbar to tick at integers
    cax = plt.colorbar(im1, ticks=np.arange(np.min(B), np.max(B) + 1))
    if state_names != []:
        cax.ax.set_yticklabels(state_names)
    axs[1].set_xlabel("time $t$")
    axs[1].set_ylabel("Behaviour")
    axs[1].set_yticks([])
    plt.show()


def plot_phase_space(Y, B, state_names, show_points=False, legend=True, **kwargs):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    plot_ps_(
        fig,
        ax,
        Y=Y,
        B=B,
        state_names=state_names,
        show_points=show_points,
        legend=legend,
        **kwargs,
    )
    plt.show()
    return fig, ax


def plot_ps_(
    fig, ax, Y, B, state_names, show_points=False, legend=True, colors=None, **kwargs
):
    if colors is None:
        colors = sns.color_palette("deep", len(state_names))

        max_arrows = 1000
        step = max(1, (len(Y) - 1) // max_arrows)

        for i in range(0, len(Y) - 1, step):
            d = Y[i + 1] - Y[i]
            norm = np.linalg.norm(d)

            if norm == 0:
                continue

            ax.quiver(
                Y[i, 0],
                Y[i, 1],
                Y[i, 2],
                d[0],
                d[1],
                d[2],
                color=colors[B[i]],
                arrow_length_ratio=0.1 / norm,
                linewidths=1,
                **kwargs,
            )
    ax.set_axis_off()

    if legend == True:
        # Create legend
        legend_elements = [
            Line2D([0], [0], color=c, lw=4, label=state)
            for c, state in zip(colors, state_names)
        ]
        ax.legend(handles=legend_elements)

    if show_points == True:
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c="k", s=1, cmap=ListedColormap(colors))
    return fig, ax


def rotating_plot(
    Y, B, state_names, show_points=False, legend=True, filename="rotation.gif", **kwargs
):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")

    def rotate(angle):
        ax.view_init(azim=angle)

    fig, ax = plot_ps_(
        fig,
        ax,
        Y=Y,
        B=B,
        state_names=state_names,
        show_points=show_points,
        legend=legend,
        **kwargs,
    )
    rot_animation = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 362, 5), interval=150
    )
    rot_animation.save(filename, dpi=150, writer="imagemagick")
    plt.show()
    return ax


def plot_latent_timeseries(Y, B, state_names):
    plt.figure(figsize=(19, 5))
    cmap = plt.get_cmap("Pastel1", np.max(B) - np.min(B) + 1)
    im = plt.imshow(
        [B], aspect=600, cmap=cmap, vmin=np.min(B) - 0.5, vmax=np.max(B) + 0.5
    )
    # cbar = plt.colorbar(ticks=np.arange(len(state_names)))
    cbar = plt.colorbar(im, ticks=np.arange(np.min(B), np.max(B) + 1))
    cbar.ax.set_yticklabels(state_names)
    plt.plot(Y / np.max(np.abs(Y)) / 3)
    plt.xlabel("time $t$")
    plt.axis([0, Y.shape[0], -0.5, 0.5])
    plt.show()


