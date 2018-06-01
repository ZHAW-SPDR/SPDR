import contextlib
import pickle
import numpy as np
import matplotlib.pyplot as plt

GRIDSEARCH_RESULT_PICKLE = "data/pickles/GRIDSEARCH_DOMINANTSET.pkl"


if __name__ == '__main__':
    with contextlib.closing(open(GRIDSEARCH_RESULT_PICKLE, "rb")) as f:
        mrs = pickle.load(f)

    thetas = np.arange(0.0, 0.9995, 0.0005)
    theta_ticks = np.arange(0.0, 0.9995, 0.1)
    epsilons = [1 / 10**e for e in range(2, 12)]

    theta_map = {}
    eps_map = {}

    for idx, theta in enumerate(thetas):
        theta_map[theta] = idx

    for idx, e in enumerate(epsilons):
        eps_map[e] = idx

    n = len(thetas)
    m = len(epsilons)

    heatmap = np.zeros((n, m))

    for (e, t), (mr, randi, acp) in mrs.items():
        heatmap[theta_map[t]][eps_map[e]] = mr

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, interpolation='nearest', aspect='auto', extent=[0, 9, 0.0, 0.9995], origin="lower", cmap="plasma")

    ax.set_title("Grid search dominant set")
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$\theta$')

    ax.set_yticks(theta_ticks)
    ax.set_xticks(np.arange(len(epsilons)))

    ax.set_xticklabels(epsilons)
    ax.set_yticklabels(["%.1f" % t for t in theta_ticks])

    fig.tight_layout()
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("MR")

    plt.savefig("data/plots/dominantset/gridsearch.png")