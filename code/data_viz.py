
import numpy as np
from typing import List
from util import neg_log_likelihood
import matplotlib.pyplot as plt
from gradient_descent import GradientDescent
from matplotlib.colors import LogNorm
from loss_functions import neg_log_likelihood

plt.rcParams.update({'font.size': 8})


x = np.linspace(-1.5, 1.5, 30)
px = 0.8
py = px**2

plt.plot(x, x**2, "b-", px, py, "ro")

#plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='blue', horizontalalignment="center")
bbox_props = dict(boxstyle="round4,pad=1,rounding_size=0.2", ec="black", fc="#EEEEFF", lw=5)
plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='black', ha="center", bbox=bbox_props)
#plt.text(px - 0.08, py, "Beautiful point", ha="right", weight="heavy")
bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="b", lw=2, fc="lightblue")
plt.text(px-0.2, py, "Beautiful point", bbox=bbox_props, ha="right")
plt.text(px, py, "x = %0.2f\ny = %0.2f"%(px, py), rotation=50, color='gray')

plt.grid(True)
plt.show()







def plot_gradient_descent(stats, f, x_opt, y_opt, dpi = 120, bbox_inches = "tight"):
    markers=['s', 'o', 'p', 'P', 'x', '+', '<', 'v']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    domain_size = 3.2
    x_min, x_max = -domain_size, domain_size
    y_min, y_max = -domain_size, domain_size

    step_size = 0.01
    x = np.arange(x_min, x_max, step_size)
    y = np.arange(y_min, y_max, step_size)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy)
    ax.contour(x, y, z, levels=np.logspace(-1, 6, 128), norm=LogNorm(), cmap="viridis", linewidths=0.3)

    for (key, value), marker in zip(stats.items(), markers):
        ax.plot(stats[key]["x"], stats[key]["y"], label=key, linewidth=0.5, marker=marker, markersize=2)

    ax.plot(x_opt, y_opt, marker="*", markersize=10, color="k")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    plt.savefig("gd.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def plot_loss(stats, x_min, y_min, dpi = 120, bbox_inches = "tight"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    for key, value in stats.items():
        x = np.array(stats[key]["x"])
        y = np.array(stats[key]["y"])
        loss = np.sqrt((x_min - x)**2 + (y_min - y)**2)
        iteration = np.linspace(1, len(loss), len(loss))
        ax.plot(iteration, loss, label=key, linewidth=0.5)
        ax.grid("True", linestyle="--", linewidth=0.5)

    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    n_points = 200
    ax2 = ax.inset_axes([0.07, 0.02, 0.42, 0.42])
    for key, value in stats.items():
        x = np.array(stats[key]["x"])
        y = np.array(stats[key]["y"])
        loss = np.sqrt((x_min - x)**2 + (y_min - y)**2)
        iteration = np.arange(len(loss))
        ax2.plot(iteration[:n_points], loss[:n_points], linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid("True", linestyle="--", linewidth=0.5)
    ax.indicate_inset_zoom(ax2, linewidth=1)
    
    plt.savefig("loss.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def plot_learning_rates(stats, dpi = 120, bbox_inches = "tight"):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    i = 0
    axes = axes.flatten()
    for key, value in stats.items():
        if "+" in key:
            lr_x = np.array(stats[key]["eta_x"])
            lr_y = np.array(stats[key]["eta_y"])
            axes[i].plot(lr_x, label="$\eta_x$", linewidth=0.5)
            axes[i].plot(lr_y, label="$\eta_y$", linewidth=0.5)
            axes[i].set_title(key)
            i += 1

    for i, ax in enumerate(axes.flatten()):
        ax.legend(loc="lower right")
        if i > 1:
            ax.set_xlabel("Iterations")
        if (i+1) % 2 != 0:
            ax.set_ylabel("Learning rate")

    plt.savefig("lr.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def main():
    optimizers = ["GD", "GD+", "GDM", "GDM+", "NAG", "NAG+", "Adam", "Adam+"]
    learning_rates = [0.01, 0.01, 0.015, 0.01, 0.006, 0.006, 0.0005, 0.0005]
    alphas = [0.0, 1e-4, 0.0, 1e-5, 0.0, 1e-6, 0.0, 1e-8]

    x_min, y_min = 3.0, 0.5
    x_ini, y_ini = -2.0, -1.0

    stats = {optimizer : {"x" : None, "y" : None, "eta_x" : None, "eta_y" : None} for optimizer in optimizers}

    n_iterations = 1000 # 10000 adam # 4400 nag # 2000 gdm 1200 gd

    for optimizer, learning_rate, alpha in zip(optimizers, learning_rates, alphas):
        print(optimizer)
        model = GradientDescent(f, x_ini, y_ini, n_iterations, learning_rate, alpha, optimizer)
        x_hist, y_hist, lr_x_hist, lr_y_hist = model.minimize()
        stats[optimizer]["x"] = x_hist
        stats[optimizer]["y"] = y_hist
        stats[optimizer]["eta_x"] = lr_x_hist
        stats[optimizer]["eta_y"] = lr_y_hist

    plot_gradient_descent(stats, f, x_min, y_min)
    plot_loss(stats, x_min, y_min)
    plot_learning_rates(stats)

def plot_neg_log_likelihood(min = 1, max = 10000):
    # Plot of the negative log likelihood function to see how incorrect predictions influence the overall error
    xs_nll: List[float] = [x / max for x in range(min, max)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs_nll, [neg_log_likelihood(1, x) for x in xs_nll])
    ax1.set_title('y = 1')
    ax2.plot(xs_nll, [neg_log_likelihood(0, x) for x in xs_nll])
    ax2.set_title('y = 0')

def plot_decision_boundary(ys, beta):
    x1s: List[float] = [x[1] for x in xs]
    x2s: List[float] = [x[2] for x in xs]
    plt.scatter(x1s, x2s, c=ys)
    plt.axis([min(x1s), max(x1s), min(x2s), max(x2s)])
    m: float = -(beta[1] / beta[2])
    b: float = -(beta[0] / beta[2])

    x2s: List[float] = [m * x[1] + b for x in xs]
    plt.plot(x1s, x2s, '--')



