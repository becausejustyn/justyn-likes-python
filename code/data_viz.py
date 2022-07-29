
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



import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.patches as patches

plt.style.use('seaborn')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
pd.set_option('max_columns', 100) # So we can see more columns


# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(y == ys[0] for y in ys)  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', color=color, **kwargs)


def _label_barh(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long
    """
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)



df.groupby('PlayId').first()['Yards'].plot(
    kind='hist', figsize=(15, 5), bins=50, title='Distribution of Yards Gained (Target)')
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
n = 0
for i, d in train.groupby('Down'):
    d['Yards'].plot(kind='hist',
                    bins=30,
                   color=color_pal[n],
                   ax=axes[n],
                   title=f'Yards Gained on down {i}')
    n+=1




# Create the DL-LB combos
train['DL_LB'] = train['DefensePersonnel'] \
    .str[:10] \
    .str.replace(' DL, ','-') \
    .str.replace(' LB','') # Clean up and convert to DL-LB combo
top_5_dl_lb_combos = train.groupby('DL_LB').count()['GameId'] \
    .sort_values() \
    .tail(10).index.tolist()
ax = train.loc[train['DL_LB'].isin(top_5_dl_lb_combos)] \
    .groupby('DL_LB').mean()['Yards'] \
    .sort_values(ascending=True) \
    .plot(kind='bar',
          title='Average Yards Top 5 Defensive DL-LB combos',
          figsize=(15, 5),
          color=color_pal[4])
# for p in ax.patches:
#     ax.annotate(str(round(p.get_height(), 2)),
#                 (p.get_x() * 1.005, p.get_height() * 1.015))

#bars = ax.bar(0.5, 5, width=0.5, align="center")
bars = [p for p in ax.patches]
value_format = "{:0.2f}"
label_bars(ax, bars, value_format, fontweight='bold')
plt.show()
