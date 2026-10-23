import matplotlib.pyplot as plt
import numpy as np
from data.make_dataset import make_dataset_one_event, make_dataset_one_group

def plot_histogram(feature, title, xlabel, ylabel, output_path):
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Histogram
    counts, bins, patches = ax.hist(
        feature,
        bins=50,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )

    # Calculate summary metrics
    median_err = np.median(feature)
    p95_err = np.percentile(feature, 95)

    # Add reference lines
    ax.axvline(
        median_err,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_err:.3f} m",
    )
    ax.axvline(
        p95_err,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"95th Percentile: {p95_err:.3f} m",
    )

    # Formatting
    ax.set_title(
        title,
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Successfully saved plot at {output_path}')
    plt.close()

def plot_scatter(coords, feature, title, label, output_path):
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=feature,
        cmap='viridis',
        s=12,
        alpha=0.9
    )

    plt.colorbar(scatter, label=label)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Successfully saved plot at {output_path}')
    plt.close()

def plot_line(x, y, title, xlabel, ylabel, output_path, ylim_bottom = None, ylim_top = None):
    curves = y if isinstance(y, dict) else {None: y}
    plt.figure(figsize=(9, 7))
    for label, values in curves.items():
        plt.plot(x, values, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Successfully saved plot at {output_path}')
    plt.close()
