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

if __name__ == '__main__':
    # CARLISLE_LF_GEOM = np.load(f'data/Carlisle/LF_geometry_data.npz')
    # CARLISLE_HF_GEOM = np.load(f'data/Carlisle/HF_geometry_data.npz')
    # CHOWILLA_LF_GEOM = np.load(f'data/Chowilla/LF_geometry_data.npz')
    # CHOWILLA_HF_GEOM = np.load(f'data/Chowilla/HF_geometry_data.npz')
    print('Running visualisation...')

    area_name = 'Chowilla'

    # hf_elevation = CHOWILLA_HF_GEOM['cell_elevation']
    dataset = make_dataset(area_name=area_name, val_group=1, mode='eval')
    wet_idx = dataset.wet_idx

    ups_wd = np.load(f'data/{area_name}/processed/upsampled_water_depth_event_1.npy', mmap_mode='r')[:, wet_idx]
    hf_wd = np.load(f'data/{area_name}/processed/hf_water_depth_event_1.npy', mmap_mode='r')[:, wet_idx]

    # ups_wd = np.max(ups_wd, axis=0)
    # hf_wd = np.max(hf_wd, axis=0)

    # plot_histogram(
    #     feature=ups_wd,
    #     title=f'{area_name}, Event 1 Wet Cells',
    #     xlabel='Maximum Upsampled Water Depth (m)',
    #     ylabel='Node Count',
    #     output_path=f'plots/{area_name}_ups_wd_dist_wet.png'
    # )

    # plot_histogram(
    #     feature=hf_wd,
    #     title=f'{area_name}, Event 1 Wet Cells',
    #     xlabel='Maximum HF Water Depth (m)',
    #     ylabel='Node Count',
    #     output_path=f'plots/{area_name}_hf_wd_dist_wet.png'
    # )

    error = ups_wd - hf_wd
    print('Errors < 5:', np.count_nonzero(error < -5.0))
    print('Errors > 5:', np.count_nonzero(error > 5.0))
    error = np.clip(error, -5.0, 5.0)

    plot_histogram(
        feature=error.flatten(),
        title=f'{area_name}, Event 1 Wet Cells, over all timesteps',
        xlabel='Water Depth Error (m)',
        ylabel='Node Count',
        output_path=f'plots/{area_name}_clipped_signed_err_wd_dist_wet.png'
    )
