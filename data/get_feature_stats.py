"""
Used to get dynamic feature statistics from the training indices for a fold for
the purpose of feature normalisation.
Part of preprocessing.
"""

import numpy as np
from utils import get_event_summary, get_train_test_indices, get_lf_dyn_node_filepath, get_ups_wd_filepath, \
        get_feature_stats_filepath


def get_feature_stats(area_name: str, val_group: int):

    print(f'Getting {area_name} feature statistics for fold {val_group}.')
    event_summary = get_event_summary(area_name)
    idx_train, idx_test = get_train_test_indices(area_name, val_group)

    lf_water_depths = []
    ups_water_depths = []

    for event_idx in range(len(event_summary)):
        group_id = event_summary['Group'][event_idx]
        if group_id == val_group:
            continue

        if area_name == 'Carlisle':
            event_id = event_summary['No'][event_idx]
        elif area_name == 'Chowilla':
            event_id = event_summary['Event'][event_idx]
        elif area_name == 'BurnettRV':
            event_id = event_summary['No'][event_idx]
        else:
            raise NotImplementedError
        
        lf_water_depth = np.load(get_lf_dyn_node_filepath(area_name, event_id), mmap_mode='r')
        lf_water_depths.append(lf_water_depth)
        ups_water_depth = np.load(get_ups_wd_filepath(area_name, event_id), mmap_mode='r')
        ups_water_depths.append(ups_water_depth)

    expected_timesteps = idx_train.shape[0] + idx_test.shape[0]
    lf_timesteps = sum([a.shape[0] for a in lf_water_depths])
    ups_timesteps = sum([a.shape[0] for a in ups_water_depths])
    assert lf_timesteps == expected_timesteps, \
        f'Mismatch in lf water depth timesteps. Expected {expected_timesteps}, got {lf_timesteps}.'
    assert sum([a.shape[0] for a in ups_water_depths]) == idx_train.shape[0] + idx_test.shape[0], \
        f'Mismatch in upsampled water depth timesteps. Expected {expected_timesteps}, got {ups_timesteps}.'

    lf_wd_mean, lf_wd_std = _stream_stats(lf_water_depths, idx_train)
    ups_wd_mean, ups_wd_std = _stream_stats(ups_water_depths, idx_train)

    print(
        f'lf wd mean: {lf_wd_mean}',
        f'lf wd std: {lf_wd_std}',
        f'ups wd mean: {ups_wd_mean}',
        f'ups wd std: {ups_wd_std}',
    )

    output_path = get_feature_stats_filepath(area_name, val_group)
    np.savez_compressed(
        output_path,
        lf_wd_mean=lf_wd_mean,
        lf_wd_std=lf_wd_std,
        ups_wd_mean=ups_wd_mean,
        ups_wd_std=ups_wd_std
    )

    print(f"Saved {area_name} feature statistics to {output_path}.")


def _stream_stats(event_arrays, idx_train, chunk_size=64):
    event_ends = np.cumsum(np.array([a.shape[0] for a in event_arrays]))
    event_starts = np.concatenate(([0], event_ends[:-1]))
    event_indices = np.searchsorted(event_ends, idx_train, side='right')

    count, total, total_squares = 0, 0.0, 0.0
    for e, arr in enumerate(event_arrays):
        # change global indices to event local indices
        local_indices = np.sort(idx_train[event_indices == e] - event_starts[e])
        for i in range(0, len(local_indices), chunk_size):
            x = arr[local_indices[i:i+chunk_size]]
            x = x.astype(np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f'Non-finite values in event array index {e}')
            count += x.size
            total += x.sum()
            total_squares += np.square(x).sum()

    mean = total / count
    std = np.sqrt(max(total_squares / count - mean ** 2, 0.0))
    return float(mean), float(std)


def _get_percentiles(area_name: str):
    print(f'Getting LF water depth percentiles for {area_name}.')
    event_summary = get_event_summary(area_name)
    for event_idx in range(len(event_summary)):
        if area_name == 'Carlisle':
            event_id = event_summary['No'][event_idx]
        elif area_name == 'Chowilla':
            event_id = event_summary['Event'][event_idx]
        elif area_name == 'BurnettRV':
            event_id = event_summary['No'][event_idx]
        else:
            raise NotImplementedError
        lf_wd = np.load(get_lf_dyn_node_filepath(area_name, event_id), mmap_mode='r')
        print(f'Event {event_id}: 99: {np.percentile(lf_wd, 99):.3f}, 99.9: {np.percentile(lf_wd, 99.9):.3f},',
            f'99.99: {np.percentile(lf_wd, 99.99):.3f}, max: {np.max(lf_wd):.3f}')
        

if __name__ == '__main__':
    _get_percentiles('BurnettRV')
    # for i in range(1, 10):
    #     get_feature_stats(area_name='Carlisle', val_group=i)
