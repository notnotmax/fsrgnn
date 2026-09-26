"""
Used to get dynamic feature statistics from the training indices for a fold for
the purpose of feature normalisation.
Part of preprocessing.
"""

import numpy as np
import os
import pandas as pd

from data.hecras_data_retrieval import get_min_cell_elevation, get_water_level

def get_carlisle_feature_stats(val_group: int):

    print(f'Getting Carlisle feature statistics for fold {val_group}.')
    DATASET_PATH = '../dataset/Carlisle'
    EVENT_SUMMARY_PATH = os.path.join(DATASET_PATH, 'Carlisle_event_summary.csv')
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = len(event_summary)
    num_groups = np.max(event_summary['Group'])
    print(f'Total: {num_events} events across {num_groups} groups.')

    ttsplit = np.load(f'../dataset/Carlisle/Train_test_split_data/Train_test_split_ValidateOnGrp_{val_group}.npz')
    idx_train = ttsplit['idx_train']

    lf_water_depths = []

    for event_idx in range(num_events):

        lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
        lf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf')

        lf_water_depth = _get_lf_water_depth(lf_filepath, 'Carlisle')
        lf_water_depths.append(lf_water_depth)

    # calc stats
    lf_water_depths = np.vstack(lf_water_depths)
    lf_water_depths = lf_water_depths[idx_train, :]

    lf_water_depth_mean = np.mean(lf_water_depths)
    lf_water_depth_std = np.std(lf_water_depths)

    output_path = f'data/Carlisle/feature_stats_fold_{val_group}.npz'
    np.savez_compressed(
        output_path,
        lf_water_depth_mean=lf_water_depth_mean,
        lf_water_depth_std=lf_water_depth_std,
    )

    print(f"Saved feature statistics to {output_path}.")

def _get_lf_water_depth(lf_path: str, area_name: str):
    PADDING = 8 # for carlisle

    # get ghost cell indices to filter out, assumed to be consistent across runs
    lf_elevation = get_min_cell_elevation(lf_path, area_name)
    ghost_idx = np.where(np.isnan(lf_elevation))[0]

    lf_water_level = get_water_level(lf_path, area_name)
    lf_water_depth = np.maximum(0.0, lf_water_level - lf_elevation)
    lf_water_depth = np.delete(lf_water_depth, ghost_idx, axis=1)
    lf_water_depth = lf_water_depth[PADDING:, :]

    return lf_water_depth

if __name__ == '__main__':
    for i in range(1, 10):
        get_carlisle_feature_stats(val_group=i)