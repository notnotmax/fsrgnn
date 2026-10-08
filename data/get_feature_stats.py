"""
Used to get dynamic feature statistics from the training indices for a fold for
the purpose of feature normalisation.
Part of preprocessing.
"""

import numpy as np
import os
import pandas as pd
import h5py

from data.hecras_data_retrieval import get_min_cell_elevation, get_water_level, get_event_timesteps

def get_feature_stats(area_name: str, val_group: int):

    print(f'Getting {area_name} feature statistics for fold {val_group}.')
    DATASET_PATH = f'../dataset/{area_name}'
    EVENT_SUMMARY_PATH = os.path.join(DATASET_PATH, f'{area_name}_event_summary.csv')
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = len(event_summary)
    num_groups = np.max(event_summary['Group'])

    ttsplit = np.load(f'../dataset/{area_name}/Train_test_split_data/Train_test_split_ValidateOnGrp_{val_group}.npz')
    idx_train = ttsplit['idx_train']
    idx_test = ttsplit['idx_test']

    lf_water_depths = []

    for event_idx in range(num_events):

        group_id = event_summary['Group'][event_idx]
        if group_id == val_group:
            continue

        if area_name == 'Carlisle':
            lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
            lf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf')
            lf_water_depth = _get_lf_water_depth(lf_filepath, area_name)

        elif area_name == 'Chowilla':
            lf_run_name = event_summary['HEC_RAS_plan_LF'][event_idx]
            lf_filepath = f'../dataset/Chowilla/HD_model_data/Low-fidelity/Chow_LF_modelG.{lf_run_name}.hdf'
            lf_water_depth = _get_lf_water_depth(lf_filepath, area_name)

        elif area_name == 'BurnettRV':
            lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
            lf_filepath = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.{lf_run_name}.hdf'
            hf_run_name = event_summary['tuflow_evt_name_old'][event_idx]
            hf_filepath = f'../dataset/BurnettRV/HD_model_data/High-fidelity/Paradise_{hf_run_name}_002.npz'
            hf_data = np.load(hf_filepath)
            hf_timesteps = hf_data['time_data'].shape[0] - 48
            lf_water_depth = _get_lf_water_depth(lf_filepath, 'BurnettRV_region', hf_timesteps)

        else:
            assert False, f"unknown area name {area_name}."

        lf_water_depths.append(lf_water_depth)

    lf_water_depths = np.vstack(lf_water_depths)
    print(f'{lf_water_depths.shape[0]} timesteps in total for this fold.')
    assert lf_water_depths.shape[0] == idx_train.shape[0] + idx_test.shape[0]

    # calc stats
    lf_water_depths = lf_water_depths[idx_train, :]
    lf_water_depth_mean = np.mean(lf_water_depths)
    lf_water_depth_std = np.std(lf_water_depths)

    output_path = f'data/{area_name}/feature_stats_fold_{val_group}.npz'
    np.savez_compressed(
        output_path,
        lf_water_depth_mean=lf_water_depth_mean,
        lf_water_depth_std=lf_water_depth_std,
    )

    print(f"Saved {area_name} feature statistics (mean: {lf_water_depth_mean}, std: {lf_water_depth_std}) to {output_path}.")

def _get_lf_water_depth(lf_path: str, area_name: str, hf_timesteps: int = -1):

    # get ghost cell indices to filter out, assumed to be consistent across runs
    lf_elevation = get_min_cell_elevation(lf_path, area_name)
    ghost_idx = np.where(np.isnan(lf_elevation))[0]

    lf_water_level = get_water_level(lf_path, area_name)
    lf_water_depth = np.maximum(0.0, lf_water_level - lf_elevation)
    lf_water_depth = np.delete(lf_water_depth, ghost_idx, axis=1)

    if area_name == 'Carlisle':
        lf_water_depth = lf_water_depth[8:, :]
    elif area_name == 'Chowilla':
        pass # LF and HF already same timesteps
    elif area_name == 'BurnettRV_region':
        # trim 48 from start of HF then trim as needed from end of LF to match
        assert hf_timesteps > 0
        lf_water_depth = lf_water_depth[:hf_timesteps, :]

    return lf_water_depth


if __name__ == '__main__':
    # TIME paddings for burnett
    # 2, 8, 2, 0 (per fold)
    # event_summary = pd.read_csv(f'../dataset/BurnettRV/BurnettRV_event_summary.csv')
    # lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
    # lf_filepath = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.{lf_run_name}.hdf'
    # lf_path = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.p12.hdf'
    # # f = h5py.File(lf_path)
    # # print(f['Geometry']['2D Flow Areas'].keys())
    # # elev = get_min_cell_elevation(lf_path, 'Burnett')
    # lf_timesteps = get_event_timesteps(lf_path)
    # print('LF timesteps', lf_timesteps.shape)
    # print(lf_timesteps[:5])
    # # print(elev[0])
    # hf_data = np.load(f'../dataset/BurnettRV/HD_model_data/High-fidelity/Paradise_e1971_real_slmin_1in2_002.npz')
    # hf_timesteps = hf_data['time_data'] # or wl_data
    # print('HF timesteps', hf_timesteps.shape)
    # print(hf_timesteps[:5])

    for i in range(1, 5):
        get_feature_stats(area_name='BurnettRV', val_group=i)