"""
Used to create Dataset objects.
Mainly handles the differing filepath names.
"""

import numpy as np
import os
import pandas as pd

from data.dataset import FloodEventDataset
from data.hecras_data_retrieval import get_event_timesteps

def get_event_data_carlisle(event_summary, event_idx: int):
    event_id = event_summary['No'][event_idx]
    group_id = event_summary['Group'][event_idx]
    lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
    lf_filepath = f'../dataset/Carlisle/HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf'
    hf_run_name = event_summary['Lisflood'][event_idx]
    hf_filepath = f'../dataset/Carlisle/HD_model_data/High-fidelity/{hf_run_name}_alltimesteps.npz'
    # hf_data = np.load(hf_filepath, mmap_mode='r') # slow
    # num_timesteps = hf_data['wse_data'].shape[0]
    EVENTS_NUM_TIMESTEPS = [266, 242, 198, 253, 318, 199, 266, 312, 316] # HF timesteps from code above
    num_timesteps = EVENTS_NUM_TIMESTEPS[event_idx]

    return event_id, group_id, lf_filepath, hf_filepath, num_timesteps


def get_event_data_chowilla(event_summary, event_idx: int):
    event_id = event_summary['Event'][event_idx]
    group_id = event_summary['Group'][event_idx]
    lf_run_name = event_summary['HEC_RAS_plan_LF'][event_idx]
    lf_filepath = f'../dataset/Chowilla/HD_model_data/Low-fidelity/Chow_LF_modelG.{lf_run_name}.hdf'
    hf_run_name = event_summary['HEC_RAS_plan_HF'][event_idx]
    hf_filepath = f'../dataset/Chowilla/HD_model_data/High-fidelity/Chow_HF.{hf_run_name}.hdf'
    num_timesteps = len(get_event_timesteps(lf_filepath)) # LF and HF have same num of timesteps TODO check this after filtering

    return event_id, group_id, lf_filepath, hf_filepath, num_timesteps


def get_event_data_burnett(event_summary, event_idx: int):
    event_id = event_summary['No'][event_idx]
    group_id = event_summary['Group'][event_idx]
    lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
    lf_filepath = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.{lf_run_name}.hdf'
    hf_run_name = event_summary['tuflow_evt_name_old'][event_idx]
    hf_filepath = f'../dataset/BurnettRV/HD_model_data/High-fidelity/Paradise_{hf_run_name}_002.npz'
    hf_data = np.load(hf_filepath)
    hf_timesteps = hf_data['time_data'].shape[0]
    num_timesteps = hf_timesteps - 48 # trim 48 timesteps from the start of each HF run

    return event_id, group_id, lf_filepath, hf_filepath, num_timesteps


def make_dataset_one_event(area_name: str, event_idx: int):
    EVENT_SUMMARY_PATH = f'../dataset/{area_name}/{area_name}_event_summary.csv'
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)
    assert 0 <= event_idx < len(event_summary), f'Event index {event_idx} out of bounds for event summary of length {len(event_summary)}.'

    if area_name == 'Carlisle':
        event_id, group_id, lf_filepath, hf_filepath, num_timesteps = get_event_data_carlisle(event_summary, event_idx)
    elif area_name == 'Chowilla':
        event_id, group_id, lf_filepath, hf_filepath, num_timesteps = get_event_data_chowilla(event_summary, event_idx)
    elif area_name == 'BurnettRV':
        event_id, group_id, lf_filepath, hf_filepath, num_timesteps = get_event_data_burnett(event_summary, event_idx)
    else:
        assert False, f'Unknown area name {area_name}.'

    print(f'Added event (event id {event_id}, group id {group_id}) to dataset. '
            f'Event has {num_timesteps} timesteps.')

    dataset = FloodEventDataset(
        mode = 'eval',
        root_dir = f'data/{area_name}',
        area_name = area_name,
        lf_geometry_path = f'data/{area_name}/LF_geometry_data.npz',
        hf_geometry_path = f'data/{area_name}/HF_geometry_data.npz',
        lf_hecras_paths = [lf_filepath],
        hf_paths = [hf_filepath],
        cat_data_path = f'../dataset/{area_name}/HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_{group_id}.npz',
        feature_stats_path = f'data/{area_name}/feature_stats_fold_{group_id}.npz',
        event_ids = [event_id],
        group_ids = [group_id],
        num_timesteps = [num_timesteps]
    )

    return dataset


def make_dataset_one_group(area_name: str, val_group: int, mode: str):
    assert val_group > 0, 'Validation group should be 1-indexed (>0). Check the event summary to confirm otherwise.'

    print(f'Making {area_name} dataset on validation group {val_group} in mode "{mode}".')
    DATASET_PATH = f'../dataset/{area_name}'
    EVENT_SUMMARY_PATH = os.path.join(DATASET_PATH, f'{area_name}_event_summary.csv')
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    events_num_timesteps = []

    for event_idx in range(len(event_summary)):

        group_id = event_summary['Group'][event_idx]
        if mode == 'train':
            if group_id == val_group: # add everything except val group
                continue
        elif mode == 'eval':
            if group_id != val_group: # only add val group
                continue
        else:
            assert False, (f'Dataset creation mode for one group should be "train" (exclude events from this group)'
                'or "eval" (only include events from this group).')

        if area_name == 'Carlisle':
            event_id, _, lf_filepath, hf_filepath, num_timesteps = get_event_data_carlisle(event_summary, event_idx)
        elif area_name == 'Chowilla':
            event_id, _, lf_filepath, hf_filepath, num_timesteps = get_event_data_chowilla(event_summary, event_idx)
        elif area_name == 'BurnettRV':
            event_id, _, lf_filepath, hf_filepath, num_timesteps = get_event_data_burnett(event_summary, event_idx)
        else:
            assert False, f'Unknown area name {area_name}.'

        event_ids.append(event_id)
        group_ids.append(group_id)
        lf_filepaths.append(lf_filepath)
        hf_filepaths.append(hf_filepath)
        events_num_timesteps.append(num_timesteps)

        print(f'Added event (event id {event_id}, group id {group_id}) to dataset. '
                f'Event has {num_timesteps} timesteps.')

    dataset = FloodEventDataset(
        mode = mode,
        root_dir = f'data/{area_name}',
        area_name = area_name,
        lf_geometry_path = f'data/{area_name}/LF_geometry_data.npz',
        hf_geometry_path = f'data/{area_name}/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        cat_data_path = f'../dataset/{area_name}/HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_{val_group}.npz',
        feature_stats_path = f'data/{area_name}/feature_stats_fold_{val_group}.npz',
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = events_num_timesteps
    )

    print(f'Made {area_name} dataset in "{mode}" mode. Contains {len(event_ids)} events and {len(dataset)} timesteps in total.')

    return dataset


def make_dataset_all_events(area_name: str):
    """
    Creates a dataset using all event data to use in preprocessing and debugging.
    Not intended for training or evaluation.
    """
    EVENT_SUMMARY_PATH = f'../dataset/{area_name}/{area_name}_event_summary.csv'
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    events_num_timesteps = []

    for event_idx in range(len(event_summary)):

        if area_name == 'Carlisle':
            event_id, group_id, lf_filepath, hf_filepath, num_timesteps = get_event_data_carlisle(event_summary, event_idx)
        elif area_name == 'Chowilla':
            event_id, group_id, lf_filepath, hf_filepath, num_timesteps = get_event_data_chowilla(event_summary, event_idx)
        elif area_name == 'BurnettRV':
            event_id, group_id, lf_filepath, hf_filepath, num_timesteps = get_event_data_burnett(event_summary, event_idx)
        else:
            assert False, f'Unknown area name {area_name}.'
        
        event_ids.append(event_id)
        group_ids.append(group_id)
        lf_filepaths.append(lf_filepath)
        hf_filepaths.append(hf_filepath)
        events_num_timesteps.append(num_timesteps)

        print(f'Added event (event id {event_id}, group id {group_id}) to dataset. '
                f'Event has {num_timesteps} timesteps.')

    dataset = FloodEventDataset(
        mode = 'all',
        root_dir = f'data/{area_name}',
        area_name = area_name,
        lf_geometry_path = f'data/{area_name}/LF_geometry_data.npz',
        hf_geometry_path = f'data/{area_name}/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        cat_data_path = None,
        feature_stats_path = None,
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = events_num_timesteps
    )

    print(f'Made {area_name} dataset in "all" mode. Contains {len(event_ids)} events and {len(dataset)} timesteps in total.')

    return dataset


def _get_all_event_timesteps_carlisle():
    # CARLISLE always has time difference 8 between LF and HF runs

    # for debugging
    CARLISLE_EVENT_SUMMARY_PATH = '../dataset/Carlisle/Carlisle_event_summary.csv'
    ev_summary = pd.read_csv(CARLISLE_EVENT_SUMMARY_PATH)
    for i in range(len(ev_summary)):
        print(f"----- run {i+1} -----")
        # get LF timesteps
        lf_run_name = ev_summary['HEC_RAS_plan'][i]
        lf_filepath = f'../dataset/Carlisle/HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf'
        print('LF timesteps', len(get_event_timesteps(lf_filepath)))

        # get HF timesteps
        hf_run_name = ev_summary['Lisflood'][i]
        hf_filepath = f'../dataset/Carlisle/HD_model_data/High-fidelity/{hf_run_name}_alltimesteps.npz'
        hf_data = np.load(hf_filepath)
        hf_wse = hf_data['wse_data']
        print('HF timesteps', hf_wse.shape)

def _get_all_event_timesteps_chowilla():
    print('Getting Chowilla timestep lengths.')

    # for debugging
    EVENT_SUMMARY_PATH = '../dataset/Chowilla/Chowilla_event_summary.csv'
    ev_summary = pd.read_csv(EVENT_SUMMARY_PATH)
    for i in range(len(ev_summary)):
        print(f"----- run {i+1} / {len(ev_summary)} -----")
        # get LF timesteps
        lf_run_name = ev_summary['HEC_RAS_plan_LF'][i]
        lf_filepath = f'../dataset/Chowilla/HD_model_data/Low-fidelity/Chow_LF_modelG.{lf_run_name}.hdf'
        lf_timesteps = get_event_timesteps(lf_filepath)
        print(f'LF timesteps: {len(lf_timesteps)}, first timesteps: {lf_timesteps[0:3]}')

        # get HF timesteps
        hf_run_name = ev_summary['HEC_RAS_plan_HF'][i]
        hf_filepath = f'../dataset/Chowilla/HD_model_data/High-fidelity/Chow_HF.{hf_run_name}.hdf'
        hf_timesteps = get_event_timesteps(hf_filepath)
        print(f'HF timesteps: {len(hf_timesteps)}, first timesteps: {hf_timesteps[0:3]}')

def _get_all_event_timesteps_burnett():
    # for debugging
    EVENT_SUMMARY_PATH = '../dataset/BurnettRV/BurnettRV_event_summary.csv'
    ev_summary = pd.read_csv(EVENT_SUMMARY_PATH)
    for i in range(len(ev_summary)):
        print(f"----- run {i+1} / {len(ev_summary)} -----")
        # get LF timesteps
        lf_run_name = ev_summary['HEC_RAS_plan'][i]
        lf_filepath = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.{lf_run_name}.hdf'
        print('LF timesteps', len(get_event_timesteps(lf_filepath)))

        # get HF timesteps
        hf_run_name = ev_summary['tuflow_evt_name_old'][i]
        hf_filepath = f'../dataset/BurnettRV/HD_model_data/High-fidelity/Paradise_{hf_run_name}_002.npz'
        hf_data = np.load(hf_filepath)
        hf_timesteps = hf_data['time_data']
        print('HF timesteps', hf_timesteps.shape)

if __name__ == '__main__':
    AREA_NAME = 'BurnettRV'
    print(f'Processing {AREA_NAME} dataset.')
    dataset = make_dataset(area_name=AREA_NAME, val_group=1, mode='all')
    dataset.process_all()
