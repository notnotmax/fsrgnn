"""
Used to create Dataset objects.
Mainly handles the differing filepath names.
"""

import numpy as np
import os
import pandas as pd

from data.dataset import FloodEventDataset
from data.hecras_data_retrieval import get_event_timesteps

def make_carlisle_dataset(validation_group: int, mode: str):

    print(f"Making Carlisle dataset on validation group {validation_group}.")
    DATASET_PATH = '../dataset/Carlisle'
    EVENT_SUMMARY_PATH = os.path.join(DATASET_PATH, 'Carlisle_event_summary.csv')
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = len(event_summary)
    num_groups = np.max(event_summary['Group'])
    EVENT_NUM_TIMESTEPS = [266, 242, 198, 253, 318, 199, 266, 312, 316] # static
    print(f'Total: {num_events} events across {num_groups} groups.')

    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    num_timesteps = []

    for event_idx in range(num_events):
        event_id = event_summary['No'][event_idx]
        group_id = event_summary['Group'][event_idx]

        if mode == 'train':
            if group_id == validation_group: # add everything except val group
                continue
        elif mode == 'eval':
            if group_id != validation_group: # only add val group
                continue
        else:
            assert False, f'Unknown dataset creation mode.'

        lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
        lf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf')
        hf_run_name = event_summary['Lisflood'][event_idx]
        hf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/High-fidelity/{hf_run_name}_alltimesteps.npz')
        # hf_data = np.load(hf_filepath, mmap_mode='r') # slow
        # event_num_timesteps = hf_data['wse_data'].shape[0]
        event_num_timesteps = EVENT_NUM_TIMESTEPS[event_idx]

        event_ids.append(event_id)
        group_ids.append(group_id)
        lf_filepaths.append(lf_filepath)
        hf_filepaths.append(hf_filepath)
        num_timesteps.append(event_num_timesteps)

        print(f'Added event (event id {event_id}, group id {group_id}) to train split. \
                Event has {event_num_timesteps} timesteps.')

    cat_data_path = os.path.join(DATASET_PATH, f'HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_{validation_group}.npz')
    feature_stats_path = f'data/Carlisle/feature_stats_fold_{validation_group}.npz'

    # triggers the preprocessing
    dataset = FloodEventDataset(
        root_dir = 'data/Carlisle',
        area_name = 'Carlisle',
        lf_geometry_path = 'data/Carlisle/LF_geometry_data.npz',
        hf_geometry_path = 'data/Carlisle/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        hf_filetype = 'npz',
        cat_data_path = cat_data_path,
        feature_stats_path = feature_stats_path,
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = num_timesteps,
        previous_timesteps = 0
    )

    print(f'Made Carlisle dataset, contains total {len(dataset)} timesteps.')

    return dataset

def make_chowilla_dataset(): # TODO refactor

    print("Making Chowilla dataset.")
    DATASET_PATH = '../dataset/Chowilla'
    EVENT_SUMMARY_PATH = osp.join(DATASET_PATH, 'Chowilla_event_summary.csv')
    event_df = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = len(event_df)
    print(f"Number of events: {num_events}")
    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    num_timesteps = []
    PAD_LENGTH = 40 # defined in LSG dataset code

    for i in range(2): # TODO

        event_ids.append(event_df['Event'][i])
        group_ids.append(event_df['Group'][i])

        lf_run_name = event_df['HEC_RAS_plan_LF'][i]
        lf_filepath = osp.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Chow_LF_modelG.{lf_run_name}.hdf')
        lf_filepaths.append(lf_filepath)

        hf_run_name = event_df['HEC_RAS_plan_HF'][i]
        hf_filepath = osp.join(DATASET_PATH, f'HD_model_data/High-fidelity/Chow_HF.{hf_run_name}.hdf')
        hf_filepaths.append(hf_filepath)

        num_timesteps = len(get_event_timesteps(hf_filepath)) - PAD_LENGTH # TODO wrong
    
    dataset = FloodEventDataset(
        root_dir = 'data/Chowilla',
        area_name = 'Chowilla',
        lf_geometry_path = 'data/Chowilla/LF_geometry_data.npz',
        hf_geometry_path = 'data/Chowilla/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = num_timesteps,
        previous_timesteps = 0
    )

def get_all_event_timesteps_carlisle():
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

def get_all_event_timesteps_burnett():
    # burnett has 2 more timesteps in LF than HF

    # for debugging
    EVENT_SUMMARY_PATH = '../dataset/BurnettRV/BurnettRV_event_summary.csv'
    ev_summary = pd.read_csv(EVENT_SUMMARY_PATH)
    for i in range(15, len(ev_summary)):
        print(f"----- run {i+1} -----")
        # get LF timesteps
        lf_run_name = ev_summary['HEC_RAS_plan'][i]
        lf_filepath = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.{lf_run_name}.hdf'
        print('LF timesteps', len(get_event_timesteps(lf_filepath)))

        # get HF timesteps
        hf_run_name = ev_summary['tuflow_evt_name_old'][i]
        hf_filepath = f'../dataset/BurnettRV/HD_model_data/High-fidelity/Paradise_{hf_run_name}_002.npz'
        hf_data = np.load(hf_filepath)
        hf_wse = hf_data['wl_data']
        print('HF timesteps', hf_wse.shape)

if __name__ == '__main__':
    make_carlisle_dataset(-1, 'train')
