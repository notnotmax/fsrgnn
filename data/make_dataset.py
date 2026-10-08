"""
Used to create Dataset objects.
Mainly handles the differing filepath names.
"""

import numpy as np
import os
import pandas as pd

from data.dataset import FloodEventDataset
from data.hecras_data_retrieval import get_event_timesteps

def make_dataset(area_name: str, val_group: int, mode: str):

    assert val_group > 0, 'Validation group should be 1-indexed (>0). Check the event summary to confirm otherwise.'

    print(f'Making {area_name} dataset on validation group {val_group} in mode "{mode}".')
    DATASET_PATH = f'../dataset/{area_name}'
    EVENT_SUMMARY_PATH = os.path.join(DATASET_PATH, f'{area_name}_event_summary.csv')
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = len(event_summary)
    num_groups = np.max(event_summary['Group'])
    print(f'Total: {num_events} events across {num_groups} groups.')

    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    num_timesteps = []

    for event_idx in range(num_events):

        group_id = event_summary['Group'][event_idx]
        if mode == 'train':
            if group_id == val_group: # add everything except val group
                continue
        elif mode == 'eval':
            if group_id != val_group: # only add val group
                continue
        elif mode == 'all':
            pass # add all events w/o filtering
        else:
            assert False, f'Unknown dataset creation mode, should be "train", "eval" or "all"'

        if area_name == 'Carlisle':
            event_id = event_summary['No'][event_idx]
            lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
            lf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf')
            hf_run_name = event_summary['Lisflood'][event_idx]
            hf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/High-fidelity/{hf_run_name}_alltimesteps.npz')
            # hf_data = np.load(hf_filepath, mmap_mode='r') # slow
            # event_num_timesteps = hf_data['wse_data'].shape[0]
            EVENT_NUM_TIMESTEPS = [266, 242, 198, 253, 318, 199, 266, 312, 316] # HF timesteps from code above
            event_num_timesteps = EVENT_NUM_TIMESTEPS[event_idx]

        elif area_name == 'Chowilla':
            event_id = event_summary['Event'][event_idx]
            lf_run_name = event_summary['HEC_RAS_plan_LF'][event_idx]
            lf_filepath = f'{DATASET_PATH}/HD_model_data/Low-fidelity/Chow_LF_modelG.{lf_run_name}.hdf'
            hf_run_name = event_summary['HEC_RAS_plan_HF'][event_idx]
            hf_filepath = f'{DATASET_PATH}/HD_model_data/High-fidelity/Chow_HF.{hf_run_name}.hdf'
            event_num_timesteps = len(get_event_timesteps(lf_filepath)) # LF and HF have same num of timesteps

        elif area_name == 'BurnettRV':
            event_id = event_summary['No'][event_idx]
            lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
            lf_filepath = f'../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.{lf_run_name}.hdf'
            hf_run_name = event_summary['tuflow_evt_name_old'][event_idx]
            hf_filepath = f'../dataset/BurnettRV/HD_model_data/High-fidelity/Paradise_{hf_run_name}_002.npz'
            hf_data = np.load(hf_filepath)
            hf_timesteps = hf_data['time_data'].shape[0]
            event_num_timesteps = hf_timesteps - 48 # trim 48 timesteps from the start of each HF run

        else:
            assert False, f"Unknown area name {area_name}."

        event_ids.append(event_id)
        group_ids.append(group_id)
        lf_filepaths.append(lf_filepath)
        hf_filepaths.append(hf_filepath)
        num_timesteps.append(event_num_timesteps)

        print(f'Added event (event id {event_id}, group id {group_id}) to dataset. \
                Event has {event_num_timesteps} timesteps.')

    cat_data_path = os.path.join(DATASET_PATH, f'HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_{val_group}.npz')
    feature_stats_path = f'data/{area_name}/feature_stats_fold_{val_group}.npz'

    if area_name == 'Carlisle':
        hf_filetype = 'npz'
    elif area_name == 'Chowilla':
        hf_filetype = 'hdf'
    elif area_name == 'BurnettRV':
        hf_filetype = 'npz'
    else:
        assert False, f'HF filetype for area {area_name} is unknown.'

    dataset = FloodEventDataset(
        root_dir = f'data/{area_name}',
        area_name = area_name,
        lf_geometry_path = f'data/{area_name}/LF_geometry_data.npz',
        hf_geometry_path = f'data/{area_name}/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        hf_filetype = hf_filetype,
        cat_data_path = cat_data_path,
        feature_stats_path = feature_stats_path,
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = num_timesteps,
        previous_timesteps = 0
    )

    print(f'Made {area_name} dataset, contains total {len(dataset)} timesteps.')

    return dataset

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

def get_all_event_timesteps_chowilla():
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

def get_all_event_timesteps_burnett():
    # burnett has 2 more timesteps in LF than HF

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
    dataset = make_dataset(area_name='BurnettRV', val_group=1, mode='all')
    print('processing')
    dataset.process()
    # area_name = 'Carlisle'
    # geom = np.load(f'../dataset/BurnettRV/Geometry_data/Tuflow_Geometry_data.npz')
    # eof_hfdata = np.load(f'../dataset/BurnettRV/HF_EOF_analysis/EOFanalysis_HFdata_WSE_ValidateOnGrp_1.npz')
    # print(f'eof_hfdata {eof_hfdata.keys()}')
    # ecs = np.load(f'../dataset/BurnettRV/HF_EOF_analysis/ECs_HFdata_ValidateOnGrp_1.npz')
    # print(f'ecs {ecs.keys()}')
    # print(f'shape {ecs['HF_ECs_train_WSE'].shape}')
    # cats = np.load(f'../dataset/BurnettRV/HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_1.npz')
    # print(cats.keys())
    # print(cats['wet_idx'].shape)
    # hf_wd = np.load(f'data/{area_name}/processed/hf_water_depth_event_1.npy', mmap_mode='r')
    # print(f'HF WD {hf_wd.shape}')
    # lf_wd = np.load(f'data/{area_name}/processed/lf_dynamic_node_features_event_1.npy', mmap_mode='r')
    # print(f'LF WD {lf_wd.shape}')
    # ups_wd = np.load(f'data/{area_name}/processed/upsampled_water_depth_event_1.npy', mmap_mode='r')
    # print(f'UPS WD {ups_wd.shape}')
    # dataset.process() # if automatic processing doesn't happen
    # get_all_event_timesteps_burnett()
