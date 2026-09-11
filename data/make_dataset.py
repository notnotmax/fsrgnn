import numpy as np
import os.path as osp
import pandas as pd

from data.dataset import FloodEventDataset
from data.hecras_data_retrieval import get_event_timesteps

def make_carlisle_dataset():

    print("Making Carlisle dataset.")
    DATASET_PATH = '../dataset/Carlisle'
    EVENT_SUMMARY_PATH = osp.join(DATASET_PATH, 'Carlisle_event_summary.csv')
    event_df = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = 1 # len(event_df)
    print(f"Number of events: {num_events}")
    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    num_timesteps = []

    for i in range(num_events):

        event_ids.append(event_df['No'][i])
        group_ids.append(event_df['Group'][i])

        lf_run_name = event_df['HEC_RAS_plan'][i]
        lf_filepath = osp.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf')
        lf_filepaths.append(lf_filepath)

        hf_run_name = event_df['Lisflood'][i]
        hf_filepath = osp.join(DATASET_PATH, f'HD_model_data/High-fidelity/{hf_run_name}_alltimesteps.npz')
        hf_filepaths.append(hf_filepath)

        hf_data = np.load(hf_filepath)
        hf_wse = hf_data['wse_data']
        num_timesteps.append(hf_wse.shape[0])
        del hf_wse

        print(f'Added details for event {i+1}.')
    
    # triggers the preprocessing
    dataset = FloodEventDataset(
        root_dir = 'data/Carlisle',
        area_name = 'Carlisle',
        lf_geometry_path = 'data/Carlisle/LF_geometry_data.npz',
        hf_geometry_path = 'data/Carlisle/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        hf_filetype = 'npz',
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = num_timesteps,
        previous_timesteps = 0
    )

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
    make_carlisle_dataset()
