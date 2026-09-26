# Adapted from https://github.com/acostacos/dual_flood_gnn

import numpy as np
import os
import torch
import torch.nn.functional as F

from data.hecras_data_retrieval import get_event_timesteps, get_min_cell_elevation, get_water_level
from scipy.spatial import KDTree
from torch_geometric.data import Data, Dataset


class MultiResolutionData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lf_edge_index':
            return int(self.lf_num_nodes)
        elif key == 'hf_edge_index':
            return int(self.hf_num_nodes)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class FloodEventDataset(Dataset):
    """
    Static lf node features: area, elevation, roughness
    Dynamic lf node features: water depth
    Static lf edge features: edge length
    Target: residual water depth at hf nodes
    """

    def __init__(
        self,
        root_dir: str,
        area_name: str,
        lf_geometry_path: str,
        hf_geometry_path: str,
        lf_hecras_paths: list[str],
        hf_paths: list[str],
        hf_filetype: str, # file extension, either npz or hdf
        cat_data_path: str,
        feature_stats_path: str,
        event_ids: list,
        group_ids: list,
        num_timesteps: list, # number of timesteps per HF run
        previous_timesteps: int = 0 # number of timesteps to look back
        ):

        # filepaths
        self.root = root_dir
        self.area_name = area_name
        self.lf_geometry_path = lf_geometry_path
        self.hf_geometry_path = hf_geometry_path
        self.lf_hecras_paths = lf_hecras_paths
        self.hf_paths = hf_paths
        self.hf_filetype = hf_filetype

        # event data
        self.event_ids = event_ids
        self.group_ids = group_ids
        self.num_timesteps = num_timesteps

        # preset filenames to save into in processed dir
        self.STATIC_FEATURES_FILE = 'static_features.npz'
        self.DYNAMIC_FEATURES_FILES = [f'dynamic_values_event_{event_id}.npz' for event_id in self.event_ids]
        self.STATIC_FEATURES_PATH = os.path.join(self.processed_dir, self.STATIC_FEATURES_FILE)
        self.DYNAMIC_FEATURES_PATHS = [os.path.join(self.processed_dir, self.DYNAMIC_FEATURES_FILES[i]) \
                                       for i in range(len(self.DYNAMIC_FEATURES_FILES))]
        self.CAT_DATA_PATH = cat_data_path
        self.FEATURE_STATS_PATH = feature_stats_path

        # other settings, unused for now
        self.previous_timesteps = previous_timesteps # TODO timesteps to look back

        super().__init__(self.root, transform = None, pre_transform = None, pre_filter = None)

        # post-process call one-time loading of static features to reduce loading from disk
        static_features = np.load(self.STATIC_FEATURES_PATH, mmap_mode='r')

        self.lf_coords = torch.from_numpy(static_features['lf_coords']).float()
        self.lf_edge_index = torch.from_numpy(static_features['lf_edge_index']).long()
        self.lf_static_node_features = torch.from_numpy(static_features['lf_static_node_features']).float()
        self.lf_static_edge_features = torch.from_numpy(static_features['lf_static_edge_features']).float()

        self.hf_coords = torch.from_numpy(static_features['hf_coords']).float()
        self.hf_edge_index = torch.from_numpy(static_features['hf_edge_index']).long()
        self.hf_static_node_features = torch.from_numpy(static_features['hf_static_node_features']).float()
        self.hf_static_edge_features = torch.from_numpy(static_features['hf_static_edge_features']).float()

        # get wet cells
        cat_data = np.load(self.CAT_DATA_PATH, allow_pickle=True)
        self.wet_idx = torch.from_numpy(cat_data['wet_idx']).long()

        # normalise static features (geometry), assuming they never change across timesteps and between train/test splits
        EPS = 1e-8

        lf_node_mean = self.lf_static_node_features.mean(dim=0, keepdim=True)
        lf_node_std = self.lf_static_node_features.std(dim=0, keepdim=True)
        self.lf_static_node_features = (self.lf_static_node_features - lf_node_mean) / (lf_node_std + EPS)

        lf_edge_mean = self.lf_static_edge_features.mean(dim=0, keepdim=True)
        lf_edge_std = self.lf_static_edge_features.std(dim=0, keepdim=True)
        self.lf_static_edge_features = (self.lf_static_edge_features - lf_edge_mean) / (lf_edge_std + EPS)

        hf_node_mean = self.hf_static_node_features.mean(dim=0, keepdim=True)
        hf_node_std = self.hf_static_node_features.std(dim=0, keepdim=True)
        self.hf_static_node_features = (self.hf_static_node_features - hf_node_mean) / (hf_node_std + EPS)

        hf_edge_mean = self.hf_static_edge_features.mean(dim=0, keepdim=True)
        hf_edge_std = self.hf_static_edge_features.std(dim=0, keepdim=True)
        self.hf_static_edge_features = (self.hf_static_edge_features - hf_edge_mean) / (hf_edge_std + EPS)

    @property
    def raw_file_names(self):
        # pass filepaths into the constructor
        return []

    @property
    def processed_file_names(self):
        return [
            self.STATIC_FEATURES_FILE,
            *self.DYNAMIC_FEATURES_FILES
        ]

    def download(self):
        # no downloading
        pass

    def process(self):

        # ----- create static file -----
        lf_geom = self._get_lf_geometry()
        lf_static_node_features = self._get_lf_static_node_features()
        lf_static_edge_features = self._get_lf_static_edge_features()
        hf_geom = self._get_hf_geometry()
        hf_static_node_features = self._get_hf_static_node_features()
        hf_static_edge_features = self._get_hf_static_edge_features()
        
        save_path = os.path.join(self.processed_dir, self.STATIC_FEATURES_FILE)
        np.savez(save_path,
                 lf_coords = lf_geom['cell_coordinates'],
                 lf_edge_index = lf_geom['edge_index'],
                 lf_static_node_features = lf_static_node_features,
                 lf_static_edge_features = lf_static_edge_features,
                 hf_coords = hf_geom['cell_coordinates'],
                 hf_edge_index = hf_geom['edge_index'],
                 hf_static_node_features = hf_static_node_features,
                 hf_static_edge_features = hf_static_edge_features)
        print(f'Saved constant values to {save_path}')
        del lf_geom, lf_static_node_features, lf_static_edge_features, hf_geom, hf_static_node_features, hf_static_edge_features

        # ----- create dynamic files -----
        for i, event_id in enumerate(self.event_ids):
            lf_dynamic_node_features = self._get_lf_dynamic_node_features(i)
            hf_water_depth, upsampled_water_depth = self._get_hf_dynamic_data(i)

            save_path = os.path.join(self.processed_dir, self.DYNAMIC_FEATURES_FILES[i])

            np.savez(save_path,
                     lf_dynamic_node_features = lf_dynamic_node_features,
                     hf_water_depth = hf_water_depth,
                     upsampled_water_depth = upsampled_water_depth)
            print(f'Saved dynamic values for event {event_id} to {save_path}')
            del lf_dynamic_node_features, hf_water_depth, upsampled_water_depth

    def len(self):
        return sum(self.num_timesteps)

    def get(self, idx):

        # find the event this index belongs to
        event_idx = -1
        timestep_in_event = -1
        timesteps_counter = 0
        for i, event_num_timesteps in enumerate(self.num_timesteps):
            if timesteps_counter + event_num_timesteps > idx:
                event_idx = i
                timestep_in_event = idx - timesteps_counter
                break
            timesteps_counter += event_num_timesteps
        assert 0 <= event_idx < len(self.DYNAMIC_FEATURES_PATHS), f"Error: event index {event_idx} is out of bounds."
        assert 0 <= timestep_in_event < self.num_timesteps[event_idx], \
            f"Error: timestep {timestep_in_event} out of bounds for event with {self.num_timesteps[event_idx]} timesteps."

        # load dynamic data
        dynamic_values = np.load(self.DYNAMIC_FEATURES_PATHS[event_idx], mmap_mode='r')
        lf_dynamic_node_features = dynamic_values['lf_dynamic_node_features'][timestep_in_event]
        lf_dynamic_node_features = torch.from_numpy(lf_dynamic_node_features).float().unsqueeze(-1)
        hf_water_depth = dynamic_values['hf_water_depth'][timestep_in_event]
        hf_water_depth = torch.from_numpy(hf_water_depth).float().unsqueeze(-1)
        upsampled_water_depth = dynamic_values['upsampled_water_depth'][timestep_in_event]
        upsampled_water_depth = torch.from_numpy(upsampled_water_depth).float().unsqueeze(-1)

        feature_stats = np.load(self.FEATURE_STATS_PATH)
        lf_dyn_mean = feature_stats['lf_water_depth_mean']
        lf_dyn_std = feature_stats['lf_water_depth_std']
        lf_dynamic_node_features = (lf_dynamic_node_features - lf_dyn_mean) / lf_dyn_std

        lf_x = torch.cat([self.lf_static_node_features, lf_dynamic_node_features], dim=-1)
        hf_x = self.hf_static_node_features # no hf dynamic node features are given

        data = MultiResolutionData(
            lf_x = lf_x,
            lf_coords = self.lf_coords,
            lf_edge_index = self.lf_edge_index,
            lf_edge_attr = self.lf_static_edge_features,
            hf_x = hf_x,
            hf_coords = self.hf_coords,
            hf_edge_index = self.hf_edge_index,
            hf_edge_attr = self.hf_static_edge_features,
            hf_water_depth = hf_water_depth, 
            upsampled_water_depth = upsampled_water_depth, # to use in reconstruction after predicting the residual

            # for PyG
            num_nodes = hf_x.size(0),
            lf_num_nodes = lf_x.size(0),
            hf_num_nodes = hf_x.size(0)
        )
        
        return data

    # ---------- other methods ---------

    def _get_num_timesteps(self, event_idx):
        return self.num_timesteps[event_idx]

    def _get_lf_geometry(self):
        return np.load(self.lf_geometry_path, mmap_mode='r')

    def _get_lf_coords(self):
        lf_geom = self._get_lf_geometry()
        return lf_geom['cell_coordinates']

    def _get_lf_edge_index(self):
        lf_geom = self._get_lf_geometry()
        return lf_geom['edge_index']

    def _get_lf_static_node_features(self):
        # want area, elevation, roughness
        lf_geom = self._get_lf_geometry()
        lf_area = lf_geom['cell_area']
        lf_elevation = lf_geom['cell_elevation']
        lf_roughness = lf_geom['cell_roughness']

        static_node_features = np.array([lf_area, lf_elevation, lf_roughness]).transpose()
        return static_node_features

    def _get_hf_static_node_features(self):
        # we are only provided area and elevation
        hf_geom = self._get_hf_geometry()
        hf_area = hf_geom['cell_area']
        hf_elevation = hf_geom['cell_elevation']

        static_node_features = np.array([hf_area, hf_elevation]).transpose()
        return static_node_features

    def _get_lf_static_edge_features(self):
        lf_geom = self._get_lf_geometry()
        lf_coords = lf_geom['cell_coordinates']
        lf_edge_index = lf_geom['edge_index']
        n1_coords, n2_coords = lf_coords[lf_edge_index[0]], lf_coords[lf_edge_index[1]]
        lf_edge_lengths = np.linalg.norm(n1_coords - n2_coords, axis = 1)

        static_edge_features = np.array([lf_edge_lengths]).transpose()
        return static_edge_features

    def _get_hf_static_edge_features(self):
        hf_geom = self._get_hf_geometry()
        hf_coords = hf_geom['cell_coordinates']
        hf_edge_index = hf_geom['edge_index']
        n1_coords, n2_coords = hf_coords[hf_edge_index[0]], hf_coords[hf_edge_index[1]]
        hf_edge_lengths = np.linalg.norm(n1_coords - n2_coords, axis = 1)

        static_edge_features = np.array([hf_edge_lengths]).transpose()
        return static_edge_features

    def _get_lf_dynamic_node_features(self, event_idx: int):
        print(f'Getting LF dynamic node features for event index {event_idx}.')
        lf_path = self.lf_hecras_paths[event_idx]

        # get ghost cell indices to filter out, assumed to be consistent across runs
        lf_elevation = get_min_cell_elevation(lf_path, self.area_name)
        ghost_idx = np.where(np.isnan(lf_elevation))[0]
            
        # check LF timesteps against expected timesteps (HF + lookback)
        lf_timesteps = len(get_event_timesteps(lf_path))
        hf_timesteps = self.num_timesteps[event_idx]
        ignored_timesteps = lf_timesteps - hf_timesteps - self.previous_timesteps

        lf_water_level = get_water_level(lf_path, self.area_name)
        lf_water_depth = np.maximum(0.0, lf_water_level - lf_elevation)
        lf_water_depth = np.delete(lf_water_depth, ghost_idx, axis=1)
        lf_water_depth = lf_water_depth[ignored_timesteps:, :]

        return lf_water_depth

    def _get_hf_geometry(self):
        return np.load(self.hf_geometry_path, mmap_mode='r')

    def _get_hf_coords(self):
        hf_geom = self._get_hf_geometry()
        return hf_geom['cell_coordinates']

    def _get_hf_edge_index(self):
        hf_geom = self._get_hf_geometry()
        return hf_geom['edge_index']

    def _get_hf_water_depth_npz(self, event_idx: int):
        print(f'Getting HF water depth from npz for event index {event_idx}.')
        # expects npz to be preprocessed to remove ghost cells
        hf_path = self.hf_paths[event_idx]
        hf_data = np.load(hf_path)
        hf_water_level = hf_data['wse_data'] # wse_data for Carlisle
        hf_geom = self._get_hf_geometry()
        hf_elevation = hf_geom['cell_elevation']
        hf_water_depth = np.maximum(0.0, hf_water_level - hf_elevation)
        return hf_water_depth

    def _get_hf_water_depth_hecras(self, event_idx: int):
        print(f'Getting HF water depth from hdf for event index {event_idx}.')
        hf_path = self.hf_paths[event_idx]

        # get ghost cell indices to filter out
        hf_elevation = get_min_cell_elevation(hf_path, self.area_name)
        ghost_idx = np.where(np.isnan(hf_elevation))[0]

        hf_water_level = get_water_level(hf_path, self.area_name)
        hf_water_depth = np.maximum(0.0, hf_water_level - hf_elevation)
        hf_water_depth = np.delete(hf_water_depth, ghost_idx, axis=1)
        return hf_water_depth

    def _get_hf_dynamic_data(self, event_idx: int):
        print(f'Getting HF dynamic data for event index {event_idx}.')
        # get lf water surface and upscale
        lf_path = self.lf_hecras_paths[event_idx]

        # get ghost cell indices to filter out
        lf_elevation = get_min_cell_elevation(lf_path, self.area_name)
        ghost_idx = np.where(np.isnan(lf_elevation))[0]

        # check LF timesteps against expected timesteps (HF + lookback)
        lf_timesteps = len(get_event_timesteps(lf_path))
        hf_timesteps = self.num_timesteps[event_idx]
        ignored_timesteps = lf_timesteps - hf_timesteps

        lf_water_level = get_water_level(lf_path, self.area_name)
        lf_water_level = np.delete(lf_water_level, ghost_idx, axis=1)
        lf_water_level = lf_water_level[ignored_timesteps:, :]

        if self.hf_filetype == 'npz':
            hf_water_depth = self._get_hf_water_depth_npz(event_idx)
        elif self.hf_filetype == 'hdf':
            hf_water_depth = self._get_hf_water_depth_hecras(event_idx)
        else:
            assert False, f'Unsupported HF filetype {self.hf_filetype}.'

        # ---------- upscale LF to HF ----------

        # load geometries (cell coordinates)
        # assume that ghost cells have been removed as part of preprocessing
        lf_geom = self._get_lf_geometry()
        lf_coords = lf_geom['cell_coordinates']
        hf_geom = self._get_hf_geometry()
        hf_coords = hf_geom['cell_coordinates']
        hf_elevation = hf_geom['cell_elevation']

        # upsampling
        lf_kdtree = KDTree(lf_coords)
        dists, nearest_lf_indices = lf_kdtree.query(hf_coords)
        lf_water_level_upscaled = lf_water_level[:, nearest_lf_indices]

        upsampled_water_depth = lf_water_level_upscaled - hf_elevation

        return hf_water_depth, upsampled_water_depth
