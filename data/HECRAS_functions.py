# adapted from https://doi.org/10.26188/24312658

"""
Function used when working with HEC-RAS data

@author: nfraehr
"""

# %% Import libraries
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import re

# %% Functions

def filter_dry_areas(data, elevation, dry_threshold=0.03, lower_dry_ele=0):
    # Filter data, so areas where the LF interpolated Water surface elevation (WSE) is
    # below the HF terrain elevation is set as dry.
    data_filtered = np.where(data > elevation+dry_threshold,
                                          data, elevation - lower_dry_ele)
    return data_filtered

def convert_hecras_timedata(time_data):
    time_series = []
    for tstep in time_data:
        time_str = tstep.decode('UTF-8')  # Convert byte data to str
        time_stamp = datetime.strptime(time_str, '%d%b%Y %H:%M:%S')  # Convert to datetime
        time_series.append(time_stamp)  # Append timestep to timeseries

    time = np.array(time_series)  # Convert to numpy array
    return time

def get_timedata(path):
    hec = h5py.File(path, 'r')  # Open to read data
    # Get time data
    time_data = hec['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series'] \
        ['Time Date Stamp']

    time = convert_hecras_timedata(time_data)

    hec.close()  # Close

    return time

def get_2d_hecras_geometry_data(path, flowareaname, incl_facedataforvelocity=False):
    hec = h5py.File(path, 'r')  # Open to read data
    # list(hec.keys()) to check what the options are
    coordinates = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Cells Center Coordinate'])
    elevation = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Cells Minimum Elevation'])
    area = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Cells Surface Area'])
    facepoint_XY_coor = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['FacePoints Coordinate'])
    cell_facepoint_idx = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Cells FacePoint Indexes'])
    perimeter_XY_coor = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Perimeter'])
    
    if incl_facedataforvelocity:
        
        faces_cell_idx = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Faces Cell Indexes'])
        faces_norm_unit_vector = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Faces NormalUnitVector and Length'])
        faces_facepoint_idx = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Faces FacePoint Indexes'])
        hec.close()  # Close
        return coordinates, elevation, area, facepoint_XY_coor, cell_facepoint_idx, perimeter_XY_coor, faces_cell_idx, faces_norm_unit_vector, faces_facepoint_idx
    
    else:
        hec.close()  # Close
        return coordinates, elevation, area, facepoint_XY_coor, cell_facepoint_idx, perimeter_XY_coor



def get_2d_hecras_results(path, flowareaname, warmupperiod="10 days", remove_ghost_cells=False):
    hec = h5py.File(path, 'r')  # Open to read data

    # Get water surface data
    data_ws = np.array(hec['Results']['Unsteady']['Output']['Output Blocks']['Base Output'] \
                       ['Unsteady Time Series']['2D Flow Areas'][flowareaname]['Water Surface'])

    # Get time data
    time = get_timedata(path)

    # Remove warmup period
    if warmupperiod is not None:
        filter_ = time > (time[0] + pd.Timedelta(warmupperiod))
        time = time[filter_]
        data_ws = data_ws[filter_, :]

    if remove_ghost_cells:
        elevation = np.array(hec['Geometry']['2D Flow Areas'][flowareaname]['Cells Minimum Elevation'])
        Ghost_cells_idx = np.where(np.isnan(elevation))[0]
        # Remove ghost cells
        data_ws = np.delete(data_ws, Ghost_cells_idx, axis=1)

    data_ws = np.round(data_ws, 3)
    data_ws = np.float32(data_ws)
    
    hec.close()  # Close

    return data_ws, time


def get_CPU_time(path):
    hec = h5py.File(path, 'r')  # Open to read data

    # Get computational time
    cpu_time_all = np.array(hec['Results']['Summary']['Compute Processes']['Compute Time (ms)'])
    cpu_time_all = cpu_time_all / 1000
    hec.close()  # Close

    return cpu_time_all.max()
    