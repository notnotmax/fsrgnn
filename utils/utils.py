import numpy as np

def filter_dry_areas(data, elevation, dry_threshold=0.03, lower_dry_ele=0):
    # Filter data, so areas where the LF interpolated Water surface elevation (WSE) is
    # below the HF terrain elevation is set as dry.
    data_filtered = np.where(data > elevation+dry_threshold,
                                          data, elevation - lower_dry_ele)
    return data_filtered