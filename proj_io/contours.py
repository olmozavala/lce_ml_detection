# External
import sys
from datetime import datetime
import xarray as xr
import numpy as np
from multiprocessing import Pool
from shapely.geometry import Polygon
import os
from os.path import join
import h5py
# EOAS Library
sys.path.append("eoas_pyutils")
from io_utils.io_common import create_folder
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import PlotMode
from proc_utils.geometries import intersect_polygon_grid

import h5py

def read_contours_polygons(file_name):
    '''
    It reads the eddy contours from a file
    :param file_name:
    :return:
    '''
    contours = h5py.File(file_name, 'r')
    num_contours = len([x for x in contours.keys() if x.startswith('xb0')])

    all_contours_polygons = []
    for i in range(1, num_contours + 1):
        cont_lons = contours[f'xb0_{i:03d}'][0, :] - 360
        cont_lats = contours[f'yb0_{i:03d}'][0, :]

        geom_poly = [(cont_lons[i], cont_lats[i]) for i in range(len(cont_lons))]
        all_contours_polygons.append(geom_poly)
    return all_contours_polygons

def read_contours_mask_and_polygons(file_name, contours_mask, lats, lons):
    '''
    It reads the eddy contours from a file and generates a mask of the contours
    :param file_name:
    :param contours_mask:
    :param lats:
    :param lons:
    :return:
    '''
    all_contours_polygons = read_contours_polygons(file_name)
    for i in range(len(all_contours_polygons)):
        intersect_polygon_grid(contours_mask, lats, lons, all_contours_polygons[i], 1).data

    return all_contours_polygons, contours_mask

