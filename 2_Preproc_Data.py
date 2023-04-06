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
from proc_utils.geometries import intersect_polygon_grid
# Local
from proj_io.contours import read_contours_mask_and_polygons

# SSH file used as reference for the size of the grid
# ssh_file = "/data/GOFFISH/AVISO/2010-01.nc"
ssh_file = "/Net/work/ozavala/DATA/GOFFISH/AVISO/2010-01.nc"  # Just a test file to read the grid
day_of_month = 5
bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used by DataReader
output_resolution = 0.1  # Needs to be the same used in DataReader
#%%
ds = xr.open_dataset(ssh_file)
ds = ds.sel(latitude=slice(bbox[0], bbox[1]), longitude=slice(bbox[2], bbox[3]))
lats_orig = ds.latitude.values
lons_orig = ds.longitude.values
lats = np.linspace(np.amin(lats_orig), np.amax(lats_orig), int((np.amax(lats_orig) - np.amin(lats_orig)) / output_resolution))
lons = np.linspace(np.amin(lons_orig), np.amax(lons_orig), int((np.amax(lons_orig) - np.amin(lons_orig)) / output_resolution))
ds = ds.interp(latitude=lats, longitude=lons)  # Increase output_resolution

if np.amax(lons) > 180:
    lons = lons - 360
viz_obj = EOAImageVisualizer(lats=lats, lons=lons)

def preproc_data(proc_id):
    '''It generates a mask of the contours and saves it to a file'''
    # input_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/raw"
    # output_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/processed"
    input_folder = "/nexsan/people/lhiron/UGOS/Lag_coh_eddies/eddy_contours/altimetry_2010_14day_coh/"
    output_folder = "/Net/work/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours/"
    create_folder(output_folder)
    all_files = os.listdir(input_folder)
    all_files.sort()
    for i, c_file in enumerate(all_files):
        if i % NUM_PROC == proc_id:
            file_name = join(input_folder, c_file)
            print(f"Processing file {file_name}")
            # ------ Reading the contours and generating a mask for eddies
            contours_mask = np.zeros_like(ds['adt'][day_of_month, :, :])
            all_contours_polygons, contours_mask = read_contours_mask_and_polygons(file_name, contours_mask, lats, lons)
            ds_contours = xr.Dataset({"contours": (("latitude", "longitude"), contours_mask), },
                                     {"latitude": lats, "longitude": lons})

            # Plotting
            # viz_obj.__setattr__('additional_polygons', [Polygon(x) for x  in all_contours_polygons])
            # viz_obj.plot_3d_data_npdict(ds, var_names=['adt'], z_levels=[0], title='SSH', file_name_prefix='')
            # viz_obj.plot_2d_data_xr(ds_contours,  var_names=['contours'], title='Contours initialized')
            # Save to netcdf
            ds_contours.to_netcdf(join(output_folder, c_file.replace('.mat', '.nc')))


if __name__ == '__main__':
    # ----------- Parallel -------
    NUM_PROC = 20
    p = Pool(NUM_PROC)
    p.map(preproc_data, range(NUM_PROC))