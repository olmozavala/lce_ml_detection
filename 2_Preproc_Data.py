# %% External
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
from proj_io.ssh_io import largest_number_divisible_by_8
from proc_utils.geometries import intersect_polygon_grid
# Local
from proj_io.contours import read_contours_mask_and_polygons

# SSH file used as reference for the size of the grid
ssh_file = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM//2010-01.nc"  # Just a test file to read the grid
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
lats = lats[:largest_number_divisible_by_8(lats.shape[0])]
lons = lons[:largest_number_divisible_by_8(lons.shape[0])]

ds = ds.interp(latitude=lats, longitude=lons)  # Increase output_resolution

if np.amax(lons) > 180:
    lons = lons - 360
viz_obj = EOAImageVisualizer(lats=lats, lons=lons)

def preproc_data(proc_id, input_folder, output_folder):
    '''It generates a mask of the contours and saves it to a file'''
    print(f'Processing folder {input_folder} with proc_id {proc_id}')
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

# %%
# ----------- Parallel -------
root_folder = "/nexsan/people/lhiron/UGOS/Lagrangian_eddy_altimetry/eddy_contours/"
output_folder = f"/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022/"
# folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]
folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [20, 30]]

NUM_PROC = 30
for c_folder in folders:
    output_folder = join(output_folder, c_folder)
    create_folder(output_folder)
    p = Pool(NUM_PROC)
    params = [(i, join(root_folder, c_folder), output_folder) for i in range(NUM_PROC)]
    p.starmap(preproc_data, params)
    p.close()