# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import os
import xarray as xr
from os.path import join
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Polygon
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from viz_utils.eoa_viz import EOAImageVisualizer
from proj_io.contours import read_contours_polygons
import numpy as np


## ------- Custom dataset ------
class EddyDataset(Dataset):
    def __init__(self, ssh_folder, eddies_folder, bbox=[18.125, 32.125, -100.125, -74.875], output_resolution=0.125,
                 transform=None):
        print("Initializing EddyDataset")
        print("SSH folder: ", ssh_folder)
        print("Eddies folder: ", eddies_folder)
        self.ssh_folder = ssh_folder
        self.eddies_folder = eddies_folder
        self.eddies_files = np.array(os.listdir(eddies_folder))

        self.total_days = len(self.eddies_files)
        self.transform = transform
        self.bbox = bbox
        self.output_resolution = output_resolution

    def __len__(self):
        return self.total_days

    def __getitem__(self, idx):
        # Read eddies
        eddies_file = join(self.eddies_folder, self.eddies_files[idx])
        day_year = int(self.eddies_files[idx].split("_")[4].split(".")[0])
        year = int(self.eddies_files[idx].split("_")[2])
        month, day_month = get_month_and_day_of_month_from_day_of_year(day_year)
        # Read SSH
        ssh_file = join(self.ssh_folder, f"{year}-{month:02d}.nc")
        ssh_month = xr.open_dataset(ssh_file)
        ssh_month = ssh_month.sel(latitude=slice(self.bbox[0], self.bbox[1]),
                                  longitude=slice(self.bbox[2], self.bbox[3]))

        # Interpolate to output resolution
        lats_orig = ssh_month.latitude.values
        lons_orig = ssh_month.longitude.values
        lats = np.linspace(np.amin(lats_orig), np.amax(lats_orig),
                           int((np.amax(lats_orig) - np.amin(lats_orig)) / self.output_resolution))
        lons = np.linspace(np.amin(lons_orig), np.amax(lons_orig),
                           int((np.amax(lons_orig) - np.amin(lons_orig)) / self.output_resolution))

        ssh_month = ssh_month.interp(latitude=lats, longitude=lons)  # Increase output_resolution
        ssh_day = ssh_month['adt'][day_month-1, :, :].data
        # Read eddies
        eddies = xr.open_dataset(eddies_file)['contours'].data

        # ------------------ Just for visualization -------------------
        # viz_obj = EOAImageVisualizer(lats=lats, lons=lons, eoas_pyutils_path="../eoas_pyutils", )
        # input_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/raw"
        # c_file = f"eddies_altimetry_{year}_7days_{day_year}.mat"
        # file_name = join(input_folder, c_file)
        # print(f"Processing file {file_name}")
        # import h5py
        # all_contours_polygons = read_contours_polygons(file_name)
        # viz_obj.__setattr__('additional_polygons', [Polygon(x) for x  in all_contours_polygons])
        # viz_obj.plot_2d_data_xr({'ssh': ssh_day, 'eddies': eddies}, var_names=['ssh', 'eddies'],
        #                         title=f"{year}-{month:02d}-{day_month:02d}")

        # return ssh_day, eddies
        ssh_day = np.flip(np.nan_to_num(ssh_day.reshape(1, ssh_day.shape[0], ssh_day.shape[1]), 0), axis=1)
        eddies = np.flip(np.nan_to_num(eddies.reshape(1, eddies.shape[0], eddies.shape[1]), 0), axis=1)
        return ssh_day[:, :136, :].astype(np.float32), eddies[:, :136, :].astype(np.float32)

## ----- DataLoader --------
if __name__ == "__main__":
    # ----------- Skynet ------------
    ssh_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/"
    eddies_folder = "/nexsan/people/lhiron/UGOS/Lag_coh_eddies/eddy_contours/altimetry_2010_14day_coh/"
    # ----------- Local ------------
    # ssh_folder = "/data/LOCAL_GOFFISH/AVISO/"
    # eddies_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/processed/"

    bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
    output_resolution = 0.1
    dataset = EddyDataset(ssh_folder, eddies_folder, bbox, output_resolution)

    myloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # --------- Just reading some lats lons ----------
    for batch in myloader:
        print(batch[0].shape)
        ssh, eddies = batch
        x = 1