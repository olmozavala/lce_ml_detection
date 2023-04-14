# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
from datetime import datetime, timedelta

import torch
import os
import xarray as xr
from os.path import join
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Polygon
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from viz_utils.eoa_viz import EOAImageVisualizer
from proj_io.contours import read_contours_polygons
from proj_io.ssh_io import read_ssh_by_date
import numpy as np


## ------- Custom dataset ------
class EddyDataset(Dataset):
    def __init__(self, ssh_folder, eddies_folder, bbox=[18.125, 32.125, -100.125, -74.875], output_resolution=0.125,
                 days_before = 0, days_after = 0, transform=None, perc_files = 1, shuffle=False):
        print("@@@@@@@@@@@@@@@@@@@@@@@@ Initializing EddyDataset@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("SSH folder: ", ssh_folder)
        print("Eddies folder: ", eddies_folder)
        self.ssh_folder = ssh_folder
        self.eddies_folder = eddies_folder
        self.eddies_files = np.array(os.listdir(eddies_folder))
        self.eddies_files.sort()
        self.total_days = int(len(self.eddies_files)*perc_files)
        self.days_before = days_before
        self.days_after = days_after

        if shuffle:
            np.random.shuffle(self.eddies_files)

        self.eddies_files = self.eddies_files[:self.total_days]

        self.transform = transform
        self.bbox = bbox
        self.output_resolution = output_resolution
        self.data = [{"file": file, "data": None} for file in self.eddies_files]

    def __len__(self):
        return self.total_days


    def __getitem__(self, idx):
        # Read eddies
        if self.data[idx]["data"] is None:
            loaded_files = np.sum(np.array([1 if d["data"] is not None else 0 for d in self.data]))
            print(f"Loading countours {idx} {loaded_files}/{self.total_days} loaded) ...")
            eddies_file = join(self.eddies_folder, self.eddies_files[idx])
            day_year = int(self.eddies_files[idx].split("_")[4].split(".")[0])
            year = int(self.eddies_files[idx].split("_")[2])
            month, day_month = get_month_and_day_of_month_from_day_of_year(day_year)
            start_date = datetime.strptime(f"{year}-{month}-{day_month}", "%Y-%m-%d")
            # Reading current day
            c_ssh, _, _ = read_ssh_by_date(start_date, self.ssh_folder, self.bbox, self.output_resolution)
            c_ssh = np.nan_to_num(c_ssh.reshape(1, c_ssh.shape[0], c_ssh.shape[1]), 0)
            ssh_all = c_ssh
            # Reading days before
            for i in range(self.days_before):
                c_date = start_date - timedelta(days=i)
                c_ssh, _, _ = read_ssh_by_date(c_date, self.ssh_folder, self.bbox, self.output_resolution)
                c_ssh = np.nan_to_num(c_ssh.reshape(1, c_ssh.shape[0], c_ssh.shape[1]), 0)
                # Concatenate previous days
                ssh_all = np.concatenate((ssh_all, c_ssh), axis=0)

            # Reading days after
            for i in range(self.days_after):
                c_date = start_date - timedelta(days=i)
                c_ssh, _, _ = read_ssh_by_date(c_date, self.ssh_folder, self.bbox, self.output_resolution)
                c_ssh = np.nan_to_num(c_ssh.reshape(1, c_ssh.shape[0], c_ssh.shape[1]), 0)
                # Concatenate previous days
                ssh_all = np.concatenate((ssh_all, c_ssh), axis=0)

            # Reading target contours
            eddies = xr.open_dataset(eddies_file)['contours'].data

            # return ssh_day, eddies
            eddies = np.nan_to_num(eddies.reshape(1, eddies.shape[0], eddies.shape[1]), 0)
            self.data[idx]["data"] = ssh_all[:, :, :].astype(np.float32), eddies[:, :, :].astype(np.float32)
        return self.data[idx]["file"],  self.data[idx]["data"]

## ----- DataLoader --------
if __name__ == "__main__":
    # ----------- Skynet ------------
    data_folder = "/unity/f1/ozavala/DATA/GOFFISH"
    folders = ['2010_coh_14days_contour_d7', '2020_coh_14days_contour_d7', '2020_coh_14days_contour_d0']
    c_folder = folders[0]

    ssh_folder = join(data_folder, "AVISO")
    eddies_folder = join(data_folder, f"EddyDetection/PreprocContours_{c_folder}")

    bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
    output_resolution = 0.1
    dataset = EddyDataset(ssh_folder, eddies_folder, bbox, output_resolution)

    # With workers I can't store all the data in memory
    # workers = 30
    # myloader = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=workers)

    # Without workers works as expected
    myloader = DataLoader(dataset, batch_size=80, shuffle=True)
    # --------- Just reading some lats lons ----------
    for epoch in range(10):
        print(f"###################### Epoch {epoch} ######################")
        # for batch_idx, (data, target) in enumerate(myloader):
        for file, (data, target) in myloader:
            print(f"------------------- New Batch -------------------")
            print(f"File: {file}")