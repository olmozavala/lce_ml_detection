# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation

import sys
sys.path.append("eoas_pyutils")
sys.path.append("ai_common")
import time

from datetime import datetime, timedelta

import os
import xarray as xr
from os.path import join
from torch.utils.data import Dataset, DataLoader
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from proj_io.ssh_io import read_ssh_by_date
import numpy as np
import pickle

## ------- Custom dataset ------
class EddyDataset(Dataset):

    def __init__(self, ssh_folder, eddies_folder, bbox=[18.125, 32.125, -100.125, -74.875], output_resolution=0.125,
                 days_before = 0, days_after = 0, transform=None):

        print("@@@@@@@@@@@@@@@@@@@@@@@@ Initializing EddyDataset @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("SSH folder: ", ssh_folder)
        print("Eddies folder: ", eddies_folder)
        self.ssh_folder = ssh_folder
        self.eddies_folder = eddies_folder

        self.eddies_paths = [file for file in os.listdir(eddies_folder) if file.endswith(".nc")]
        self.eddies_paths.sort()
        self.total_days = int(len(self.eddies_paths))
        self.days_before = days_before
        self.days_after = days_after

        self.transform = transform
        self.bbox = bbox
        self.output_resolution = output_resolution

        # Verifying if the file is already saved
        file_name = join(self.eddies_folder, f"preloaded_db_{days_before}_da_{days_after}.pkl")
        self.preload_file = file_name
        print(f"Searching preloaded file: {file_name}")
        if os.path.exists(file_name):
            print(f"Loading preloaded file: {file_name}")
            with open(file_name, 'rb') as file:
                self.data = pickle.load(file)
        else:
            print(f"File not found, preloading data...")
            self.data = [{"file": file, "data": None} for file in self.eddies_paths]
            self.__preload__()

        # Verify all the indexes have data and remove the Nones
        self.data = [d for d in self.data if d["data"] is not None]
        self.total_days = len(self.data)

    def __preload__(self):
        start_time = time.time()
        for idx in range(self.total_days):
            try: 
                loaded_files = np.sum(np.array([1 if d["data"] is not None else 0 for d in self.data]))
                if idx % 100 == 0:
                    print(f"Loading countours {idx}, {loaded_files}/{self.total_days} loaded Time to load (100): {(time.time() - start_time)} ...")
                    start_time = time.time()

                eddies_file = os.path.basename(self.eddies_paths[idx])
                day_year = int(eddies_file.split("_")[4].split(".")[0])
                year = int(eddies_file.split("_")[2])
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
                eddies_masks = xr.open_dataset(os.path.join(self.eddies_folder,self.eddies_paths[idx]))['contours'].data
                eddies_masks = np.nan_to_num(eddies_masks.reshape(1, eddies_masks.shape[0], eddies_masks.shape[1]), 0)

                self.data[idx]["data"] = ssh_all[:, :, :].astype(np.float32), eddies_masks[:, :, :].astype(np.float32)
            except Exception as e:
                print(f"Error reading file {idx}, {self.eddies_paths[idx]}: {e}")
                # Remove this file from the list

                continue

        # Check all the files at self.data are loaded

        print("All data loaded")
        with open(self.preload_file, 'wb') as file:
            pickle.dump(self.data, file)
        print("Done saving loaded data!")

    def __len__(self):
        return self.total_days

    def __getitem__(self, idx):
        while self.data[idx]["data"] is None:
            idx = (idx + 1) % self.total_days
        return self.data[idx]["file"],  self.data[idx]["data"]


## ----- DataLoader --------
if __name__ == "__main__":
    # filename = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_2010_coh_14days_contour_d7_PreprocContours_2015_coh_14days_contour_d7"
    # with open(filename, 'rb') as file:
    #     x = pickle.load(file)
    # print(x)
    # ----------- Skynet ------------
    data_folder = "/unity/f1/ozavala/DATA/"
    preproc_eddies_folder = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours2024/"

    tested_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]

    all_days_before = [0, 6]
    all_days_after = [0, 6]

    for days_before in all_days_before:
        for days_after in all_days_after:
            for c_folder in tested_folders:
                print(f"Testing {c_folder} with {days_before} days before and {days_after} days after")

                ssh_folder = join(data_folder, "GOFFISH/AVISO/GoM")
                eddies_folder = join(preproc_eddies_folder, c_folder)

                bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
                output_resolution = 0.1
                dataset = EddyDataset(ssh_folder, eddies_folder, bbox, output_resolution, days_before=days_before, days_after=days_after)
                # With workers I can't store all the data in memory
                # workers = 30
                # myloader = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=workers)

                # Without workers works as expected
                myloader = DataLoader(dataset, batch_size=80, shuffle=True)
                # --------- Just reading some lats lons ----------
                for epoch in range(1):
                    print(f"###################### Epoch {epoch} ######################")
                    # for batch_idx, (data, target) in enumerate(myloader):
                    for file, (data, target) in myloader:
                        print(f"------------------- New Batch -------------------")
                        print(f"File: {file}")