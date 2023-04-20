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
import pickle


## ------- Custom dataset ------
class EddyDataset(Dataset):
    preloaded_folder = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/"
    def __init__(self, ssh_folder, eddies_folders, bbox=[18.125, 32.125, -100.125, -74.875], output_resolution=0.125,
                 days_before = 0, days_after = 0, transform=None, perc_files = 1, shuffle=False):
        print("@@@@@@@@@@@@@@@@@@@@@@@@ Initializing EddyDataset@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("SSH folder: ", ssh_folder)
        print("Eddies folder: ", eddies_folders)
        self.ssh_folder = ssh_folder
        self.eddies_folders = eddies_folders
        self.percentage = perc_files

        if type(eddies_folders) is list:
            self.eddies_paths = []
            for i, folder in enumerate(eddies_folders):
                files = np.array(os.listdir(folder))
                if i == 0:
                    self.eddies_paths = [join(folder, x) for x in files]
                else:
                    self.eddies_paths += [join(folder, x) for x in files]
        else:
            self.eddies_paths = [join(eddies_folders,x) for x in np.array(os.listdir(eddies_folders))]
        self.eddies_paths.sort()
        self.total_days = int(len(self.eddies_paths)*perc_files)
        self.days_before = days_before
        self.days_after = days_after

        if shuffle:
            np.random.shuffle(self.eddies_paths)

        self.eddies_paths = self.eddies_paths[:self.total_days]

        self.transform = transform
        self.bbox = bbox
        self.output_resolution = output_resolution
        self.save_preloaded = True

        # Verifying if the file is alraedy saved
        file_name = join(self.preloaded_folder, '_'.join([os.path.basename(x) for x in self.eddies_folders]))
        file_name += f"_perc{int(perc_files*100):02d}_preloaded_db_{days_before}_da_{days_after}.pkl"
        print(f"Searching preloaded file: {file_name}")
        if os.path.exists(file_name):
            print(f"Loading preloaded file: {file_name}")
            with open(file_name, 'rb') as file:
                self.data = pickle.load(file)
                self.save_preloaded = False
        else:
            self.data = [{"file": file, "data": None} for file in self.eddies_paths]

    def __len__(self):
        return self.total_days

    def __getitem__(self, idx):
        # Read eddies_masks
        if self.data[idx]["data"] is None:
            loaded_files = np.sum(np.array([1 if d["data"] is not None else 0 for d in self.data]))
            print(f"Loading countours {idx} {loaded_files}/{self.total_days} loaded) ...")
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
            eddies_masks = xr.open_dataset(self.eddies_paths[idx])['contours'].data
            eddies_masks = np.nan_to_num(eddies_masks.reshape(1, eddies_masks.shape[0], eddies_masks.shape[1]), 0)

            self.data[idx]["data"] = ssh_all[:, :, :].astype(np.float32), eddies_masks[:, :, :].astype(np.float32)

        loaded_files = np.sum(np.array([1 if d["data"] is not None else 0 for d in self.data]))
        if loaded_files == self.total_days and self.save_preloaded:
            print("All data loaded")
            root_folder = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/"
            file_name = join(root_folder, '_'.join([os.path.basename(x) for x in self.eddies_folders]))
            file_name += f"_perc{int(self.percentage* 100):02d}_preloaded_db_{self.days_before}_da_{self.days_after}.pkl"
            with open(file_name, 'wb') as file:
                pickle.dump(self.data, file)
            print("Done saving loaded data!")
            self.save_preloaded = False

        return self.data[idx]["file"],  self.data[idx]["data"]

## ----- DataLoader --------
if __name__ == "__main__":
    # filename = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_2010_coh_14days_contour_d7_PreprocContours_2015_coh_14days_contour_d7"
    # with open(filename, 'rb') as file:
    #     x = pickle.load(file)
    # print(x)
    # ----------- Skynet ------------
    data_folder = "/unity/f1/ozavala/DATA/GOFFISH"
    # folders = ['2010_coh_14days_contour_d7', '2020_coh_14days_contour_d7', '2020_coh_14days_contour_d0']
    # folders = ['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7']
    # folders = ['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7']
    # folders = ['2020_coh_14days_contour_d7']

    tested_folders = [ ['2010_coh_14days_contour_d7'],
                       ['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7'],
                       ['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7', '2018_coh_14days_contour_d7']]

    all_days_before = [0, 6]
    all_days_after = [0, 6]

    for days_before in all_days_before:
        for days_after in all_days_after:
            for folders in tested_folders:
                print(f"Testing {folders} with {days_before} days before and {days_after} days after")

                ssh_folder = join(data_folder, "AVISO")
                # eddies_folders = join(data_folder, f"EddyDetection/PreprocContours_{c_folder}")
                eddies_folders = [join(data_folder, f"EddyDetection/PreprocContours_{c_folder}") for c_folder in folders]

                bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
                output_resolution = 0.1
                # perc_files = 90/365
                perc_files = 1
                dataset = EddyDataset(ssh_folder, eddies_folders, bbox, output_resolution, days_before=days_before, days_after=days_after, perc_files=perc_files)
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