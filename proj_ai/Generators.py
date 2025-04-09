# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation

import sys
sys.path.append("/unity/f1/ozavala/CODE/lce_ml_detection/ai_common")
sys.path.append("/unity/f1/ozavala/CODE/lce_ml_detection/eoas_pyutils")
sys.path.append("/unity/f1/ozavala/CODE/lce_ml_detection")
# sys.path.append("eoas_pyutils")
# sys.path.append("ai_common")
import time

from datetime import datetime, timedelta

import os
import xarray as xr
from os.path import join
from torch.utils.data import Dataset, DataLoader
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from proj_io.ssh_io import read_chlora_by_date_fast, read_ssh_by_date_fast, read_sst_by_date_fast
import numpy as np
import pickle

## ------- Custom dataset ------
class EddyDataset(Dataset):
    """PyTorch Dataset for loading eddy tracking data.
    
    Loads and preprocesses eddy tracking data, including optional SSH, SST
    and chlorophyll data. Can preload data to disk for faster access.
    
    Args:
        eddies_folder: Directory containing eddy data
        days_before: Number of days before eddy detection to include
        days_after: Number of days after detection
        bbox: Geographic bounding box
        output_resolution: Output image resolution
        only_ssh: Whether to load only SSH data
        start_year: Optional start year to filter data
        
    Attributes:
        data: Loaded and preprocessed data
        preload_file: Path to cached data file if used
    """

    def __init__(self, ssh_folder, sst_folder, chlora_folder, eddies_folder, bbox=[18.125, 32.125, -100.125, -74.875], output_resolution=0.125,
                 days_before = 0, days_after = 0, transform=None, start_year=None, only_ssh=False, multi_stream=False):

        print("@@@@@@@@@@@@@@@@@@@@@@@@ Initializing EddyDataset @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("SSH folder: ", ssh_folder)
        print("SST folder: ", sst_folder)
        print("Eddies folder: ", eddies_folder)
        self.ssh_folder = ssh_folder
        self.sst_folder = sst_folder
        self.chlora_folder = chlora_folder
        self.eddies_folder = eddies_folder
        self.multi_stream = multi_stream
        if start_year is None:
            self.eddies_paths = [file for file in os.listdir(eddies_folder) if file.endswith(".nc")]
        else:
            self.eddies_paths = [file for file in os.listdir(eddies_folder) if file.endswith(".nc") and int(file.split("_")[2]) >= start_year]
        self.eddies_paths.sort()
        self.total_days = int(len(self.eddies_paths))
        self.days_before = days_before
        self.days_after = days_after

        self.transform = transform
        self.bbox = bbox
        self.output_resolution = output_resolution
        self.only_ssh = only_ssh

        # Verifying if the file is already saved
        if start_year is None:
            file_name = join(self.eddies_folder, f"preloaded_db_{days_before}_da_{days_after}_submean.pkl")
        else:
            if only_ssh: # This is to PATCHED but  just to make it work fast
                file_name = join(self.eddies_folder, f"preloaded_db_{days_before}_da_{days_after}_submean_startyear_{start_year}_only_ssh.pkl")
            else:
                file_name = join(self.eddies_folder, f"preloaded_db_{days_before}_da_{days_after}_submean_startyear_{start_year}_sst_chlora.pkl")

        self.preload_file = file_name
        print(f"Searching preloaded file: {file_name}")
        if os.path.exists(file_name):
            print(f"Loading preloaded file: {file_name}")
            try: 
                with open(file_name, 'rb') as file:
                    self.data = pickle.load(file)
            except Exception as e:
                print(f"Error loading preloaded file {file_name}: {e}")
                print(f"Preloading data...")
                self.data = [{"file": file, "data": None} for file in self.eddies_paths]
                self.__preload__()
        else:
            print(f"File not found, preloading data...")
            self.data = [{"file": file, "data": None} for file in self.eddies_paths]
            self.__preload__()

        # Verify all the indexes have data and remove the Nones
        self.data = [d for d in self.data if d["data"] is not None]
        self.total_days = len(self.data)

    def __preload__(self):
        start_time = time.time()
        prev_lats = None
        prev_lons = None
        prev_ssh_month = None
        prev_ssh_file = None
        prev_sst_data = None
        prev_sst_file = None
        prev_chlora_file = None
        prev_chlora_data = None
        
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
                # ------------------------ Current day -----------------------------
                c_ssh, prev_lats, prev_lons, prev_ssh_month, prev_ssh_file = read_ssh_by_date_fast(start_date, self.ssh_folder, self.bbox, self.output_resolution,
                                                                                                   prev_ssh_month, prev_ssh_file, prev_lats, prev_lons)
                c_ssh = c_ssh - np.nanmean(c_ssh)
                ssh_all = np.nan_to_num(c_ssh.reshape(1, c_ssh.shape[0], c_ssh.shape[1]), 0)

                if not self.only_ssh:
                    # Reading SST
                    c_sst, prev_sst_data, prev_sst_file = read_sst_by_date_fast(start_date, self.sst_folder, self.bbox, prev_lats, 
                                                                                prev_lons, prev_sst_data, prev_sst_file)
                    # Reading CHLORA
                    c_chlora, prev_chlora_data, prev_chlora_file = read_chlora_by_date_fast(start_date, self.chlora_folder, self.bbox, prev_lats, 
                                                                                            prev_lons, prev_chlora_data, prev_chlora_file)

                    # Substract mean nonnan values to SSH
                    c_sst = c_sst - np.nanmean(c_sst)
                    c_chlora = c_chlora - np.nanmean(c_chlora)
                    
                    sst_all = np.nan_to_num(c_sst.reshape(1, c_sst.shape[0], c_sst.shape[1]), 0)
                    chlora_all = np.nan_to_num(c_chlora.reshape(1, c_chlora.shape[0], c_chlora.shape[1]), 0)

                # ----------------------- Reading days before -----------------------
                for i in range(1,self.days_before+1): # We need to start from 1
                    c_date = start_date - timedelta(days=i)
                    c_ssh, prev_lats, prev_lons, prev_ssh_month, prev_ssh_file = read_ssh_by_date_fast(c_date, self.ssh_folder, self.bbox, self.output_resolution,
                                                                                                   prev_ssh_month, prev_ssh_file, prev_lats, prev_lons)

                    c_ssh = c_ssh - np.nanmean(c_ssh)
                    c_ssh = np.nan_to_num(c_ssh.reshape(1, c_ssh.shape[0], c_ssh.shape[1]), 0)
                    ssh_all = np.concatenate((ssh_all, c_ssh), axis=0)

                    if not self.only_ssh:
                        # Reading SST
                        c_sst, prev_sst_data, prev_sst_file = read_sst_by_date_fast(c_date, self.sst_folder, self.bbox, prev_lats, 
                                                                                                        prev_lons, prev_sst_data, prev_sst_file)

                        # Reading CHLORA
                        c_chlora, prev_chlora_data, prev_chlora_file = read_chlora_by_date_fast(c_date, self.chlora_folder, self.bbox, prev_lats,
                                                                                                prev_lons, prev_chlora_data, prev_chlora_file)
                        
                        # Substract mean nonnan values to SSH
                        c_sst = c_sst - np.nanmean(c_sst)
                        c_chlora = c_chlora - np.nanmean(c_chlora)
                        c_sst = np.nan_to_num(c_sst.reshape(1, c_sst.shape[0], c_sst.shape[1]), 0)
                        c_chlora = np.nan_to_num(c_chlora.reshape(1, c_chlora.shape[0], c_chlora.shape[1]), 0)
                        # Concatenate previous days
                        sst_all = np.concatenate((sst_all, c_sst), axis=0)
                        chlora_all = np.concatenate((chlora_all, c_chlora), axis=0)

                # -----------------------   Reading days after -----------------------
                for i in range(1,self.days_after+1): # We need to start from 1
                    c_date = start_date + timedelta(days=i)
                    c_ssh, prev_lats, prev_lons, prev_ssh_month, prev_ssh_file = read_ssh_by_date_fast(c_date, 
                                                                                                   self.ssh_folder, self.bbox, self.output_resolution,
                                                                                                   prev_ssh_month, prev_ssh_file, prev_lats, prev_lons)

                    c_ssh = c_ssh - np.nanmean(c_ssh)
                    c_ssh = np.nan_to_num(c_ssh.reshape(1, c_ssh.shape[0], c_ssh.shape[1]), 0)
                    ssh_all = np.concatenate((ssh_all, c_ssh), axis=0)

                    if not self.only_ssh:
                        # Reading SST
                        c_sst, prev_sst_data, prev_sst_file = read_sst_by_date_fast(c_date, self.sst_folder, self.bbox, prev_lats, 
                                                                                    prev_lons, prev_sst_data, prev_sst_file)
                        # Reading CHLORA
                        c_chlora, prev_chlora_data, prev_chlora_file = read_chlora_by_date_fast(c_date, self.chlora_folder, self.bbox, prev_lats, 
                                                                                                prev_lons, prev_chlora_data, prev_chlora_file)
                        
                        # Substract mean nonnan values to SSH
                        c_sst = c_sst - np.nanmean(c_sst)
                        c_chlora = c_chlora - np.nanmean(c_chlora)
                        c_sst = np.nan_to_num(c_sst.reshape(1, c_sst.shape[0], c_sst.shape[1]), 0)
                        c_chlora = np.nan_to_num(c_chlora.reshape(1, c_chlora.shape[0], c_chlora.shape[1]), 0)
                        # Concatenate previous days
                        sst_all = np.concatenate((sst_all, c_sst), axis=0)
                        chlora_all = np.concatenate((chlora_all, c_chlora), axis=0)

                # Reading target contours
                eddies_masks = xr.open_dataset(os.path.join(self.eddies_folder,self.eddies_paths[idx]))['contours'].data
                eddies_masks = np.nan_to_num(eddies_masks.reshape(1, eddies_masks.shape[0], eddies_masks.shape[1]), 0)

                if self.only_ssh:
                    input = ssh_all
                else:
                    input = np.concatenate((ssh_all, sst_all, chlora_all), axis=0)

                self.data[idx]["data"] = input[:, :, :].astype(np.float32), eddies_masks[:, :, :].astype(np.float32)

            except Exception as e:
                print(f"Error reading file {idx}, {self.eddies_paths[idx]} date: {year}-{month}-{day_month}: {e}")
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
        # print(f"Returning data for index {idx} shape: {self.data[idx]['data'][0].shape} file {self.data[idx]['file']}")

        if self.multi_stream:
            # ---------------- multi_streams ----------------------------------
            x = self.data[idx]["data"][0]
            streams = x.shape[0]
            # Reshape input to streams, 1, w and h
            input = x.reshape(streams, 1, x.shape[1], x.shape[2])
            input = tuple( input[i, :, :, :] for i in range(streams) )
            label = self.data[idx]["data"][1]
            # For multi_stream UNet
            return self.data[idx]["file"],  (input, label)
        else:
            # ---------------- Single Stream----------------------------------
            return self.data[idx]["file"],  self.data[idx]["data"]

## ----- DataLoader --------
if __name__ == "__main__":
    # filename = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_2010_coh_14days_contour_d7_PreprocContours_2015_coh_14days_contour_d7"
    # with open(filename, 'rb') as file:
    #     x = pickle.load(file)
    # print(x)
    # ----------- Skynet ------------
    data_folder = "/unity/f1/ozavala/DATA/"
    ssh_folder = join(data_folder, "GOFFISH/AVISO/GoM")
    sst_folder = join(data_folder, "GOFFISH/SST/OSTIA")
    chlora_folder = join(data_folder, "GOFFISH/CHLORA/COPERNICUS")
    preproc_eddies_folder = "/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022_gaps_filled"
    output_folder = "/unity/f1/ozavala/CODE/lce_ml_detection/output"
    # start_year = 1998
    start_year = 2022

    # tested_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]
    tested_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5]]

    all_days_before = [1]
    all_days_after = [5]

    
    for days_before in all_days_before:
        for days_after in all_days_after:
            for c_folder in tested_folders:
                # ================= Create the dataset =====================
                print(f"Testing {c_folder} with {days_before} days before and {days_after} days after")

                eddies_folder = join(preproc_eddies_folder, c_folder)

                bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
                output_resolution = 0.1
                dataset = EddyDataset(ssh_folder, sst_folder, chlora_folder, eddies_folder, bbox, output_resolution, 
                                      days_before=days_before, days_after=days_after, start_year=start_year)

                # ================= Test data loader =====================
                # With workers I can't store all the data in memory
                # workers = 30
                # myloader = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=workers)

                # Without workers works as expected
                myloader = DataLoader(dataset, batch_size=1, shuffle=True)
                start_date = datetime.strptime(f"{start_year}-{1}-{1}", "%Y-%m-%d")
                c_date_str = f"{start_year}-{1}-{1}"

                bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
                output_resolution = 0.1
                from proj_io.ssh_io import read_ssh_by_date
                from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer
                from eoas_pyutils.viz_utils.constants import BackgroundType
                ssh, lats, lons = read_ssh_by_date(start_date, ssh_folder, bbox=bbox, output_resolution=output_resolution)

                for batch, (file, (data, target)) in enumerate(myloader): # Iterate over the batches of the test_loader
                    print(f"###################### Batch {batch} ######################")
                    # Plot the input data
                    cpu_data = data.cpu().detach().numpy()
                    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, contourf=False, disp_images=False,
                                                    background=BackgroundType.BLUE_MARBLE_HR,
                                                    max_imgs_per_row = 4, figsize = 4,
                                                    output_folder=output_folder)

                    p = 0 # Which example from the batch to plot
                    tot = days_before+days_after+1 # days_before+days_after+current
                    input_data = {
                        'adt': cpu_data[p,0,:,:],
                        'sst': cpu_data[p,tot,:,:],
                        'chlora': cpu_data[p,tot*2,:,:],
                        'diff': cpu_data[p,0,:,:] - cpu_data[p,tot-1,:,:],

                        'adt2': cpu_data[p,1,:,:],
                        'sst2': cpu_data[p,tot+1,:,:],
                        'chlora2': cpu_data[p,tot*2+1,:,:],
                        'diff2': cpu_data[p,tot,:,:] - cpu_data[p,tot+p-1,:,:],

                        'adt3': cpu_data[p,2,:,:],
                        'sst3': cpu_data[p,tot+2,:,:],
                        'chlora3': cpu_data[p,tot*2+2,:,:],
                        'diff3': cpu_data[p,tot*2,:,:] - cpu_data[p,tot*2+p-1,:,:],

                        'adt4': cpu_data[p,3,:,:],
                        'sst4': cpu_data[p,tot+3,:,:],
                        'chlora4': cpu_data[p,tot*2+3,:,:],
                        'diff4': cpu_data[p,0,:,:] - cpu_data[p,1,:,:],
                        
                        'adt5': cpu_data[p,4,:,:],
                        'sst5': cpu_data[p,tot+4,:,:],
                        'chlora5': cpu_data[p,tot*2+4,:,:],
                        'diff5': cpu_data[p,1,:,:] - cpu_data[p,2,:,:],
                        
                        'diff6': cpu_data[p,1,:,:] - cpu_data[p,2,:,:],
                        'diff7': cpu_data[p,2,:,:] - cpu_data[p,3,:,:],
                        'diff8': cpu_data[p,3,:,:] - cpu_data[p,4,:,:],
                        'diff9': cpu_data[p,4,:,:] - cpu_data[p,5,:,:],
                        } 
                    
                    viz_obj.plot_2d_data_xr(input_data, var_names=['adt','sst','chlora','diff','adt2','sst2','chlora2','diff2','adt3','sst3','chlora3','diff3',
                                                                    'adt4','sst4','chlora4','diff4','adt5','sst5','chlora5','diff5', 'diff6', 'diff7', 'diff8', 'diff9'],
                                            title=f"Input SSH, SST, CHLORA, {c_date_str}", file_name_prefix=f"B{batch}_Input_Data_{c_date_str}")
                                            # norm=[None, None, LogNorm(0.001, 10)])