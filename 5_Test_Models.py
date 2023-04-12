# External
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os.path import join
from proj_ai.Generators import EddyDataset
from datetime import datetime
# Common
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
# AI common
sys.path.append("ai_common")
# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer
from eoas_pyutils.io_utils.io_common import create_folder
from proj_io.contours import read_contours_mask_and_polygons
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from models.ModelsEnum import Models
from models.ModelSelector import select_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# From project
data_folder = "/unity/f1/ozavala/DATA/GOFFISH"
folders = ['2010_coh_14days_contour_d7', '2020_coh_14days_contour_d7', '2020_coh_14days_contour_d0']
c_folder = folders[1]

ssh_folder = join(data_folder, "AVISO")
preproc_folder = join(data_folder,f"EddyDetection/PreprocContours_{c_folder}")
contours_folder = f"/nexsan/people/lhiron/UGOS/Lag_coh_eddies/eddy_contours/altimetry_{c_folder}/"
output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection/"
summary_file = join(output_folder, "Summary", "summary.csv")

bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
output_resolution = 0.1

test_dataset = EddyDataset(ssh_folder, preproc_folder, bbox, output_resolution)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Total number of test samples: ", len(test_dataset))

# *********** Reads the parameters ***********
df = pd.read_csv(summary_file)

save_predictions = False
save_imgs = True

# Iterates over all the models in the file
for model_id in range(len(df)):
    model = df.iloc[model_id]
    model_name = model["Name"]
    model_weights_file = model["Path"]
    print(F"Model model_name: {model_name}")

    c_output_folder =join(output_folder,model_name)
    create_folder(c_output_folder)

    # *********** Reads the weights***********
    print("Initializing model...")
    model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1,
                         output_channels=1, start_filters=32, kernel_size=3).to(device)
    print('Reading model ....')
    model.load_state_dict(torch.load(model_weights_file, map_location=device))
    model.to(device)
    #
    # plot_model(model, to_file=join(c_output_folder,F'{model_name}.png'), show_shapes=True)
    # graph = torchviz.make_dot(output_tensor, params=dict(model.named_parameters()))
    # graph.render("model", format="pdf")

    # Working only with the test indexes
    for file, (data, target) in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        file_name_parts = file[0].split("_")
        print(file_name_parts)
        year = int(file_name_parts[2])
        day_year = int(file_name_parts[4].replace(".nc",""))
        month, day_month = get_month_and_day_of_month_from_day_of_year(day_year, year)
        c_date = datetime.strptime(f"{year}-{month}-{day_month}", '%Y-%m-%d')
        ssh_file = join(ssh_folder, f"{c_date.year}-{c_date.month:02d}.nc")
        contours_file = join(contours_folder, f"eddies_altimetry_{c_date.year}_14days_{day_year:03d}.mat")
        print(contours_file)

        ds = xr.open_dataset(ssh_file)
        ds = ds.sel(latitude=slice(bbox[0], bbox[1]), longitude=slice(bbox[2], bbox[3]))
        lats = ds.latitude.values
        lons = ds.longitude.values

        print(f"min lat: {np.amin(lats)}, max lat: {np.amax(lats)}")
        print(f"min lon: {np.amin(lons)}, max lon: {np.amax(lons)}")
        lons = lons - 360 if np.amax(lons) > 180 else lons
        print("Done!")

        # %%
        # Plotting some SSH data
        print(ds.head())
        viz_obj = EOAImageVisualizer(lats=lats, lons=lons)
        contours_mask = np.zeros_like(ds['adt'][day_month, :, :])
        all_contours_polygons, contours_mask = read_contours_mask_and_polygons(contours_file, contours_mask, lats, lons)
        viz_obj.__setattr__('additional_polygons', [Polygon(x) for x in all_contours_polygons])
        # viz_obj.plot_3d_data_npdict(ds,  var_names=['adt'], z_levels=[day_month], title=f'SSH and contours for {c_date}')
        # viz_obj.plot_2d_data_np(contours_mask,  var_names=['mask'],  title='Eddies mask')
        target_flip = np.flip(target[0, 0, :, :].detach().cpu().numpy(), axis=0)
        target_flip = np.where(target_flip == 0, np.nan, target_flip)
        pred_flip = np.flip(output[0,0,:,:].detach().cpu().numpy(), axis=0)
        pred_flip = np.where(pred_flip < 0.1, np.nan, pred_flip)
        viz_obj.plot_2d_data_xr({'adt': ds['adt'][day_month], 'target': target_flip,
                                 'prediction':pred_flip}, var_names=['adt', 'target', 'prediction'],
                                title=c_date)