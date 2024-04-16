# %% External
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os.path import join
from proj_ai.Generators import EddyDataset
from PIL import Image
from datetime import datetime
import cartopy.crs as ccrs
import cmocean
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Common
import sys
import xarray as xr
import numpy as np
import time
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
# AI common
sys.path.append("ai_common")
# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer
from eoas_pyutils.viz_utils.constants import BackgroundType
from eoas_pyutils.io_utils.io_common import create_folder
from proj_io.contours import read_contours_mask_and_polygons
from proj_io.ssh_io import read_ssh_by_date
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from models.ModelsEnum import Models
from models.ModelSelector import select_model


def getExtent(lats, lons, expand_ext=0.0):
    minLat = np.amin(lats).item() - expand_ext
    maxLat = np.amax(lats).item() + expand_ext
    minLon = np.amin(lons).item() - expand_ext
    maxLon = np.amax(lons).item() + expand_ext
    bbox = (minLon, maxLon, minLat, maxLat)
    return bbox

# Create a color dictionary for linear gradient filling
cdict = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}

# Create the colormap using the dictionary
yellow_cmap = LinearSegmentedColormap('YellowCMap', segmentdata=cdict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# From project
c_folder = 'PreprocContours_ALL_1993_2022'

all_input_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20]]

aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM"
eddies_folder = "/nexsan/people/lhiron/UGOS/Lagrangian_eddy_altimetry/eddy_contours/"
output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection_ALL_1993_2022"
summary_file = join(output_folder, "Summary", "summary.csv")

bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
output_resolution = 0.1
perc_files_to_use = 0.2
threshold = 0.5

# *********** Reads the parameters ***********
df = pd.read_csv(summary_file)

save_predictions = False
save_imgs = True

# %% Iterates over all the models in the file
for input_folder in all_input_folders:
    preproc_folder = join("/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022", input_folder)
    lcv_length_str = input_folder.split("_")[0].replace("days", "")
    # Choose the best model grouped by LCV_LEN = "LCV length"
    LCV_LEN = "LCV length"
    grouped = df.groupby(LCV_LEN)
    for group_key, group_data in grouped:
        # Get the model with the minimum loss
        group_data = group_data.sort_values(by="Loss value")
        model = group_data.iloc[0]
        print(f"Using model {model}")
        start_time = time.time()
        model_name = model["Name"]
        model_weights_file = model["Path"]
        print(F"Model model_name: {model_name}")

        days_before = int(model_name.split("_")[1])
        days_after = int(model_name.split("_")[3])

        dataset = EddyDataset(aviso_folder, preproc_folder, bbox, output_resolution, days_before=days_before, days_after=days_after)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print("Total number of test samples: ", len(test_dataset))

        c_output_folder = join(output_folder,"TestModels", model_name)
        create_folder(c_output_folder)

        # *********** Reads the weights***********
        print("Initializing model...")
        model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1+days_before+days_after,
                            output_channels=1, start_filters=32, kernel_size=3).to(device)
        print('Reading model ....')
        model.load_state_dict(torch.load(model_weights_file, map_location=device))
        model.to(device)
        
        # plot_model(model, to_file=join(c_output_folder,F'{model_name}.png'), show_shapes=True)
        # graph = torchviz.make_dot(output_tensor, params=dict(model.named_parameters()))
        # graph.render("model", format="pdf")

        out_folder_predictions = join(c_output_folder, "Predictions")
        create_folder(out_folder_predictions)
        # Working only with the test indexes
        start_time = time.time() # Just for the first case not true

        for file, (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Gets the output as a mask
            target = target[0, 0, :, :].detach().cpu().numpy()
            target = np.where(target == 0, np.nan, target)
            pred = output[0,0,:,:].detach().cpu().numpy()
            pred = np.where(pred < threshold, np.nan, pred)

            # These two lines are just to save only the predicted masks
            # filename_prediction = join(out_folder_predictions, os.path.basename(file[0].replace(".nc", ".jpg")))
            # Image.fromarray((output.cpu().detach().numpy()[0,0,:,:] * 255).astype(np.uint8)).save(filename_prediction)

            print("Time to process one sample: ", time.time() - start_time)
            start_time = time.time()
            file_name_parts = file[0].split("_")
            print(file_name_parts) # eddies_altimetry_2019_05days_078.nc
            year = int(file_name_parts[2])
            day_year = int(file_name_parts[-1].replace(".nc",""))
            month, day_month = get_month_and_day_of_month_from_day_of_year(day_year, year)
            c_date = datetime.strptime(f"{year}-{month}-{day_month}", '%Y-%m-%d')
            c_date_str = f"{year}-{month}-{day_month}"
            aviso_file = join(aviso_folder, f"{c_date.year}-{c_date.month:02d}.nc")
            contours_file = join(eddies_folder, input_folder, f"eddies_altimetry_{c_date.year}_{lcv_length_str}days_{day_year:03d}.mat")
            print(f'Reading contours from {contours_file}')
            
            # reads the corresponing ad and makes the masks and obtains the corresponding polygons contours
            ssh, lats, lons = read_ssh_by_date(c_date, aviso_folder, bbox=bbox, output_resolution=output_resolution)
            contours_mask = np.zeros_like(ssh)
            all_contours_polygons, contours_mask = read_contours_mask_and_polygons(contours_file, contours_mask, lats, lons)

            # Save the prediction as a netcdf file using xarray
            filename_prediction = join(out_folder_predictions, os.path.basename(file[0]))
            # Make a new xarray dataset considering lats, lons and the prediction as the mask
            ds = xr.Dataset(
                {
                    "prediction": (["lat", "lon"], pred),
                    "target": (["lat", "lon"], target),
                    "ssh": (["lat", "lon"], ssh)
                },
                coords={"lat": lats, "lon": lons},
            )
            # Save this new ds as a netcdf file
            ds.to_netcdf(filename_prediction)


            # Plotting some SSH data
            # Make a plot using cartopy and the data using lats and lons for bbox, ssh for the data, and plot the contours

            # =========== Manual plot =====================
            fig, ax = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent(bbox, crs=ccrs.PlateCarree())
            ax.coastlines(resolution='10m')
            gl = ax.gridlines(draw_labels=True, dms=True, linestyle='--', alpha=0.5, color='black')

            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

            # Reading the polygon data 
            pol_lats = []
            pol_lons = []
            as_polygons = [Polygon(x) for x in all_contours_polygons]
            for c_polygon in as_polygons:
                x, y = c_polygon.exterior.xy
                pol_lats += y
                pol_lons += x
                ax.plot(x,y, transform=ccrs.PlateCarree(), c='blue', linewidth=1, linestyle='--')

            # extent = getExtent(lats, lons, 0.5)
            extent = getExtent(lats, lons)
            # ax.set_extent(getExtent(list(lats) + pol_lats, list(lons) + pol_lons, 0.5))
            ax.set_extent(extent)

            # ax.contourf(lons, lats, ssh, transform=ccrs.PlateCarree(), cmap=cmocean.cm.curl)
            ax.imshow(ssh, extent=extent, cmap=cmocean.cm.curl, transform=ccrs.PlateCarree(), origin='lower')
            # ax.imshow(pred, extent=extent, cmap=yellow_cmap, transform=ccrs.PlateCarree(), origin='lower')
            ax.imshow(pred, extent=extent, cmap=cmocean.cm.thermal, transform=ccrs.PlateCarree(), origin='lower')
            # Set x and y axis labels
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Predicted LCV for {c_date_str}')
            plt.savefig(join(c_output_folder, f"Manual_Plot_{c_date_str}.jpg"))
            plt.close()

            # =========== Using our library =====================
            # viz_obj = EOAImageVisualizer(lats=lats, lons=lons, contourf=False, disp_images=True,
            #                                 background=BackgroundType.BLUE_MARBLE_HR,
            #                                 output_folder=c_output_folder)
            # viz_obj.__setattr__('additional_polygons', [Polygon(x) for x in all_contours_polygons])
            # viz_obj.plot_2d_data_np(contours_mask,  var_names=['mask'],  title='Eddies mask')
            # viz_obj.plot_2d_data_xr({'adt': ssh, 'target': target,
            #                          'prediction':pred}, var_names=['adt', 'target', 'prediction'],
            #                         title=c_date, file_name_prefix=f"adt_target_prediction_{c_date}")
            # viz_obj.plot_2d_data_xr({'LAVD': ssh, 'Prediction': pred}, var_names=['LAVD', 'Prediction'],
            #                 title=[f'{c_date_str} Target', f'{c_date_str} Prediction'], file_name_prefix=f"prediction_adt{c_date}")
# %%
