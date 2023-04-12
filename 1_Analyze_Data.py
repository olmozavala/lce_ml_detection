# External
import sys
sys.path.append("eoas_pyutils")
sys.path.append("ai_common")
import xarray as xr
from os.path import join
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime
# Common
from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.dates_utils import get_day_of_year_from_month_and_day
# Local
from proj_io.contours import read_contours_mask_and_polygons

#%%  Reading a single example (contours and ssh)
print("Reading data...")
# Define which day are we going to use

ssh_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO"
contours_folder = "/nexsan/people/lhiron/UGOS/Lag_coh_eddies/eddy_contours/altimetry_2010_coh_14days_contour_d7/"

bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used by DataReader

dates_to_plot = ('2010-01-14', '2010-03-14', '2010-05-14','2010-08-14','2010-10-14','2010-12-14')
for c_date_str in dates_to_plot:
    c_date = datetime.strptime(c_date_str,'%Y-%m-%d')
    day_year = get_day_of_year_from_month_and_day(c_date.month, c_date.day, c_date.year)
    day_month = c_date.day
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

    #%%
    # Plotting some SSH data
    print(ds.head())
    viz_obj = EOAImageVisualizer(lats=lats, lons=lons)
    contours_mask = np.zeros_like(ds['adt'][day_month, :, :])
    all_contours_polygons, contours_mask = read_contours_mask_and_polygons(contours_file, contours_mask, lats, lons)
    viz_obj.__setattr__('additional_polygons', [Polygon(x) for x  in all_contours_polygons])
    # viz_obj.plot_3d_data_npdict(ds,  var_names=['adt'], z_levels=[day_month], title=f'SSH and contours for {c_date}')
    # viz_obj.plot_2d_data_np(contours_mask,  var_names=['mask'],  title='Eddies mask')
    viz_obj.plot_2d_data_xr({'adt':ds['adt'][day_month], 'mask':contours_mask},  var_names=['adt', 'mask'],  title=c_date)