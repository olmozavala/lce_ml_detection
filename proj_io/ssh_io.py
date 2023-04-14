from os.path import join
import xarray as xr
import numpy as np

def largest_number_divisible_by_8(N):
    quotient, remainder = divmod(N, 8)
    if remainder == 0:
        return N
    else:
        return N - remainder

def read_ssh_by_date(c_date, ssh_folder, bbox, output_resolution):
    '''
    Reads the SSH data for the given date and interpolates to the desired resolution
    Args:
        c_date:

    Returns:
    '''
    ssh_file = join(ssh_folder, f"{c_date.year}-{c_date.month:02d}.nc")
    ssh_month = xr.open_dataset(ssh_file)
    ssh_month = ssh_month.sel(latitude=slice(bbox[0], bbox[1]),
                              longitude=slice(bbox[2], bbox[3]))

    # Interpolate to output resolution
    lats_orig = ssh_month.latitude.values
    lons_orig = ssh_month.longitude.values
    lats = np.linspace(np.amin(lats_orig), np.amax(lats_orig),
                       int((np.amax(lats_orig) - np.amin(lats_orig)) / output_resolution))
    lons = np.linspace(np.amin(lons_orig), np.amax(lons_orig),
                       int((np.amax(lons_orig) - np.amin(lons_orig)) / output_resolution))

    # Fixing to the largest size that is divisible by 8 in both coordinates
    lats = lats[:largest_number_divisible_by_8(lats.shape[0])]
    lons = lons[:largest_number_divisible_by_8(lons.shape[0])]

    ssh_month = ssh_month.interp(latitude=lats, longitude=lons)  # Increase output_resolution
    ssh = ssh_month['adt'][c_date.day - 1, :, :].data

    # ------------------ Just for visualization -------------------
    # viz_obj = EOAImageVisualizer(lats=lats, lons=lons, eoas_pyutils_path="../eoas_pyutils", )
    # import h5py
    # viz_obj.plot_2d_data_xr({'ssh': ssh}, var_names=['ssh'],
    #                         title=c_date.strftime("%Y-%m-%d"))
    return ssh, lats, lons