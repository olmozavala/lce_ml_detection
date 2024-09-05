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
    # u = ssh_month['ugos'][c_date.day - 1, :, :].data
    # v = ssh_month['vgos'][c_date.day - 1, :, :].data

    # ------------------ Just for visualization -------------------
    # viz_obj = EOAImageVisualizer(lats=lats, lons=lons, eoas_pyutils_path="../eoas_pyutils", )
    # import h5py
    # viz_obj.plot_2d_data_xr({'ssh': ssh}, var_names=['ssh'],
    #                         title=c_date.strftime("%Y-%m-%d"))
    # return ssh, u, v, lats, lons
    return ssh, lats, lons

def read_ssh_by_date_fast(c_date, ssh_folder, bbox, output_resolution, 
                          prev_ssh_month=None, prev_ssh_file=None, prev_lats=None, prev_lons=None):
    '''
    Reads the SSH data for the given date and interpolates to the desired resolution
    Args:
        c_date:

    Returns:
    '''
    ssh_file = join(ssh_folder, f"{c_date.year}-{c_date.month:02d}.nc")
    use_prev = False

    if prev_ssh_file is not None:
        if prev_ssh_file == ssh_file:
            use_prev = True

    if use_prev:
        ssh_month = prev_ssh_month
        lats = prev_lats
        lons = prev_lons
    else:
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


    ssh = ssh_month['adt'][c_date.day - 1, :, :].data # -1 because it goes from 0 to 30
    return ssh, lats, lons, ssh_month, ssh_file

def read_sst_by_date_fast(c_date, sst_folder, bbox, lats, lons,
                          prev_sst_data=None, prev_sst_file=None):
    '''
    Reads the sst data for the given date and interpolates to the desired resolution
    Args:
        c_date:

    Returns:
    '''
    sst_file = join(sst_folder, f"OSTIA_SST_{c_date.year}.nc")
    use_prev = False

    if prev_sst_file is not None:
        if prev_sst_file == sst_file:
            use_prev = True

    if use_prev:
        sst_year = prev_sst_data
    else:
        sst_year = xr.open_dataset(sst_file)
        sst_year = sst_year.sel(latitude=slice(bbox[0], bbox[1]),
                                longitude=slice(bbox[2], bbox[3]))

        sst_year = sst_year.interp(latitude=lats, longitude=lons)  # Increase output_resolution

    assert lats.shape == sst_year.latitude.shape
    assert lons.shape == sst_year.longitude.shape

    day_of_year = c_date.timetuple().tm_yday
    sst = sst_year['analysed_sst'][day_of_year - 1, :, :].data
    return sst, sst_year, sst_file

def read_chlora_by_date_fast(c_date, chlora_folder, bbox, lats=None, lons=None,
                          prev_chlora_year=None, prev_chlora_file=None):

    chlora_file = join(chlora_folder, f"Ocean_Color_{c_date.year}.nc")
    use_prev = False

    if prev_chlora_file is not None:
        if prev_chlora_file == chlora_file:
            use_prev = True

    if use_prev:
        chlora_data = prev_chlora_year
    else:
        chlora_data = xr.open_dataset(chlora_file, decode_times=False)
        chlora_data = chlora_data.sel(latitude=slice(bbox[0], bbox[1]),
                                longitude=slice(bbox[2], bbox[3]))

        if (lats is not None) and (lons is not None):
            chlora_data = chlora_data.interp(latitude=lats, longitude=lons) 

    if (lats is not None) and (lons is not None):
        assert lats.shape == chlora_data.latitude.shape
        assert lons.shape == chlora_data.longitude.shape

    day_of_year = c_date.timetuple().tm_yday
    chlora = chlora_data['CHL'][day_of_year - 1, :, :].data

    return chlora, chlora_data, chlora_file