from os import path, listdir, remove
from dask.distributed import Client
from goes2go import GOES

import xarray as xr
import numpy as np
from dask import array as da
from dask.distributed import print

from datetime import datetime as dt, timedelta
from scipy.interpolate import griddata

from pyxlma import coords

base_path = path.dirname(__file__)
USE_DASK = True
FIELD_I_WANT = 'L2-MCMIPC'
# L2-ACHTF - derived cloud top temperature
# L2-MCMIPC - channel 13 brightness temperature
# L2-ACHAC - derived cloud top height



def read_tobac_ds(tobac_path):
    from glmtools.io.lightning_ellipse import lightning_ellipse_rev
    tfm = xr.open_dataset(tobac_path).load()
    radar_lat, radar_lon = 29.47, -95.08
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=radar_lat, ctrLon=radar_lon, ctrAlt=0)
    geosys = coords.GeographicSystem()
    ltg_ell = lightning_ellipse_rev[1]
    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=-75.19999694824219, sweep_axis='x', ellipse=ltg_ell)
    grid_ecef_coords = tpcs.toECEF(tfm.x.data, tfm.y.data, np.zeros_like(tfm.x.data))
    grid_lon, grid_lat, _ = geosys.fromECEF(*grid_ecef_coords)
    grix_g16_scan_x, grix_g16_scan_y, _ = satsys.fromECEF(*grid_ecef_coords)

    tfm = tfm.assign({'lat' : (('y'), grid_lat), 'lon' : (('x'), grid_lon)})
    tfm = tfm.assign({'g16_scan_x' : (('x'), grix_g16_scan_x), 'g16_scan_y' : (('y'), grix_g16_scan_y)})
    tfm.attrs['center_lat'] = radar_lat
    tfm.attrs['center_lon'] = radar_lon

    feature_ecef_coords = tpcs.toECEF(tfm.feature_projection_x_coordinate.data, tfm.feature_projection_y_coordinate.data, np.zeros_like(tfm.feature_projection_x_coordinate.data))
    feature_lon, feature_lat, _ = geosys.fromECEF(*feature_ecef_coords)
    tfm = tfm.assign({'feature_lat' : (('feature'), feature_lat), 'feature_lon' : (('feature'), feature_lon)})
    return tfm

def fixup_transformed_coords(transformed_coords, input_shape):
    for transformed in transformed_coords:
        if not isinstance(transformed, np.ndarray):
            transformed = da.atleast_1d(transformed)
        transformed = transformed.reshape(input_shape)
    return transformed_coords


def find_satellite_temp_for_feature(tfm_time, feature_i_want, area_i_want):
    # Find the index boundaries of the feature
    x_indices_valid, y_indices_valid = np.asarray(tfm_time.segmentation_mask == feature_i_want).nonzero()
    first_x_idx = np.min(x_indices_valid)
    first_y_idx = np.min(y_indices_valid)
    last_x_idx = np.max(x_indices_valid)
    last_y_idx = np.max(y_indices_valid)

    # Trim the grid to a rectangle surrounding the feature
    grid_x2d, grid_y2d = np.meshgrid(tfm_time.x, tfm_time.y)
    this_feature_x2d = grid_x2d[first_x_idx:last_x_idx+1, first_y_idx:last_y_idx+1]
    this_feature_y2d = grid_y2d[first_x_idx:last_x_idx+1, first_y_idx:last_y_idx+1]
    this_seg_mask = tfm_time.segmentation_mask.data[first_x_idx:last_x_idx+1, first_y_idx:last_y_idx+1]

    # Create coordinate systems
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=tfm_time.center_lat,
                                              ctrLon=tfm_time.center_lon, ctrAlt=0)
    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=area_i_want.nominal_satellite_subpoint_lon.data.item(),
                                                 sweep_axis='x')

    # Get the ECEF coordinates of the grid centers of the rectangle containing the feature
    this_feature_ECEF = tpcs.toECEF(this_feature_x2d.flatten(), this_feature_y2d.flatten(), np.zeros_like(this_feature_y2d).flatten())
    this_feature_ECEF_X, this_feature_ECEF_Y, this_feature_ECEF_Z  = fixup_transformed_coords(this_feature_ECEF, this_feature_x2d.flatten().shape)

    # Convert the ECEF coordinates of the rectangle to GOES-16 scan angles to trim the satellite data even further--performance enhancement
    this_feature_g16 = satsys.fromECEF(this_feature_ECEF_X.flatten(), this_feature_ECEF_Y.flatten(), this_feature_ECEF_Z.flatten())
    this_feature_g16_x, this_feature_g16_y, _ = fixup_transformed_coords(this_feature_g16, this_feature_ECEF_X.flatten().shape)
    this_feature_g16_xmax = np.max(this_feature_g16_x)
    this_feature_g16_ymax = np.max(this_feature_g16_y)
    this_feature_g16_xmin = np.min(this_feature_g16_x)
    this_feature_g16_ymin = np.min(this_feature_g16_y)
    feature_padding = .0002 # A little bit of padding to be safe
    feature_area_i_want = area_i_want.sel(y=slice(this_feature_g16_ymax+feature_padding, this_feature_g16_ymin-feature_padding),
                                  x=slice(this_feature_g16_xmin-feature_padding, this_feature_g16_xmax+feature_padding))
    
    this_satellite_scan_x, this_satellite_scan_y = da.meshgrid(feature_area_i_want.x, feature_area_i_want.y)
    sat_ecef = satsys.toECEF(this_satellite_scan_x.flatten(), this_satellite_scan_y.flatten(), np.zeros_like(this_satellite_scan_x.flatten()))
    sat_ecef_X, sat_ecef_Y, sat_ecef_Z = fixup_transformed_coords(sat_ecef, this_satellite_scan_x.flatten().shape)
    sat_tpcs = tpcs.fromECEF(sat_ecef_X.flatten(), sat_ecef_Y.flatten(), sat_ecef_Z.flatten())
    sat_tpcs_X, sat_tpcs_Y, _ = fixup_transformed_coords(sat_tpcs, sat_ecef_X.flatten().shape)
    # if not isinstance(sat_tpcs_X, np.ndarray):
    #     return np.nan, np.nan, np.nan
    sat_tpcs_X.shape = this_satellite_scan_x.shape
    sat_tpcs_Y.shape = this_satellite_scan_y.shape
    if 'TEMP' in feature_area_i_want.data_vars:
        vals_to_interp = feature_area_i_want.TEMP.data.flatten()
    elif 'CMI_C13' in feature_area_i_want.data_vars:
        vals_to_interp = feature_area_i_want.CMI_C13.data.flatten()
    elif 'HT' in feature_area_i_want.data_vars:
        vals_to_interp = feature_area_i_want.HT.data.flatten()
        if len(vals_to_interp) < 4:
            print(feature_area_i_want)
            raise ValueError('No temperature data found in feature_area_i_want')
            return np.nan, np.nan, np.nan
    else:
        raise ValueError('No temperature data found in feature_area_i_want')
    vals = griddata(
        np.array([sat_tpcs_X.flatten(), sat_tpcs_Y.flatten()]).T,
        vals_to_interp,
        np.array([this_feature_x2d.flatten(), this_feature_y2d.flatten()]).T,
        method='linear'
    )
    vals.shape = this_feature_x2d.shape
    vals_i_want = vals[this_seg_mask == feature_i_want]
    if np.all(np.isnan(vals_i_want)):
        min_sat_temp = np.nan
        mean_sat_temp = np.nan
        std_sat_temp = np.nan
    else:
        min_sat_temp = da.nanmin(vals_i_want)
        mean_sat_temp = da.nanmean(vals_i_want)
        std_sat_temp = da.nanstd(vals_i_want)
    return min_sat_temp, mean_sat_temp, std_sat_temp


def find_satellite_temp_for_timestep(tfm_time, goes_file_path, num_features):
    satellite_data = xr.open_dataset(goes_file_path, chunks='auto').load()
    max_x = tfm_time.g16_scan_x.max().data.item()
    min_x = tfm_time.g16_scan_x.min().data.item()
    max_y = tfm_time.g16_scan_y.max().data.item()
    min_y = tfm_time.g16_scan_y.min().data.item()
    padding = .001
    area_i_want = satellite_data.sel(y=slice(max_y+padding, min_y-padding), x=slice(min_x-padding, max_x+padding))
    features_in_time = np.setdiff1d(np.unique(tfm_time.segmentation_mask.data), [0])
    mins = np.full(num_features, np.nan)
    means = np.full(num_features, np.nan)
    stds = np.full(num_features, np.nan)
    for feature_index, feature_i_want in enumerate(features_in_time):
        min_sat_temp, mean_sat_temp, std_sat_temp = find_satellite_temp_for_feature(tfm_time, feature_i_want, area_i_want)
        mins[feature_index] = min_sat_temp
        means[feature_index] = mean_sat_temp
        stds[feature_index] = std_sat_temp
    satellite_data.close()
    return da.array(mins), da.array(means), da.array(stds)


def add_goes_data_to_tobac_path(tobac_path, thing_to_plot):
    tfm = read_tobac_ds(tobac_path)
    min_sat_temp = da.full((tfm.feature.shape[0], tfm.time.shape[0]), np.nan)
    mean_sat_temp = da.full((tfm.feature.shape[0], tfm.time.shape[0]), np.nan)
    std_satellite_temp = da.full((tfm.feature.shape[0], tfm.time.shape[0]), np.nan)

    goes_time_range_start = tfm.time.data.astype('datetime64[s]').astype(dt)[0]
    goes_time_range_end = tfm.time.data.astype('datetime64[s]').astype(dt)[-1]
    goes_ctt = GOES(satellite=16, product=f'ABI-{thing_to_plot}')
    print('Start download')
    download_results = goes_ctt.timerange(goes_time_range_start-timedelta(minutes=15), goes_time_range_end+timedelta(minutes=15))
    print('End download')
    
    download_results['valid'] = download_results[['start', 'end']].mean(axis=1)
    valid_times = download_results['valid'].values.astype('datetime64[s]')
    tobac_times = tfm.time.data.astype('datetime64[s]')
    time_diffs = np.abs(tobac_times[:, np.newaxis] - valid_times)
    goes_time_matching_tobac_indices = np.argmin(time_diffs, axis=0)
    download_results['tobac_idx'] = goes_time_matching_tobac_indices
    for idx, num_occurrences in zip(*np.unique(goes_time_matching_tobac_indices, return_counts=True)):
        if num_occurrences > 1:
            dup_df = download_results[download_results['tobac_idx'] == idx]
            time_to_match = tobac_times[idx]
            diff = np.abs(dup_df['valid'].values - time_to_match)
            download_results.drop(dup_df[diff > diff.min()].index, inplace=True)
    download_results.reset_index(drop=True, inplace=True)
    long_time_gaps = np.diff(download_results['valid'].values).astype('timedelta64[s]') >= 601
    gap_time_start = download_results['valid'].values[:-1][long_time_gaps]
    gap_time_end = download_results['valid'].values[1:][long_time_gaps]
    for gap in zip(gap_time_start, gap_time_end):
        gap = np.array(gap).astype('datetime64[s]')
        print(f'Warning, long gap between {gap[0]} and {gap[1]}')
    rets = []
    num_rows = len(download_results)
    for a, row in download_results.iterrows():
        goes_file_path = path.join('/Volumes/LtgSSD/', row['file'])
        tobac_idx_with_goes_time = row['tobac_idx']
        tfm_time = tfm.isel(time=tobac_idx_with_goes_time)
        print(f'({(100*a/num_rows):.1f}%) {tfm_time.time.data.astype("datetime64[s]").astype(dt)}')
        rets.append(find_satellite_temp_for_timestep(tfm_time, goes_file_path, tfm.feature.shape[0]))
    res = da.array(rets)
    for res_idx, full_idx in enumerate(download_results['tobac_idx'].values):
        min_sat_temp[:, full_idx] = res[res_idx, 0, :]
        mean_sat_temp[:, full_idx] = res[res_idx, 1, :]
        std_satellite_temp[:, full_idx] = res[res_idx, 2, :]
    tfm[f'min_{thing_to_plot}'] = xr.DataArray(min_sat_temp, dims=('feature', 'time'))
    tfm[f'avg_{thing_to_plot}'] = xr.DataArray(mean_sat_temp, dims=('feature', 'time'))
    tfm[f'std_{thing_to_plot}'] = xr.DataArray(std_satellite_temp, dims=('feature', 'time'))
    path_to_save = tobac_path.replace('merges.nc', 'merges_augmented.nc')
    if path.exists(path_to_save):
        remove(path_to_save)
    comp = dict(zlib=True, complevel=5)
    enc = {var: comp for var in tfm.data_vars if not np.issubdtype(tfm[var].dtype, str)}
    tfm.to_netcdf(path_to_save, encoding=enc)

if __name__ == '__main__':
    if USE_DASK:
        client = Client('tcp://127.0.0.1:8786')
    tobac_main_path = path.join(path.sep, 'Volumes', 'LtgSSD', 'tobac_saves')
    tobac_dayfiles = []
    for d in sorted(listdir(tobac_main_path)):
        if path.isdir(path.join(tobac_main_path, d)):
            if path.exists(path.join(tobac_main_path, d, 'Track_features_merges_augmented.nc')):
                tobac_dayfiles.append(path.join(tobac_main_path, d, 'Track_features_merges_augmented.nc'))
            else:
                if path.exists(path.join(tobac_main_path, d, 'Track_features_merges.nc')):
                    tobac_dayfiles.append(path.join(tobac_main_path, d, 'Track_features_merges.nc'))
                else:
                    print(f'Need to tobac track for day: {d}')

    # TESTING: Only do the first dayfile
    tobac_dayfiles = [tobac_dayfiles[4]]
    if USE_DASK:
        print('started mapping')
        res = client.map(add_goes_data_to_tobac_path, tobac_dayfiles, [FIELD_I_WANT]*len(tobac_dayfiles))
        print('done mapping')
        client.gather(res)
    else:
        for tobac_file in tobac_dayfiles:
            add_goes_data_to_tobac_path(tobac_file, FIELD_I_WANT)
    
