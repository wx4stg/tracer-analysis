#!/usr/bin/python3

from os import listdir, path
from pathlib import Path
from shutil import rmtree
from datetime import datetime as dt, timedelta
import sys

from dask.distributed import Client

import numpy as np
import xarray as xr
from goes2go import GOES

from glmtools.io.lightning_ellipse import lightning_ellipse_rev
from pyxlma import coords
from scipy.interpolate import griddata

def fixup_transformed_coords(transformed_coords, input_shape):
    for transformed in transformed_coords:
        if not isinstance(transformed, np.ndarray):
            transformed = np.atleast_1d(transformed)
        transformed = transformed.reshape(input_shape)
    return transformed_coords


def apply_coord_transforms(tfm):
    tfm = tfm.copy()
    radar_lat, radar_lon = 29.47, -95.08
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=radar_lat, ctrLon=radar_lon, ctrAlt=0)
    ltg_ell = lightning_ellipse_rev[1]
    
    x2d, y2d = np.meshgrid(tfm.x.data, tfm.y.data)
    grid_ecef_coords = tpcs.toECEF(x2d.flatten(), y2d.flatten(), np.zeros_like(x2d).flatten())
    
    geosys = coords.GeographicSystem()
    grid_lon, grid_lat, _ = geosys.fromECEF(*grid_ecef_coords)
    grid_lon = grid_lon.reshape(x2d.shape)
    grid_lat = grid_lat.reshape(x2d.shape)
    tfm = tfm.assign({'lat' : (('x', 'y'), grid_lat), 'lon' : (('x', 'y'), grid_lon)})


    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=-75.19999694824219, sweep_axis='x', ellipse=ltg_ell)
    grid_g16_scan_x, grid_g16_scan_y, _ = satsys.fromECEF(*grid_ecef_coords)
    grid_g16_scan_x = grid_g16_scan_x.reshape(x2d.shape)
    grid_g16_scan_y = grid_g16_scan_y.reshape(x2d.shape)
    tfm = tfm.assign({'g16_scan_x' : (('x', 'y'), grid_g16_scan_x), 'g16_scan_y' : (('x', 'y'), grid_g16_scan_y)})

    tfm.attrs['center_lat'] = radar_lat
    tfm.attrs['center_lon'] = radar_lon

    feature_ecef_coords = tpcs.toECEF(tfm.feature_projection_x_coordinate.data, tfm.feature_projection_y_coordinate.data, np.zeros_like(tfm.feature_projection_x_coordinate.data))
    feature_lon, feature_lat, _ = geosys.fromECEF(*feature_ecef_coords)
    tfm = tfm.assign({'feature_lat' : (('feature'), feature_lat), 'feature_lon' : (('feature'), feature_lon)})
    return tfm


def add_eet_to_radar_data(tobac_day, client=None):
    def compute_eet_radar(this_radar_path):
        if not this_radar_path.endswith('.nc'):
            return
        this_radar_path = path.join(radar_path, this_radar_path)
        radar_data = xr.open_dataset(this_radar_path)

        where18 = (radar_data.reflectivity >= 18)
        et_diff_from_top = 500*(where18.isel(z=slice(None, None, -1)).argmax(dim='z'))
        et_diff_from_top = xr.where(where18.any(dim='z'), et_diff_from_top, np.nan)
        echo_top = np.max(radar_data.z) - et_diff_from_top
        echo_top = xr.where(echo_top == np.max(radar_data.z), -1, echo_top)
        radar_data['eet_sam'] = echo_top
        save_path = this_radar_path.replace('nexrad_gridded', 'nexrad_zarr').replace('.nc', '.zarr')
        Path(path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        radar_data.to_zarr(save_path)
    radar_path = path.join(path.sep, 'Volumes', 'LtgSSD', 'nexrad_gridded', tobac_day.strftime('%B').upper(), tobac_day.strftime('%Y%m%d'))
    print('Finding elevations for all tobac grid points')
    radar_path_contents = sorted(listdir(radar_path))
    if len(radar_path_contents) == 0:
        raise ValueError('No gridded radar data found!')
    print('Starting processing...')
    all_res = []
    for this_radar_path in radar_path_contents:
        if client is None:
            compute_eet_radar(this_radar_path)
        else:
            all_res.append(client.submit(compute_eet_radar, this_radar_path))

    if client is not None:
        client.gather(all_res)


def add_eet_to_tobac_data(tfm, date_i_want, client=None):
    def find_eet_feature(tfm, radar_path, time_idx):
        radar = xr.open_dataset(radar_path, engine='zarr')
        feature_indicies_at_time = np.nonzero(tfm.feature_time_index.compute().data == time_idx)[0]
        features_at_time = tfm.isel(feature=feature_indicies_at_time, time=time_idx)
        feature_eet = np.full(features_at_time.feature.data.shape, np.nan)
        for j, feat_to_find in enumerate(features_at_time.feature.data):
            this_seg_mask = features_at_time.segmentation_mask.data
            this_eet = radar.eet_sam.isel(time=0).data
            if not np.any(this_seg_mask == feat_to_find):
                continue
            feature_eet[j] = np.nanmax(this_eet[this_seg_mask == feat_to_find])
        start_idx = feature_indicies_at_time[0]
        end_idx = feature_indicies_at_time[-1] + 1
        radar.close()
        return feature_eet, start_idx, end_idx
    
    def prepare_eet_features(i):
        rf = radar_files[i]
        if not rf.endswith('.zarr'):
            return
        radar_dt = dt.strptime(rf, 'KHGX%Y%m%d_%H%M%S_V06_grid.zarr')
        expected_dt = tfm_dts[i]
        if radar_dt != expected_dt:
            raise ValueError(f'Error at index {i}! Expected {expected_dt}, got {radar_dt}')
        rp = radar_top_path + rf
        return find_eet_feature(tfm, rp, i)


    radar_top_path = f'/Volumes/LtgSSD/nexrad_zarr/{date_i_want.strftime('%B').upper()}/{date_i_want.strftime('%Y%m%d')}/'
    radar_files = sorted(listdir(radar_top_path))
    tfm_dts = tfm.time.data.astype('datetime64[s]').astype(dt)
    feature_eet = np.full(tfm.feature.data.shape, np.nan)

    
    if client is not None:
        res = client.map(prepare_eet_features, range(len(radar_files)))
        client.gather(res)
        for promised_res in res:
            d, s, e = promised_res.result()
            feature_eet[s:e] = d
    else:
        for i in range(len(radar_files)):
            d, s, e = prepare_eet_features(i)
            feature_eet[s:e] = d

    tfm = tfm.assign(
        feature_echotop = (('feature'), feature_eet)
    )
    return tfm


def find_satellite_temp_for_feature(tfm_time, feature_i_want, area_i_want, feat_echotop=7e3):
    if np.isnan(feat_echotop):
        feat_echotop = 7e3
    ltg_ell = lightning_ellipse_rev[1]
    # Find the index boundaries of the feature
    x_indices_valid, y_indices_valid = np.asarray(tfm_time.segmentation_mask == feature_i_want).nonzero()
    if len(x_indices_valid) == 0:
        return np.nan
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
                                                 sweep_axis='x', ellipse=(ltg_ell[0] - 14e3 + feat_echotop, ltg_ell[1] - 6e3 + feat_echotop))

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
    
    this_satellite_scan_x, this_satellite_scan_y = np.meshgrid(feature_area_i_want.x, feature_area_i_want.y)
    sat_ecef = satsys.toECEF(this_satellite_scan_x.flatten(), this_satellite_scan_y.flatten(), np.zeros_like(this_satellite_scan_x.flatten()))
    sat_ecef_X, sat_ecef_Y, sat_ecef_Z = fixup_transformed_coords(sat_ecef, this_satellite_scan_x.flatten().shape)
    sat_tpcs = tpcs.fromECEF(sat_ecef_X.flatten(), sat_ecef_Y.flatten(), sat_ecef_Z.flatten())
    sat_tpcs_X, sat_tpcs_Y, _ = fixup_transformed_coords(sat_tpcs, sat_ecef_X.flatten().shape)
    sat_tpcs_X.shape = this_satellite_scan_x.shape
    sat_tpcs_Y.shape = this_satellite_scan_y.shape
    vals_to_interp = feature_area_i_want.CMI_C13.data.flatten()
    vals = griddata(
        np.array([sat_tpcs_X.flatten(), sat_tpcs_Y.flatten()]).T,
        vals_to_interp,
        np.array([this_feature_x2d.flatten(), this_feature_y2d.flatten()]).T,
        method='linear'
    )
    vals.shape = this_feature_x2d.shape
    try:
        vals_i_want = vals[this_seg_mask == feature_i_want]
    except IndexError as e:
        vals_i_want = np.array([np.nan])
    if np.all(np.isnan(vals_i_want)):
        min_sat_temp = np.nan
    else:
        min_sat_temp = np.nanmin(vals_i_want)
    return min_sat_temp


def add_goes_data_to_tobac_path(tfm, client=None):
    def prepare_find_satellite(this_feature_time_idx):
        time = tfm.time.data[this_feature_time_idx]
        feature_indicies_at_time = np.nonzero(tfm.feature_time_index.data == this_feature_time_idx)[0]
        tfm_time = tfm.isel(feature=feature_indicies_at_time, time=this_feature_time_idx)
        this_timestep_sat_temps = np.full(tfm_time.feature.shape[0], np.nan)
        if this_feature_time_idx not in download_results['tobac_idx'].values:
            this_feat_time_dt = time.astype('datetime64[s]').astype(dt)
            print(f'>>>>>>>Warning, no satellite data for {this_feat_time_dt}>>>>>>>')
            return np.nan, 0, 0
        goes_file_path = download_results[download_results['tobac_idx'] == this_feature_time_idx]['file'].values[0]
        goes_file_path = path.join('/Volumes/LtgSSD/', goes_file_path)
        satellite_data = xr.open_dataset(goes_file_path).sel(y=goes_yslice, x=goes_xsclice)
        for j, feat_id in enumerate(tfm_time.feature.data):
            this_feat = tfm_time.sel(feature=feat_id)
            this_feature_time_idx = this_feat.feature_time_index.data.item()
            # Load satellite data for this index
            this_min_sat_temp = find_satellite_temp_for_feature(tfm_time, feat_id, satellite_data, feat_echotop=this_feat.feature_echotop.data.item())
            this_timestep_sat_temps[j] = this_min_sat_temp
        satellite_data.close()
        start_idx = feature_indicies_at_time[0]
        end_idx = feature_indicies_at_time[-1] + 1
        return this_timestep_sat_temps, start_idx, end_idx
    tfm = tfm.copy()
    min_sat_temp = np.full(tfm.feature.shape[0], np.nan)

    goes_time_range_start = tfm.time.data.astype('datetime64[s]').astype(dt)[0]
    goes_time_range_end = tfm.time.data.astype('datetime64[s]').astype(dt)[-1]
    goes_ctt = GOES(satellite=16, product='ABI-L2-MCMIPC')
    print('Start download')
    download_results = goes_ctt.timerange(goes_time_range_start-timedelta(minutes=15), goes_time_range_end+timedelta(minutes=15), max_cpus=4)
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
        print(f'>>>>>>>Warning, long gap between {gap[0]} and {gap[1]}.>>>>>>>')
    
    goes_max_x = tfm.g16_scan_x.max().data.item()
    goes_min_x = tfm.g16_scan_x.min().data.item()
    goes_max_y = tfm.g16_scan_y.max().data.item()
    goes_min_y = tfm.g16_scan_y.min().data.item()
    padding = .001
    goes_xsclice = slice(goes_min_x-padding, goes_max_x+padding)
    goes_yslice = slice(goes_max_y+padding, goes_min_y-padding)
    if client is not None:
        res = client.map(prepare_find_satellite, range(tfm.time.shape[0]))
        client.gather(res)
        for promised_res in res:
            d, s, e = promised_res.result()
            min_sat_temp[s:e] = d
    else:
        for this_feature_time_idx in range(tfm.time.shape[0]):
            d, s, e = prepare_find_satellite(this_feature_time_idx)
            min_sat_temp[s:e] = d
    tfm[f'min_L2-MCMIPC'] = xr.DataArray(min_sat_temp, dims=('feature'))
    return tfm
