#!/usr/bin/python3

from os import listdir, path
from pathlib import Path
from datetime import datetime as dt, timedelta
import sys

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


def generate_seg_mask_cell(tobac_data):
    tobac_data = tobac_data.copy()
    print('-Overwriting 0 in segmask with nan')
    tobac_data['segmentation_mask'] = xr.where(tobac_data.segmentation_mask == 0, np.nan, tobac_data.segmentation_mask.astype(np.float32)).compute()
    feature_ids = tobac_data.feature.data
    seg_data_feature = tobac_data.segmentation_mask.data
    cell_ids = tobac_data.feature_parent_cell_id.sel(feature=feature_ids).data.compute()
    feature_to_cell_map = dict(zip(feature_ids, cell_ids))
    seg_data_cell = seg_data_feature.copy()
    print('-Mapping')
    seg_data_cell = np.vectorize(feature_to_cell_map.get)(seg_data_cell)
    print('-Converting')
    seg_data_cell = seg_data_cell.astype(np.float32)
    print('-seg mask cell to xarray')
    tobac_data['segmentation_mask_cell'] = xr.DataArray(seg_data_cell, dims=('time', 'y', 'x'), coords={'time': tobac_data.time.data, 'y': tobac_data.y.data, 'x': tobac_data.x.data})
    return tobac_data


def add_eet_to_radar_data(tobac_day):
    radar_path = path.join(path.sep, 'Volumes', 'LtgSSD', 'nexrad_gridded', tobac_day.strftime('%B').upper(), tobac_day.strftime('%Y%m%d'))
    print('Finding elevations for all tobac grid points')
    radar_path_contents = sorted(listdir(radar_path))
    first_radar_file = path.join(radar_path, radar_path_contents[0])
    radar_data = xr.open_dataset(first_radar_file, chunks='auto')
    print('Starting processing...')

    for this_radar_path in radar_path_contents:
        this_radar_path = path.join(radar_path, this_radar_path)
        radar_data = xr.open_dataset(this_radar_path, chunks='auto')

        where18 = (radar_data.reflectivity >= 18)
        et_diff_from_top = 500*(where18.isel(z=slice(None, None, -1)).argmax(dim='z'))
        et_diff_from_top = xr.where(where18.any(dim='z'), et_diff_from_top, np.nan)
        echo_top = np.max(radar_data.z) - et_diff_from_top
        echo_top = xr.where(echo_top == np.max(radar_data.z), -1, echo_top)
        radar_data['eet_sam'] = echo_top
        save_path = this_radar_path.replace('nexrad_gridded', 'nexrad_zarr').replace('.nc', '.zarr')
        Path(path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        radar_data.to_zarr(save_path)


def add_eet_to_tobac_data(tfm, date_i_want):
    tfm = tfm.copy()
    radar_top_path = f'/Volumes/LtgSSD/nexrad_zarr/{date_i_want.strftime('%B').upper()}/{date_i_want.strftime('%Y%m%d')}/'
    radar_files = listdir(radar_top_path)
    radar_dts = [dt.strptime(rf, 'KHGX%Y%m%d_%H%M%S_V06_grid.zarr') for rf in radar_files if rf.endswith('.zarr')]
    radar_dts = np.array(radar_dts).astype('datetime64[s]').astype(dt)
    radar_files = [path.join(radar_top_path, rf) for rf in radar_files if rf.endswith('.zarr')]
    all_feature_eet = xr.zeros_like(tfm.feature)
    max_itr = len(tfm.feature.data)
    for i, feature_id in enumerate(tfm.feature.data):
        print(f'-{100*(i/max_itr):.1f}%')
        this_feature = tfm.sel(feature=feature_id)
        this_feature_dt = this_feature.feature_time.data.compute().astype('datetime64[s]').item()
        radar_path_i_want = radar_files[np.argmin(np.abs(radar_dts - this_feature_dt))]
        radar = xr.open_dataset(radar_path_i_want, engine='zarr', chunks='auto')
        this_eet = radar.eet_sam.isel(time=0)
        this_seg_mask = this_feature.sel(time=this_feature_dt, method='nearest').segmentation_mask
        feature_eet = np.nanmax(np.where(this_seg_mask.data == feature_id, this_eet, 0))
        all_feature_eet[i] = feature_eet
    tfm['feature_echotop'] = all_feature_eet
    return tfm


def add_timeseries_data_to_toabc_path(tobac_data, date_i_want):
    tobac_data = tobac_data.copy()
    tobac_save_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime('%Y%m%d')}/'
    for f in listdir(tobac_save_path):
        if f.startswith('timeseries_data_melt') and f.endswith('.nc'):
            tobac_timeseries_path = path.join(tobac_save_path, f)
            break
    else:
        print('>>>>>>>Unable to find timeseries data...>>>>>>>')
        return tobac_data
    timeseries_data = xr.open_dataset(tobac_timeseries_path, chunks='auto')
    timeseries_data = timeseries_data.reindex(feature=tobac_data.feature.data, fill_value=np.nan)
    for dv in timeseries_data.data_vars:
        if dv not in tobac_data.data_vars:
            tobac_data[dv] = timeseries_data[dv].copy()
    return tobac_data


def find_satellite_temp_for_feature(tfm_time, feature_i_want, area_i_want):
    # Find the index boundaries of the feature
    x_indices_valid, y_indices_valid = np.asarray(tfm_time.segmentation_mask == feature_i_want).nonzero()
    if len(x_indices_valid) == 0:
        return np.nan, np.nan, np.nan
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
    vals_i_want = vals[this_seg_mask == feature_i_want]
    if np.all(np.isnan(vals_i_want)):
        min_sat_temp = np.nan
        mean_sat_temp = np.nan
        std_sat_temp = np.nan
    else:
        min_sat_temp = np.nanmin(vals_i_want)
        mean_sat_temp = np.nanmean(vals_i_want)
        std_sat_temp = np.nanstd(vals_i_want)
    return min_sat_temp, mean_sat_temp, std_sat_temp


def add_goes_data_to_tobac_path(tfm):
    thing_to_plot =  'L2-MCMIPC'
    tfm = tfm.copy()
    min_sat_temp = np.full(tfm.feature.shape[0], np.nan)
    mean_sat_temp = np.full(tfm.feature.shape[0], np.nan)
    std_satellite_temp = np.full(tfm.feature.shape[0], np.nan)

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
        print(f'>>>>>>>Warning, long gap between {gap[0]} and {gap[1]}.>>>>>>>')
    
    max_iter = len(tfm.feature.data)
    goes_max_x = tfm.g16_scan_x.max().data.item()
    goes_min_x = tfm.g16_scan_x.min().data.item()
    goes_max_y = tfm.g16_scan_y.max().data.item()
    goes_min_y = tfm.g16_scan_y.min().data.item()
    padding = .001
    goes_xsclice = slice(goes_min_x-padding, goes_max_x+padding)
    goes_yslice = slice(goes_max_y+padding, goes_min_y-padding)
    for i, feat_id in enumerate(tfm.feature.data):
        this_feat = tfm.sel(feature=feat_id)
        this_feature_time_idx = this_feat.feature_time_index.data.compute().item()
        tfm_time = this_feat.isel(time=this_feature_time_idx)
        this_feat_time_dt = tfm_time.time.data.astype("datetime64[s]").astype(dt).item()
        print(this_feat_time_dt)
        print(f'({100*(i/max_iter):.1f}%) {this_feat_time_dt.strftime('%Y-%m-%d %H:%M:%S')}')
        # Load satellite data for this index
        if this_feature_time_idx not in download_results['tobac_idx'].values:
            print(f'>>>>>>>Warning, no satellite data for {this_feat_time_dt}>>>>>>>')
            continue
        goes_file_path = download_results[download_results['tobac_idx'] == this_feature_time_idx]['file'].values[0]
        goes_file_path = path.join('/Volumes/LtgSSD/', goes_file_path)
        satellite_data = xr.open_dataset(goes_file_path, chunks='auto').load()
        area_i_want = satellite_data.sel(y=goes_yslice, x=goes_xsclice)
        this_min_sat_temp, this_mean_sat_temp, this_std_sat_temp = find_satellite_temp_for_feature(tfm_time, feat_id, area_i_want)
        min_sat_temp[i] = this_min_sat_temp
        mean_sat_temp[i] = this_mean_sat_temp
        std_satellite_temp[i] = this_std_sat_temp

    tfm[f'min_{thing_to_plot}'] = xr.DataArray(min_sat_temp, dims=('feature'))
    tfm[f'avg_{thing_to_plot}'] = xr.DataArray(mean_sat_temp, dims=('feature'))
    tfm[f'std_{thing_to_plot}'] = xr.DataArray(std_satellite_temp, dims=('feature'))
    return tfm


if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    print('Reading')
    path_to_read = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime('%Y%m%d')}/Track_features_merges.nc'
    tobac_data = xr.open_dataset(path_to_read, chunks='auto')
    print('Applying coord transforms')
    tobac_data = apply_coord_transforms(tobac_data)
    print('Generating seg_mask_cell')
    tobac_data = generate_seg_mask_cell(tobac_data)
    if not path.exists(f'/Volumes/LtgSSD/nexrad_zarr/{date_i_want.strftime('%B').upper()}/{date_i_want.strftime('%Y%m%d')}'):
        print('I don\'t have EET for this day, computing it')
        add_eet_to_radar_data(date_i_want)
    print('Finding echo top heights')
    tobac_data = add_eet_to_tobac_data(tobac_data, date_i_want)
    print('Adding timeseries data to tobac data')
    tobac_data = add_timeseries_data_to_toabc_path(tobac_data, date_i_want)
    print('Adding satellite data to tobac data')
    tobac_data = add_goes_data_to_tobac_path(tobac_data)
    print('Saving')
    path_to_write = path_to_read.replace('.nc', '_augmented.zarr')
    tobac_data.chunk('auto').to_zarr(path_to_write)