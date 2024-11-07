#!/usr/bin/python3

from os import listdir, path
from pathlib import Path

import numpy as np
import xarray as xr
import sys
from datetime import datetime as dt


def fixup_transformed_coords(transformed_coords, input_shape):
    for transformed in transformed_coords:
        if not isinstance(transformed, np.ndarray):
            transformed = np.atleast_1d(transformed)
        transformed = transformed.reshape(input_shape)
    return transformed_coords


def generate_seg_mask_cell(tobac_data):
    tobac_data = tobac_data.copy()
    print('Overwriting 0 in segmask with nan')
    tobac_data['segmentation_mask'] = xr.where(tobac_data.segmentation_mask == 0, np.nan, tobac_data.segmentation_mask.astype(np.float32))
    feature_ids = tobac_data.feature.data.compute()
    seg_data_feature = tobac_data.segmentation_mask.data.compute()
    cell_ids = tobac_data.feature_parent_cell_id.sel(feature=feature_ids).data.compute()
    feature_to_cell_map = dict(zip(feature_ids, cell_ids))
    seg_data_cell = seg_data_feature.copy()
    print('Mapping')
    seg_data_cell = np.vectorize(feature_to_cell_map.get)(seg_data_cell)
    print('Filtering')
    seg_data_cell[seg_data_cell == None] = np.nan
    print('Converting')
    seg_data_cell = seg_data_cell.astype(np.float32)
    print('Saving')
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
    for i, feature_id in enumerate(tfm.feature.data):
        print(feature_id)
        this_feature = tfm.sel(feature=feature_id)
        this_feature_dt = this_feature.feature_time.data.compute().astype('datetime64[s]').item()
        radar_path_i_want = radar_files[np.argmin(np.abs(radar_dts - this_feature_dt))]
        radar = xr.open_dataset(radar_path_i_want, engine='zarr', chunks='auto')
        this_eet = radar.eet_sam.isel(time=0)
        this_seg_mask = this_feature.sel(time=this_feature_dt, method='nearest').segmentation_mask
        feature_eet = np.nanmax(np.where(this_seg_mask.data == feature_id, this_eet, 0)).compute()
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
    timeseries_data = timeseries_data.reindex(feature=tobac_data.feature.data.compute(), fill_value=np.nan)
    for dv in timeseries_data.data_vars:
        if dv not in tobac_data.data_vars:
            tobac_data[dv] = timeseries_data[dv].copy()
    return tobac_data


if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    print('Reading')
    tobac_data = xr.open_dataset(f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime('%Y%m%d')}/Track_features_merges.nc', chunks='auto')
    print('Generating seg_mask_cell')
    tobac_data = generate_seg_mask_cell(tobac_data)
    if not path.exists(f'/Volumes/LtgSSD/nexrad_zarr/{date_i_want.strftime('%B').upper()}/{date_i_want.strftime('%Y%m%d')}'):
        print('I don\'t have EET for this day, computing it')
        add_eet_to_radar_data(date_i_want)
    print('Finding echo top heights')
    tobac_data = add_eet_to_tobac_data(tobac_data, date_i_want)
    print('Adding timeseries data to tobac data')
    tobac_data = add_timeseries_data_to_toabc_path(tobac_data, date_i_want)
