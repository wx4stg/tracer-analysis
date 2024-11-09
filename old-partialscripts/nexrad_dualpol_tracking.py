#!/usr/bin/env python3
import xarray as xr
import numpy as np
from dask import array as da
from os import path, listdir, remove
from datetime import datetime as dt, timedelta
from pathlib import Path

from pyxlma import coords

USE_DASK = True
base_path = path.dirname(__file__)


def fixup_transformed_coords(transformed_coords, input_shape):
    for transformed in transformed_coords:
        if not isinstance(transformed, np.ndarray):
            transformed = da.atleast_1d(transformed)
        transformed = transformed.reshape(input_shape)
    return transformed_coords

def add_eet_to_radar_data(tobac_day):
    radar_path = path.join(path.sep, 'Volumes', 'LtgSSD', 'nexrad_gridded', tobac_day.strftime('%B').upper(), tobac_day.strftime('%Y%m%d'))
    # dbz_gridded = xr.full_like(tobac_data.segmentation_mask, np.nan, dtype=np.float32)
    # rhohv_grid = dbz_gridded.copy()
    # zdr_grid = dbz_gridded.copy()
    # kdp_grid = dbz_gridded.copy()
    # median_volume_diameter_grid = dbz_gridded.copy()
    # liquid_water_mass_grid = dbz_gridded.copy()
    # ice_water_mass_grid = dbz_gridded.copy()
    print('Finding elevations for all tobac grid points')
    radar_path_contents = sorted(listdir(radar_path))
    first_radar_file = path.join(radar_path, radar_path_contents[0])
    radar_data = xr.open_dataset(first_radar_file, chunks='auto')
    
    # tpcs = coords.TangentPlaneCartesianSystem(radar_data.origin_latitude.data.compute()[0], radar_data.origin_longitude.data.compute()[0], radar_data.origin_altitude.data.compute()[0])
    # rcs = coords.RadarCoordinateSystem(radar_data.origin_latitude.data.compute()[0], radar_data.origin_longitude.data.compute()[0], radar_data.origin_altitude.data.compute()[0])
    # x3d, y3d, z3d = np.meshgrid(radar_data.x.data, radar_data.y.data, radar_data.z.data)
    # grid_X, grid_Y, grid_Z = tpcs.toECEF(x3d.flatten(), y3d.flatten(), z3d.flatten())
    # grid_r, grid_az, grid_el = rcs.fromECEF(grid_X, grid_Y, grid_Z)
    # grid_el = grid_el.reshape(x3d.shape)

    # grid_el_closest_05 = grid_el > 1

    # grid_el_closest_05 = np.argmax(grid_el_closest_05[:, :, 1:], axis=2)+1


    print('Starting processing...')

    for this_radar_path in radar_path_contents:
        this_radar_path = path.join(radar_path, this_radar_path)
        radar_data = xr.open_dataset(this_radar_path, chunks='auto')

        where18 = (radar_data.reflectivity >= 18)
        et_diff_from_top = 500*(where18.isel(z=slice(None, None, -1)).argmax(dim='z'))
        et_diff_from_top = xr.where(where18.any(dim='z'), et_diff_from_top, np.nan)
        echo_top = da.max(radar_data.z) - et_diff_from_top
        echo_top = xr.where(echo_top == da.max(radar_data.z), -1, echo_top)
        # this_radar_time = radar_data.isel(time=0)
        radar_data['eet_sam'] = echo_top
        # radar_data['lowest_refl'] = radar_data.reflectivity.isel(time=0).isel(z=xr.DataArray(grid_el_closest_05, dims=('y', 'x')))

        # dbz_gridded[i, :, :] = this_radar_time.reflectivity.data
        # dbz_gridded = dbz_gridded.chunk('auto')
        # rhohv_grid[i, :, :] = this_radar_time.cross_correlation_ratio.data
        # rhohv_grid = rhohv_grid.chunk('auto')
        # zdr_grid[i, :, :] = this_radar_time.differential_reflectivity.data
        # zdr_grid = zdr_grid.chunk('auto')
        # kdp_grid[i, :, :] = this_radar_time.KDP_CSU.data
        # kdp_grid = kdp_grid.chunk('auto')
        # median_volume_diameter_grid[i, :, :] = this_radar_time.D0.data
        # median_volume_diameter_grid = median_volume_diameter_grid.chunk('auto')
        # liquid_water_mass_grid[i, :, :] = this_radar_time.MU.data
        # liquid_water_mass_grid = liquid_water_mass_grid.chunk('auto')
        # ice_water_mass_grid[i, :, :] = this_radar_time.MI.data
        # ice_water_mass_grid = ice_water_mass_grid.chunk('auto')
        save_path = this_radar_path.replace('nexrad_gridded', 'nexrad_zarr').replace('.nc', '.zarr')
        Path(path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        radar_data.to_zarr(save_path)
    # tobac_data['reflectivity_grid'] = dbz_gridded
    # tobac_data['rhohv_grid'] = rhohv_grid
    # tobac_data['zdr_grid'] = zdr_grid
    # tobac_data['kdp_grid'] = kdp_grid
    # tobac_data['median_volume_diameter_grid'] = median_volume_diameter_grid
    # tobac_data['liquid_water_mass_grid'] = liquid_water_mass_grid
    # tobac_data['ice_water_mass_grid'] = ice_water_mass_grid

def add_timeseries_data_to_toabc_path(tobac_path):
    tobac_data = xr.open_dataset(tobac_path, chunks='auto')
    tobac_times = tobac_data.time.data.astype('datetime64[s]').astype(dt)
    tobac_save_path = path.dirname(tobac_path)
    for f in listdir(tobac_save_path):
        if f.startswith('timeseries_data_melt') and f.endswith('.nc'):
            tobac_timeseries_path = path.join(tobac_save_path, f)
            break
    else:
        return
    timeseries_data = xr.open_dataset(tobac_timeseries_path, chunks='auto')
    timeseries_data = timeseries_data.reindex(feature=tobac_data.feature.data, fill_value=np.nan)
    for dv in timeseries_data.data_vars:
        if dv not in tobac_data.data_vars:
            tobac_data[dv] = timeseries_data[dv].copy()
    out_path = path.join(tobac_save_path, 'Track_features_merges_augmented2.zarr')
    tobac_data.chunk('auto').to_zarr(out_path)


if __name__ == '__main__':
    if USE_DASK:
        from dask.distributed import Client
        client = Client('tcp://127.0.0.1:8786')
    tobac_main_path = path.join(path.sep, 'Volumes', 'LtgSSD', 'tobac_saves')
    tobac_dayfiles = []
    dayfile_days = []
    for d in sorted(listdir(tobac_main_path)):
        if path.isdir(path.join(tobac_main_path, d)):
            if path.exists(path.join(tobac_main_path, d, 'Track_features_merges_augmented_cell.nc')):
                tobac_dayfiles.append(path.join(tobac_main_path, d, 'Track_features_merges_augmented_cell.nc'))
                dayfile_days.append(dt.strptime(d.split('_')[-1], '%Y%m%d'))
            else:
                if path.exists(path.join(tobac_main_path, d, 'Track_features_merges.nc')):
                    pass
                    # tobac_dayfiles.append(path.join(tobac_main_path, d, 'Track_features_merges.nc'))
                else:
                    print(f'Need to tobac track for day: {d}')
    
    for i, f in enumerate(tobac_dayfiles):
        # add_timeseries_data_to_toabc_path(f)
        add_eet_to_radar_data(dayfile_days[i])