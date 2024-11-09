#!/usr/bin/env python3

import geopandas as gpd
import xarray as xr
from datetime import datetime as dt
from os import path
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import interp1d
from pyxlma import coords
import sys


def interp_seabreeze_times(all_seabreezes, seabreeze_indices, seabreeze_times_num, zero_indices, zero_times_num):
    all_seabreezes = all_seabreezes.copy()
    # Interpolation: loop through each grid point (lat, lon) and interpolate over time
    for lat_idx in range(all_seabreezes.shape[0]):
        for lon_idx in range(all_seabreezes.shape[1]):
            # Get the seabreeze field values at valid times for this (lat, lon) grid point
            seabreeze_values = all_seabreezes[lat_idx, lon_idx, seabreeze_indices]
            
            # Skip interpolation if all values are zero
            if np.all(seabreeze_values == -2):
                continue
            
            # Create interpolator for the seabreeze field values based on actual times
            interpolator = interp1d(
                seabreeze_times_num,
                seabreeze_values,
                kind='linear',
                bounds_error=False,
                fill_value=-2  # You can choose another fill method if appropriate
            )
            
            # Interpolate for missing times (all-zero slices)
            interpolated_values = interpolator(zero_times_num)
            
            # Update all_seabreezes with interpolated values at zero_indices
            all_seabreezes[lat_idx, lon_idx, zero_indices] = interpolated_values
    return all_seabreezes

if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    tfm_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges_augmented.zarr'
    tfm = xr.open_dataset(tfm_path, engine='zarr', chunks='auto')
    polyline_path = f'/Volumes/LtgSSD/analysis/sam_polyline/{date_i_want.strftime("%Y-%m-%d")}.json'
    polyline = gpd.read_file(polyline_path)

    lon_wide_1d = np.arange(-98.3, -91+.005, .01)
    lat_wide_1d = np.arange(25.5, 30+.005, .01)
    lon_wide, lat_wide = np.meshgrid(lon_wide_1d, lat_wide_1d)
    all_seabreezes_wide = np.full((lon_wide.shape[0], lon_wide.shape[1], tfm.time.shape[0]), -2, dtype='float32')
    seabreeze_indices = []
    seabreeze_times = []
    zero_indices = []
    zero_times = []

    radar_lat, radar_lon = tfm.attrs['center_lat'], tfm.attrs['center_lon']
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=radar_lat, ctrLon=radar_lon, ctrAlt=0)
    geosys = coords.GeographicSystem()
    x2d, y2d = np.meshgrid(tfm.x.data, tfm.y.data)
    grid_ecef_coords = tpcs.toECEF(x2d.flatten(), y2d.flatten(), np.zeros_like(x2d).flatten())
    grid_lon, grid_lat, _ = geosys.fromECEF(*grid_ecef_coords)
    grid_lon = grid_lon.reshape(x2d.shape)
    grid_lat = grid_lat.reshape(x2d.shape)

    tfm = tfm.assign({'lat' : (('x', 'y'), grid_lat), 'lon' : (('x', 'y'), grid_lon)})

    all_seabreezes_ds = xr.full_like(tfm.segmentation_mask, -2)
    for i, time in enumerate(tfm.time.data):
        this_seabreeze = np.zeros_like(lon_wide)
        time_dt = np.array(time).astype('datetime64[s]').astype(dt).item()
        if time_dt in polyline['index'].values.astype(dt):
            seabreeze_indices.append(i)
            seabreeze_times.append(time)
            this_polyline = polyline[polyline['index'] == time_dt]['geometry'].values[0]
            this_polyline_mpl = Path(np.array(this_polyline.exterior.coords))
            this_seabreeze = this_polyline_mpl.contains_points(np.array([lon_wide.flatten(), lat_wide.flatten()]).T).reshape(lon_wide.shape)
            all_seabreezes_wide[:, :, i] = this_seabreeze.astype('float32') - 2
            this_seabreeze_ds = this_polyline_mpl.contains_points(np.array([grid_lon.flatten(), grid_lat.flatten()]).T).reshape(grid_lon.shape)
            all_seabreezes_ds[i, :, :] = this_seabreeze_ds.astype('float32') - 2
        else:
            zero_indices.append(i)
            zero_times.append(time)
    seabreeze_indices = np.array(seabreeze_indices)
    seabreeze_times = np.array(seabreeze_times)
    zero_indices = np.array(zero_indices)
    zero_times = np.array(zero_times)

    # Convert seabreeze_times and zero_times to numeric for interpolation
    seabreeze_times_num = seabreeze_times.astype('float64')
    zero_times_num = zero_times.astype('float64')

    interp_wide = interp_seabreeze_times(all_seabreezes_wide, seabreeze_indices, seabreeze_times_num, zero_indices, zero_times_num).astype('float32')
    wide_ds = xr.DataArray(
        interp_wide,
        dims=('latitude', 'longitude', 'time'),
        coords={'latitude': lat_wide_1d, 'longitude': lon_wide_1d, 'time': tfm.time}
    ).to_dataset(name='seabreeze')
    comp = dict(zlib=True, complevel=5)
    enc = {var: comp for var in wide_ds.data_vars if not np.issubdtype(wide_ds[var].dtype, str)}
    wide_ds.to_netcdf(polyline_path.replace('.json', '_seabreeze.nc').replace('sam_polyline/', 'sam_sbf/'), encoding=enc)

    all_seabreezes_ds = interp_seabreeze_times(all_seabreezes_ds.T.compute(), seabreeze_indices, seabreeze_times_num, zero_indices, zero_times_num).T.astype('float32')

    tfm['seabreeze'] = all_seabreezes_ds

    feature_seabreeze = xr.zeros_like(tfm.feature)
    cell_seabreeze = np.zeros_like(np.meshgrid(tfm.cell, tfm.time))
    for i, feat_id in enumerate(tfm.feature.data):
        this_feat = tfm.sel(feature=feat_id)
        this_feat_lon = this_feat.feature_lon.compute()
        this_feat_lat = this_feat.feature_lat.compute()
        ll_dist = ((tfm.lon - this_feat_lon)**2 + (tfm.lat - this_feat_lat)**2)**(0.5)
        ll_dist_min = np.unravel_index(np.argmin(ll_dist.data.flatten()), ll_dist.shape)
        this_feat_time_index = this_feat.feature_time_index.data.compute().item()
        closest_seabreeze = tfm.seabreeze.isel(time=this_feat_time_index)[ll_dist_min]
        feature_seabreeze[i] = closest_seabreeze

    tfm['feature_seabreeze'] = feature_seabreeze

    tfm.chunk('auto').to_zarr(tfm_path.replace('Track_features_merges_augmented.zarr', 'seabreeze.zarr'))
