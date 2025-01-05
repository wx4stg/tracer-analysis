#!/usr/bin/env python3

import geopandas as gpd
import xarray as xr
from datetime import datetime as dt
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon
from pyxlma import coords
import sys

from associate_obs import apply_coord_transforms

from numba import njit

@njit
def interpolate_sbf_polygons(longer_polyline, shorter_polyline, times_i_want_between_float, last_time_float, this_time_float, later_is_longer):
    interpolated_polys = np.zeros((longer_polyline.shape[0], 2, times_i_want_between_float.shape[0]))
    # Pair each vertex of the longer polyline with the closest vertex of the shorter polyline
    for i in range(longer_polyline.shape[0]):
        # Find distance between this point and all points in the shorter polyline
        this_point = longer_polyline[i, :]
        distances = (np.sum(((shorter_polyline - this_point)**2), axis=1))**(0.5)
        # Pair this point with the closest point in the shorter polyline
        point_pair = np.argmin(distances)
        matching_point = shorter_polyline[point_pair, :]
        # Interpolate so that the point moves from the "this point" (aka, longer polyline) to the "matching point" (aka, shorter polyline)
        moving_x = this_point[0] + (times_i_want_between_float - last_time_float) * (matching_point[0] - this_point[0]) / (this_time_float - last_time_float)
        moving_y = this_point[1] + (times_i_want_between_float - last_time_float) * (matching_point[1] - this_point[1]) / (this_time_float - last_time_float)
        point_in_motion = np.empty((2, moving_x.shape[0]))
        point_in_motion[0, :] = moving_x
        point_in_motion[1, :] = moving_y
        # If the longer polygon is later in time, flip the motion of the point so that it moves from "previous time" to "now"
        if later_is_longer:
            interpolated_polys[i, :, :] = point_in_motion[:, ::-1]
        else:
            interpolated_polys[i, :, :] = point_in_motion
    # Duplicate the first point of the interpolated polygon to the last point so that it's a closed polygon
    interpolated_polys_repeat = np.concatenate((interpolated_polys, interpolated_polys[0:1, :, :]), axis=0)
    return interpolated_polys_repeat


if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    tfm_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges.nc'
    tfm = xr.open_dataset(tfm_path, chunks='auto')
    tfm = apply_coord_transforms(tfm)
    polyline_path = f'/Volumes/LtgSSD/analysis/sam_polyline/{date_i_want.strftime("%Y-%m-%d")}.json'
    polyline = gpd.read_file(polyline_path)
    polyline = polyline.set_index('index')

    seabreeze_indices = np.where(np.isin(tfm.time.data.astype('datetime64[s]').astype(dt), polyline.index.values))
    times_i_have = tfm.time.data[seabreeze_indices].copy()
    times_i_want = tfm.time.data.copy()
    times_i_want = times_i_want[(times_i_want > times_i_have[0]) & (times_i_want < times_i_have[-1])]

    print('Interpolating polygons')
    for i, this_time in enumerate(times_i_have[1:]):
        # Get times in a whole bunch of different formats
        this_time_dt = this_time.astype('datetime64[s]').astype(dt)
        this_time_float = this_time.astype(float)
        last_time = times_i_have[i]
        last_time_dt = last_time.astype('datetime64[s]').astype(dt)
        last_time_float = last_time.astype(float)
        
        # Find desired times that are between this_time and last_time
        times_i_want_between = times_i_want[(times_i_want > last_time) & (times_i_want < this_time)]
        times_i_want_between_dt = times_i_want_between.astype('datetime64[s]').astype(dt)
        times_i_want_between_float = times_i_want_between.astype(float)

        # Get coordinates of this polyline and the last polyline
        this_polyline = polyline[polyline.index == this_time_dt]['geometry'].values[0]
        this_polyline_coords = np.array(this_polyline.exterior.coords)[:-1, :]
        last_polyline = polyline[polyline.index == last_time_dt]['geometry'].values[0]
        last_polyline_coords = np.array(last_polyline.exterior.coords)[:-1, :]
        
        # Determine which polyline has more vertices
        later_is_longer = False
        if this_polyline_coords.shape[0] >= last_polyline_coords.shape[0]:
            longer_polyline = this_polyline_coords
            shorter_polyline = last_polyline_coords
            later_is_longer = True
        else:
            longer_polyline = last_polyline_coords
            shorter_polyline = this_polyline_coords
        
        
        interpolated_polys = interpolate_sbf_polygons(longer_polyline, shorter_polyline, times_i_want_between_float, last_time_float, this_time_float, later_is_longer)
        for ti, t in enumerate(times_i_want_between_dt):
            this_interp_poly = interpolated_polys[:, :, ti]
            interp_poly_shape = Polygon(this_interp_poly)
            polyline.loc[t, 'geometry'] = interp_poly_shape


    start_time = np.min(polyline.index.values)
    end_time = np.max(polyline.index.values)
    lon_wide_1d = np.arange(-98.3, -91+.005, .01)
    lat_wide_1d = np.arange(25.5, 30+.005, .01)
    lon_wide, lat_wide = np.meshgrid(lon_wide_1d, lat_wide_1d)
    all_seabreezes_wide = np.full((lon_wide.shape[0], lon_wide.shape[1], tfm.time.shape[0]), -2, dtype=int)

    radar_lat, radar_lon = tfm.attrs['center_lat'], tfm.attrs['center_lon']
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=radar_lat, ctrLon=radar_lon, ctrAlt=0)
    geosys = coords.GeographicSystem()
    x2d, y2d = np.meshgrid(tfm.x.data, tfm.y.data)
    grid_ecef_coords = tpcs.toECEF(x2d.flatten(), y2d.flatten(), np.zeros_like(x2d).flatten())
    grid_lon, grid_lat, _ = geosys.fromECEF(*grid_ecef_coords)
    grid_lon = grid_lon.reshape(x2d.shape)
    grid_lat = grid_lat.reshape(x2d.shape)

    tfm = tfm.assign({'lat' : (('x', 'y'), grid_lat), 'lon' : (('x', 'y'), grid_lon)})

    all_seabreezes_ds = xr.full_like(tfm.segmentation_mask, -2).astype(int)
    print('Gridding...')
    for i, time in enumerate(tfm.time.data):
        if time < start_time:
            time = start_time
        elif time > end_time:
            time = end_time
        this_seabreeze = np.zeros_like(lon_wide)
        time_dt = np.array(time).astype('datetime64[s]').astype(dt).item()
        if time_dt in polyline.index.values.astype(dt):
            this_polyline = polyline[polyline.index == time_dt]['geometry'].values[0]
            this_polyline_mpl = Path(np.array(this_polyline.exterior.coords))
            this_seabreeze = this_polyline_mpl.contains_points(np.array([lon_wide.flatten(), lat_wide.flatten()]).T).reshape(lon_wide.shape)
            all_seabreezes_wide[:, :, i] = this_seabreeze.astype('float32') - 2
            this_seabreeze_ds = this_polyline_mpl.contains_points(np.array([grid_lon.flatten(), grid_lat.flatten()]).T).reshape(grid_lon.shape)
            all_seabreezes_ds[i, :, :] = (this_seabreeze_ds.astype('float32') - 2).T
        else:
            raise ValueError(f'No polyline for {time_dt}')

    print('Saving wide dataset...')
    wide_ds = xr.DataArray(
        all_seabreezes_wide,
        dims=('latitude', 'longitude', 'time'),
        coords={'latitude': lat_wide_1d, 'longitude': lon_wide_1d, 'time': tfm.time}
    ).to_dataset(name='seabreeze')
    comp = dict(zlib=True, complevel=5)
    enc = {var: comp for var in wide_ds.data_vars if not np.issubdtype(wide_ds[var].dtype, str)}
    wide_ds.to_netcdf(polyline_path.replace('.json', '_seabreeze.nc').replace('sam_polyline/', 'sam_sbf/'), encoding=enc)

    tfm['seabreeze'] = all_seabreezes_ds
    print('Identifying features...')
    feature_seabreeze = xr.zeros_like(tfm.feature, dtype=int)
    for i, feat_id in enumerate(tfm.feature.data):
        this_feat = tfm.sel(feature=feat_id)
        this_feat_time_idx = this_feat.feature_time_index.compute().data.item()
        this_feat_time = tfm.time.data[this_feat_time_idx].astype('datetime64[s]').astype(dt)
        if this_feat_time < start_time:
            this_feat_time = start_time
        elif this_feat_time > end_time:
            this_feat_time = end_time
        this_feat_lon = this_feat.feature_lon.compute().data.item()
        this_feat_lat = this_feat.feature_lat.compute().data.item()
        this_polyline = polyline[polyline.index.values == this_feat_time]['geometry'].values[0]
        this_polyline_mpl = Path(np.array(this_polyline.exterior.coords))
        this_seabreeze = int(this_polyline_mpl.contains_point((this_feat_lon, this_feat_lat))) - 2
        feature_seabreeze.data[i] = this_seabreeze

    tfm['feature_seabreeze'] = feature_seabreeze
    print('Saving zarr...')
    tfm.chunk('auto').to_zarr(tfm_path.replace('Track_features_merges.nc', 'seabreeze.zarr'))

    polyline = polyline.sort_index()
    polyline.to_file(polyline_path.replace('.json', '_interpolated.json'), driver='GeoJSON')
