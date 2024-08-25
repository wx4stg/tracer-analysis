from os import path
from datetime import datetime as dt

import holoviews as hv
import geoviews as gv
import panel as pn

import numpy as np
import xarray as xr

from goes2go improt GOES

from pyxlma import coords

def plot_satellite(dataset, time):
    index_of_tobac = (dataset.time.data == time).nonzero()[0][0]
    if index_of_tobac in download_results['tobac_idx'].values:
        path_of_sat = download_results[download_results['tobac_idx'] == index_of_tobac]['file'].values[0]
        alpha = 0.5
    else:
        print('No satellite data for this time...')
        path_of_sat = download_results[download_results['tobac_idx'] == 0]['file'].values[0]
        alpha = 0
    path_of_sat = path.join('/Users/stgardner4/data/', path_of_sat)
    max_x = dataset.g16_scan_x.max().data.item()
    min_x = dataset.g16_scan_x.min().data.item()
    max_y = dataset.g16_scan_y.max().data.item()
    min_y = dataset.g16_scan_y.min().data.item()
    sat = xr.open_dataset(path_of_sat)
    padding = .001
    area_i_want = sat.sel(y=slice(max_y+padding, min_y-padding), x=slice(min_x-padding, max_x+padding))
    geosys = coords.GeographicSystem()
    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=sat.nominal_satellite_subpoint_lon.data.item(), sweep_axis='x')
    this_satellite_scan_x, this_satellite_scan_y = np.meshgrid(area_i_want.x, area_i_want.y)
    sat_ecef_X, sat_ecef_Y, sat_ecef_Z = satsys.toECEF(this_satellite_scan_x.flatten(), this_satellite_scan_y.flatten(), np.zeros_like(this_satellite_scan_x.flatten()))
    sat_lon, sat_lat, _ = geosys.fromECEF(sat_ecef_X, sat_ecef_Y, sat_ecef_Z)
    sat_lon.shape = this_satellite_scan_x.shape
    sat_lat.shape = this_satellite_scan_y.shape
    if 'TEMP' not in area_i_want.data_vars:
        fk1 = 1.08033e+04
        fk2 = 1.39274e+03
        bc1 = 0.07550
        bc2 = 0.99975
        area_i_want['TEMP'] = (fk2 / (np.log((fk1 / area_i_want.Rad) + 1))  -  bc1) / bc2
    plot = gv.QuadMesh((sat_lon, sat_lat, area_i_want.TEMP.data), kdims=['lon', 'lat'], vdims=['Cloud Top Temperature']).opts(cmap='viridis', tools=['hover'], alpha=alpha)
    return plot


def get_time(utc_nanoseconds):
    if type(utc_nanoseconds) == np.datetime64:
        return utc_nanoseconds
    else:
        return np.datetime64(utc_nanoseconds, 'ns')


def get_time_str(utc_nanoseconds):
    time_str = dt.fromtimestamp(float(utc_nanoseconds) / 1e9, UTC).strftime('%Y-%m-%d %H:%M:%S')
    return time_str


def plot_seg_mask(dataset, time):
    this_time = dataset.sel(time=time, method='nearest')
    seg_mask = this_time.segmentation_mask.data.copy().astype(float)
    seg_mask[seg_mask == 0] = np.nan
    plot = gv.QuadMesh((this_time.lon, this_time.lat, seg_mask), kdims=['lon', 'lat'], vdims=['segmentation mask']).opts(cmap='plasma', colorbar=False)
    return plot

def plot_features(dataset, time):
    lats = dataset.feature_lat[dataset.feature_time == time]
    lons = dataset.feature_lon[dataset.feature_time == time]
    feat_ids = dataset.feature[dataset.feature_time == time]
    return gv.Points((lons, lats, feat_ids), kdims=['lon', 'lat'], vdims=['Feature ID']).opts(color='Cell ID', cmap='plasma', colorbar=True, tools=['hover'], size=4, line_color='k', line_width=0.5)


def get_satellite_data_for_time_range(time_start, time_end):
    
    

if __name__ == '__main__':
    tfm = xr.open_dataset('tobac_15/tobac_Save_20220601/Track_features_merges_with_goes.nc')
    unique_times = np.unique(tfm.time.data).astype('datetime64[us]').astype(dt)
    date_slider = pn.widgets.DiscreteSlider(name='Date', options=unique_times, value=unique_times[0])
    live_time_str = pn.bind(get_time_str, date_slider)
    live_time = pn.bind(get_time, date_slider)
    date_label = pn.pane.Markdown(live_time_str)
    mesh = gv.DynamicMap(pn.bind(plot_seg_mask, tfm, live_time))
    satellite = gv.DynamicMap(pn.bind(plot_satellite, tfm, live_time))
    feature_points = gv.DynamicMap(pn.bind(plot_features, tfm, live_time))
    my_map = (gv.tile_sources.OSM * mesh * satellite * feature_points.opts(alpha=0.7)).opts(width=800, height=800)
    col = pn.Column(date_slider, date_label, my_map)