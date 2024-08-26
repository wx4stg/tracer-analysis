from os import path
from datetime import datetime as dt, timedelta

import holoviews as hv
import geoviews as gv
import panel as pn

import numpy as np
import xarray as xr

from goes2go import GOES

from pyxlma import coords

hv.extension('bokeh')

def plot_satellite(dl_res, time, min_x, max_x, min_y, max_y):
    closest_time = dl_res.iloc[(dl_res['valid'] - time).abs().argsort()[:1]]
    sat = xr.open_dataset(closest_time['path'].values[0])
    padding = .001
    area_i_want = sat.sel(y=slice(max_y+padding, min_y-padding), x=slice(min_x-padding, max_x+padding))
    geosys = coords.GeographicSystem()
    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=sat.nominal_satellite_subpoint_lon.data.item(), sweep_axis='x')
    this_satellite_scan_x, this_satellite_scan_y = np.meshgrid(area_i_want.x, area_i_want.y)
    sat_ecef_X, sat_ecef_Y, sat_ecef_Z = satsys.toECEF(this_satellite_scan_x.flatten(), this_satellite_scan_y.flatten(), np.zeros_like(this_satellite_scan_x.flatten()))
    sat_lon, sat_lat, _ = geosys.fromECEF(sat_ecef_X, sat_ecef_Y, sat_ecef_Z)
    sat_lon.shape = this_satellite_scan_x.shape
    sat_lat.shape = this_satellite_scan_y.shape
    plot = gv.QuadMesh((sat_lon, sat_lat, area_i_want.CMI_C13.data), kdims=['lon', 'lat'], vdims=['IR Brightness Temperature']).opts(cmap='viridis', tools=['hover'])
    return plot


def plot_seg_mask(dataset, time):
    this_time = dataset.sel(time=time, method='nearest')
    plot = gv.QuadMesh((this_time.lon.data, this_time.lat.data, this_time.segmentation_mask.data), kdims=['lon', 'lat'], vdims=['segmentation mask']).opts(cmap='plasma', colorbar=False)
    return plot

def plot_features(dataset, time):
    lats = dataset.feature_lat[dataset.feature_time == time]
    lons = dataset.feature_lon[dataset.feature_time == time]
    feat_ids = dataset.feature[dataset.feature_time == time]
    return gv.Points((lons, lats, feat_ids), kdims=['lon', 'lat'], vdims=['Feature ID']).opts(color='Cell ID', cmap='plasma', colorbar=True, tools=['hover'], size=4, line_color='k', line_width=0.5)


if __name__ == '__main__':
    tfm = xr.open_dataset('tobac_15/tobac_Save_20220601/Track_features_merges_augmented.nc')
    tfm.segmentation_mask.data = tfm.segmentation_mask.data.astype(float)
    tfm.segmentation_mask.data[tfm.segmentation_mask.data == 0] = np.nan
    
    sat_min_x = tfm.g16_scan_x.min().data.item()
    sat_max_x = tfm.g16_scan_x.max().data.item()
    sat_min_y = tfm.g16_scan_y.min().data.item()
    sat_max_y = tfm.g16_scan_y.max().data.item()

    g16 = GOES(satellite=16, product='ABI-L2-MCMIPC')
    goes_time_range_start = tfm.time.data.astype('datetime64[us]').astype(dt).min()
    goes_time_range_end = tfm.time.data.astype('datetime64[us]').astype(dt).max()
    download_results = g16.timerange(goes_time_range_start-timedelta(minutes=15), goes_time_range_end+timedelta(minutes=15), max_cpus=12)
    download_results['valid'] = download_results[['start', 'end']].mean(axis=1)
    download_results['path'] = '/Volumes/LtgSSD/' + download_results['file'].values
    unique_times = np.unique(tfm.time.data).astype('datetime64[us]').astype(dt).tolist()
    date_slider = pn.widgets.Select(name='Date', options=unique_times, value=unique_times[0])
    mesh = gv.DynamicMap(pn.bind(plot_seg_mask, tfm, date_slider.value))
    satellite = gv.DynamicMap(pn.bind(plot_satellite, download_results, date_slider.value,
                                      sat_min_x, sat_max_x, sat_min_y, sat_max_y))
    feature_points = gv.DynamicMap(pn.bind(plot_features, tfm, date_slider.value))
    my_map = (gv.tile_sources.OSM *
              mesh.opts(tools=['hover']) *
              feature_points.opts(alpha=0.7, tools=[]) *
              satellite
              ).opts(width=800, height=800)
    col = pn.Column(date_slider, my_map)
    pn.serve(col, port=5006)