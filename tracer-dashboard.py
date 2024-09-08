from os import path
from datetime import datetime as dt, timedelta

import holoviews as hv
import geoviews as gv
import panel as pn

import numpy as np
import xarray as xr

from goes2go import GOES

from pyxlma import coords
from glmtools.io.lightning_ellipse import lightning_ellipse_rev
import cmweather

hv.extension('bokeh')

def plot_satellite(dl_res, time, min_x, max_x, min_y, max_y, channel_select, satellite_tick):
    closest_time = dl_res.iloc[(dl_res['valid'] - time).abs().argsort()[:1]]
    sat = xr.open_dataset(closest_time['path'].values[0])
    padding = .001
    area_i_want = sat.sel(y=slice(max_y+padding, min_y-padding), x=slice(min_x-padding, max_x+padding))
    geosys = coords.GeographicSystem()
    ltg_ell = lightning_ellipse_rev[1]
    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=sat.nominal_satellite_subpoint_lon.data.item(), sweep_axis='x', ellipse=ltg_ell)
    this_satellite_scan_x, this_satellite_scan_y = np.meshgrid(area_i_want.x, area_i_want.y)
    sat_ecef_X, sat_ecef_Y, sat_ecef_Z = satsys.toECEF(this_satellite_scan_x.flatten(), this_satellite_scan_y.flatten(), np.zeros_like(this_satellite_scan_x.flatten()))
    sat_lon, sat_lat, _ = geosys.fromECEF(sat_ecef_X, sat_ecef_Y, sat_ecef_Z)
    sat_lon.shape = this_satellite_scan_x.shape
    sat_lat.shape = this_satellite_scan_y.shape
    if channel_select == 'CMI_C02':
        cmap_to_use = 'Greys'
    else:
        cmap_to_use = 'viridis'
    plot = gv.QuadMesh((sat_lon, sat_lat, area_i_want.CMI_C13.data), kdims=['Longitude', 'Latitude'], vdims=[channel_select]).opts(cmap=cmap_to_use, visible=satellite_tick)
    return plot


def plot_radar(dataset, time, radar_selector, z_selector, radar_tick):
    print(time)
    this_time = dataset.sel(time=time, method='nearest')
    print(this_time.time.data.astype('datetime64[us]').astype(dt).item())
    lons = this_time.lon.data
    lats = this_time.lat.data
    clim_to_use = None
    if radar_selector == 'Reflectivity':
        data2plot = this_time.reflectivity
        cmap_to_use = 'ChaseSpectral'
        clim_to_use = (-10, 80)
    elif radar_selector == 'RhoHV':
        data2plot = this_time.cross_correlation_ratio
        cmap_to_use = 'plasmidis'
        clim_to_use = (0, 1)
    elif radar_selector == 'ZDR':
        data2plot = this_time.differential_reflectivity
        cmap_to_use = 'turbone'
        clim_to_use = (-2, 8)
    data2plot = data2plot.max(dim='z').data.compute()
    plot = gv.QuadMesh((lons, lats, data2plot), kdims=['Longitude', 'Latitude'], vdims=[radar_selector]).opts(
        cmap=cmap_to_use, colorbar=False, tools=['hover'], visible=radar_tick)
    if clim_to_use is not None:
        plot = plot.opts(clim=clim_to_use)
    return plot

def plot_seg_mask(dataset, time, seg_selector, seg_tick):
    this_time = dataset.sel(time=time)
    lons = this_time.lon.data
    lats = this_time.lat.data
    if seg_selector == 'Feature ID':
        seg_mask = this_time.segmentation_mask.data.compute()
    elif seg_selector == 'Cell ID':
        seg_mask = this_time.segmentation_mask_cell.data.compute()
    plot = gv.QuadMesh((lons, lats, seg_mask), kdims=['Longitude', 'Latitude'], vdims=['segmentation_mask']).opts(
        cmap='plasma', colorbar=False, tools=['hover'], visible=seg_tick)
    return plot


if __name__ == '__main__':
    tfm = xr.open_dataset('/Volumes/LtgSSD/tobac_saves/tobac_Save_20220602/Track_features_merges_augmented2.zarr', chunks='auto')
    goes_time_range_start = tfm.time.data.astype('datetime64[us]').astype(dt).min()
    goes_time_range_end = tfm.time.data.astype('datetime64[us]').astype(dt).max()

    radar = xr.open_mfdataset(goes_time_range_start.strftime('/Volumes/LtgSSD/nexrad_zarr/%B/%Y%m%d/')+'*.zarr', engine='zarr', chunks='auto')
    radar = radar.isel(nradar=0)
    radar['lat'] = tfm.lat
    radar['lon'] = tfm.lon

    lma = xr.open_dataset('/Volumes/LtgSSD/'+goes_time_range_start.strftime('%B').lower()+goes_time_range_start.strftime('/6sensor_minimum/LYLOUT_%y%m%d_000000_86400_map500m.nc'),
                          chunks='auto')
    print(lma)
    
    
    sat_min_x = tfm.g16_scan_x.min().data.compute()
    sat_max_x = tfm.g16_scan_x.max().data.compute()
    sat_min_y = tfm.g16_scan_y.min().data.compute()
    sat_max_y = tfm.g16_scan_y.max().data.compute()

    g16 = GOES(satellite=16, product='ABI-L2-MCMIPC')
    print('Downloading GOES data')
    download_results = g16.timerange(goes_time_range_start-timedelta(minutes=15), goes_time_range_end+timedelta(minutes=15), max_cpus=12)
    download_results['valid'] = download_results[['start', 'end']].mean(axis=1)
    download_results['path'] = '/Volumes/LtgSSD/' + download_results['file'].values
    unique_times = np.unique(tfm.time.data).astype('datetime64[us]').astype(dt).tolist()

    date_slider = pn.widgets.Select(name='Date', options=unique_times, value=unique_times[0])
    channel_select = pn.widgets.Select(name='Satellite Channel', options=['CMI_C02', 'CMI_C13'], value='CMI_C02')
    satellite_tick = pn.widgets.Checkbox(name='Show Satellite')
    satellite = hv.DynamicMap(pn.bind(plot_satellite, download_results, date_slider,
                                      sat_min_x, sat_max_x, sat_min_y, sat_max_y,
                                      channel_select, satellite_tick))
    
    seg_tick = pn.widgets.Checkbox(name='Show Segmentation')
    seg_sel = pn.widgets.Select(name='Grid Product', options=['Feature ID', 'Cell ID'], value='Cell ID')
    seg = hv.DynamicMap(pn.bind(plot_seg_mask, tfm, date_slider, seg_sel, seg_tick))

    radar_sel = pn.widgets.Select(name='Radar Product', options=['Reflectivity', 'RhoHV', 'ZDR'], value='Reflectivity')
    z_sel = pn.widgets.IntSlider(name='Radar Z Level', start=0, end=15000, step=500, value=0)
    radar_tick = pn.widgets.Checkbox(name='Show Radar')
    radar_mesh = hv.DynamicMap(pn.bind(plot_radar, radar, date_slider, radar_sel, z_sel, radar_tick))

    
    my_map = (gv.tile_sources.OSM *
              radar_mesh.opts(alpha=0.85) *
              satellite.opts(alpha=0.85) *
              seg.opts(alpha=0.5)
              ).opts(width=800, height=800)
    col = pn.Row(pn.Column(date_slider, my_map), pn.Column(seg_sel, seg_tick, channel_select, satellite_tick, radar_sel, z_sel, radar_tick))
    pn.serve(col, port=5006, websocket_origin='100.83.93.83:5006')