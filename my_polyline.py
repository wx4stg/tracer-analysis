from os import path, listdir
from datetime import datetime as dt, timedelta

import holoviews as hv
import geoviews as gv
import panel as pn

import geopandas as gpd
from shapely.geometry import Polygon

import numpy as np
import xarray as xr
import pyart

from goes2go import GOES

from pyxlma import coords
from glmtools.io.lightning_ellipse import lightning_ellipse_rev
import cmweather

from functools import partial


import sys

hv.extension('bokeh')

select_geoms = {}

def plot_satellite(dl_res, time, min_x, max_x, min_y, max_y, channel_select, satellite_tick):
    if not satellite_tick:
        return gv.QuadMesh(([0, 1], [0, 1], [[0, 1], [2, 3]]), kdims=['Longitude', 'Latitude'], vdims=[channel_select]).opts(visible=False)
    print('Plotting satellite')
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
    print(f'Satellite plot done for {time}')
    return plot


def plot_radar(dataset, time, radar_selector, radar_tick):
    if not radar_tick:
        return gv.QuadMesh(([0, 1], [0, 1], [[0, 1], [2, 3]]), kdims=['Longitude', 'Latitude'], vdims=[radar_selector]).opts(visible=False)
    print('Plotting radar')
    ds_times = [dt.strptime(path.basename(f)[4:-4], '%Y%m%d_%H%M%S') for f in dataset]
    ds_timedeltas = [abs(time - t) for t in ds_times]
    if min(ds_timedeltas) > timedelta(minutes=15):
        return gv.QuadMesh(([0, 1], [0, 1], [[0, 1], [2, 3]]), kdims=['Longitude', 'Latitude'], vdims=[radar_selector]).opts(visible=False)
    rdr = pyart.io.read(dataset[np.argmin(ds_timedeltas)])
    nyq = rdr.instrument_parameters['nyquist_velocity']['data'].max()
    if nyq == 0:
        maxr = rdr.instrument_parameters['unambiguous_range']['data'].min()
        prf = 2.998e8/(2*maxr)
        nyq = .05*prf/4
    if radar_selector == 'Reflectivity':
        data2plot = 'reflectivity'
        cmap_to_use = 'ChaseSpectral'
        sweep_i_want = 0
        clim_to_use = (-10, 80)
    elif radar_selector == 'RhoHV':
        data2plot = 'cross_correlation_ratio'
        cmap_to_use = 'plasmidis'
        sweep_i_want = 0
        clim_to_use = (0, 1)
    elif radar_selector == 'ZDR':
        data2plot = 'differential_reflectivity'
        cmap_to_use = 'turbone'
        sweep_i_want = 0
        clim_to_use = (-2, 8)
    elif radar_selector == 'Velocity':
        data2plot = 'velocity'
        cmap_to_use = 'balance'
        sweep_i_want = 1
        clim_to_use = (-nyq/2, nyq/2)
    elif radar_selector == 'Spectrum Width':
        data2plot = 'spectrum_width'
        cmap_to_use = 'cubehelix_r'
        sweep_i_want = 1
        clim_to_use = (0, nyq/2)
    rdr = rdr.extract_sweeps([sweep_i_want])
    lats, lons, _ = rdr.get_gate_lat_lon_alt(0)
    data2plot = rdr.fields[data2plot]['data']
    plot = gv.QuadMesh((lons, lats, data2plot), kdims=['Longitude', 'Latitude'], vdims=[radar_selector]).opts(
        cmap=cmap_to_use, colorbar=False, tools=['hover'], visible=radar_tick)
    if clim_to_use is not None:
        plot = plot.opts(clim=clim_to_use)
    print(f'Radar plot done for {time}')
    return plot

def plot_seg_mask(dataset, time, seg_selector, seg_tick):
    if not seg_tick:
        return gv.QuadMesh(([0, 1], [0, 1], [[0, 1], [2, 3]]), kdims=['Longitude', 'Latitude'], vdims=['segmentation_mask']).opts(visible=False)
    print('Plotting segmentation')
    this_time = dataset.sel(time=time)
    lons = this_time.lon.data
    lats = this_time.lat.data
    if seg_selector == 'Feature ID':
        seg_mask = this_time.segmentation_mask.data.compute()
    elif seg_selector == 'Cell ID':
        seg_mask = this_time.segmentation_mask_cell.data.compute()
    plot = gv.QuadMesh((lons, lats, seg_mask), kdims=['Longitude', 'Latitude'], vdims=['segmentation_mask']).opts(
        cmap='plasma', colorbar=False, tools=['hover'], visible=seg_tick)
    print(f'Segmentation plot done for {time}')
    return plot


def handle_lasso(data, time):
    global select_geoms
    x = data['xs']
    if len(x) == 0:
        return
    x = x[0]
    if len(x) == 0:
        return
    y = data['ys'][0]
    select_geoms[time.value.strftime('%Y-%m-%dT%H:%M:%S')] = (x, y)

def write_json(_):
    global select_geoms
    this_date = list(select_geoms.keys())[0].split('T')[0]
    this_file = f'sam_polyline/{this_date}.json'

    if path.exists(this_file):
        gdf = gpd.read_file(this_file)
        gdf = gdf.set_index('index')
    else:
        gdf = gpd.GeoDataFrame({})
    for time, coords in select_geoms.items():
        lon = coords[0]
        lat = coords[1]
        coords_ready = list(zip(lon, lat))
        gdf.loc[time, 'geometry'] = Polygon(coords_ready)
    gdf = gdf.set_crs(epsg=4326)
    gdf.to_file(this_file, driver='GeoJSON')
    select_geoms = {}

if __name__ == '__main__':
    date_i_want = dt(2022, 6, 2, 0, 0, 0)
    tfm = xr.open_dataset(f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges_augmented2.zarr', engine='zarr', chunks='auto')

    grid_max_lon = tfm.lon.max().compute()
    grid_min_lon = tfm.lon.min().compute()
    grid_max_lat = tfm.lat.max().compute()
    grid_min_lat = tfm.lat.min().compute()

    goes_time_range_start = tfm.time.data.astype('datetime64[us]').astype(dt).min()
    goes_time_range_end = tfm.time.data.astype('datetime64[us]').astype(dt).max()

    radar_files = [f for f in sorted(listdir(f'/Volumes/LtgSSD/nexrad_l2/{date_i_want.strftime("%Y%m%d")}/')) if f.endswith('V06') or f.endswith('V08')]
    radar_files = [f'/Volumes/LtgSSD/nexrad_l2/{date_i_want.strftime("%Y%m%d")}/' + f for f in radar_files]
    khgx_files = [f for f in radar_files if 'KHGX' in f]
    klch_files = [f for f in radar_files if 'KLCH' in f]
    tiah_files = [f for f in radar_files if 'TIAH' in f]
    thou_files = [f for f in radar_files if 'THOU' in f]

    lma = xr.open_dataset('/Volumes/LtgSSD/'+date_i_want.strftime('%B').lower()+date_i_want.strftime('/6sensor_minimum/LYLOUT_%y%m%d_000000_86400_map500m.nc'),
                          chunks='auto')
    
    
    sat_min_x = tfm.g16_scan_x.min().data.compute()
    sat_max_x = tfm.g16_scan_x.max().data.compute()
    sat_min_y = tfm.g16_scan_y.min().data.compute()
    sat_max_y = tfm.g16_scan_y.max().data.compute()

    g16 = GOES(satellite=16, product='ABI-L2-MCMIPC')
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
    seg_mask_func = partial(plot_seg_mask, dataset=tfm.copy())
    seg = hv.DynamicMap(pn.bind(seg_mask_func, time=date_slider, seg_selector=seg_sel, seg_tick=seg_tick))

    radar_sel = pn.widgets.Select(name='Radar Product', options=['Reflectivity', 'RhoHV', 'ZDR', 'Velocity', 'Spectrum Width'], value='Reflectivity')
    hgx_tick = pn.widgets.Checkbox(name='Show KHGX', value=False)
    radar_mesh_hgx = hv.DynamicMap(pn.bind(plot_radar, khgx_files, date_slider, radar_sel, hgx_tick))
    lch_tick = pn.widgets.Checkbox(name='Show KLCH', value=False)
    radar_mesh_lch = hv.DynamicMap(pn.bind(plot_radar, klch_files, date_slider, radar_sel, lch_tick))
    iah_tick = pn.widgets.Checkbox(name='Show TIAH', value=False)
    radar_mesh_iah = hv.DynamicMap(pn.bind(plot_radar, tiah_files, date_slider, radar_sel, iah_tick))
    hou_tick = pn.widgets.Checkbox(name='Show THOU', value=False)
    radar_mesh_hou = hv.DynamicMap(pn.bind(plot_radar, thou_files, date_slider, radar_sel, hou_tick))

    polygon = gv.Polygons([[(-120, 26.5), (-50, 26.5), (-50, 20), (-120, 20)], [(-91.5, 26.5), (-50, 26.5), (-50, 70), (-91.5, 70)]]).opts(fill_alpha=0.3, fill_color='black')
    sfc_lasso = hv.streams.PolyDraw(source=polygon, drag=False, num_objects=1)
    handle_with_time = partial(handle_lasso, time=date_slider)
    sfc_lasso.add_subscriber(handle_with_time)

    # lower_limit = gv.HLine(26.5).opts(color='red')
    # eastern_limit = gv.VLine(-91.5).opts(color='red')

    write_btn = pn.widgets.Button(name='Write JSON')
    pn.bind(write_json, write_btn, watch=True)


    lim_mins = (-98.3, 25.5)
    lim_maxs = (-91, 32)
    xmin, ymin = hv.util.transform.lon_lat_to_easting_northing(*lim_mins)
    xmax, ymax = hv.util.transform.lon_lat_to_easting_northing(*lim_maxs)

    my_map = (gv.tile_sources.OSM *
              satellite.opts(alpha=0.85) *
              radar_mesh_hgx.opts(alpha=0.85) *
              radar_mesh_lch.opts(alpha=0.85) *
              radar_mesh_iah.opts(alpha=0.85) *
              radar_mesh_hou.opts(alpha=0.85) *
              seg.opts(alpha=0.5, tools=['hover']) * polygon #* lower_limit * eastern_limit
              ).opts(width=2200, height=1200, xlim=(xmin, xmax), ylim=(ymin, ymax))
    control_column = pn.Column(date_slider, seg_sel, seg_tick, channel_select, satellite_tick, radar_sel, hgx_tick, lch_tick, iah_tick, hou_tick, write_btn)
    col = pn.Row(pn.Column(my_map), control_column)


    if '--anim' not in sys.argv:
        print('Serving')
        pn.serve(col, port=5006, websocket_origin=['localhost:5006', '100.83.93.83:5006'])
    else:
        times_to_plot = tfm.time.where(((tfm.time > np.datetime64('2022-06-16T18:00:00')) & (tfm.time < np.datetime64('2022-06-16T19:30:00'))), drop=True).data
        unique_times_np = np.unique(tfm.time.data).astype('datetime64[us]')
        for i, time in enumerate(times_to_plot):
            save_path = f'20220616/{str(i+1).zfill(4)}.png'
            if path.exists(save_path):
                continue
            time = np.array([time]).astype('datetime64[us]')[0]
            date_slider_index = np.argmin(np.abs(unique_times_np - time))
            date_slider.value = unique_times[date_slider_index]
            print(f'Plotting {date_slider.value}')
            this_time_plot = pn.Column(date_slider, my_map, width=900, height=1000)
            # save to PNG
            this_time_plot.save(save_path)