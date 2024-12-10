from os import path, listdir
from datetime import datetime as dt, timedelta
import sys

import holoviews as hv
import geoviews as gv
import panel as pn

import geopandas as gpd
from shapely.geometry import Polygon, LineString

import numpy as np
import xarray as xr

from goes2go import GOES

from pyxlma import coords
from glmtools.io.lightning_ellipse import lightning_ellipse_rev

from functools import partial


import sys

lim_mins = (-98.3, 25.5)
lim_maxs = (-91, 32)
xmin, ymin = hv.util.transform.lon_lat_to_easting_northing(*lim_mins)
xmax, ymax = hv.util.transform.lon_lat_to_easting_northing(*lim_maxs)

hv.extension('bokeh')

select_geoms = {}

def plot_divergence(time, divergence_tick):
    bnds = (lim_mins[0], lim_mins[1]+.09, lim_maxs[0], lim_maxs[1])
    blank = gv.RGB((np.array([0, 1]), np.array([0, 1]), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))).opts(visible=False)
    if not divergence_tick:
        return blank
    path_to_image = path.join('/Volumes','LtgSSD','analysis', 'mpl-generated', 'sfcdiv', f'{time.strftime("%Y%m%d_%H%M%S")}.png')
    if not path.exists(path_to_image):
        return blank
    plot = gv.RGB.load_image(path_to_image, bounds=bnds)
    print(f'Divergence plot done for {time}')
    return plot


def plot_stations(time, station_tick):
    bnds = (lim_mins[0], lim_mins[1]+.09, lim_maxs[0], lim_maxs[1])
    blank = gv.RGB((np.array([0, 1]), np.array([0, 1]), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))).opts(visible=False)
    if not station_tick:
        return blank
    path_to_image = path.join('/Volumes','LtgSSD','analysis', 'mpl-generated', 'sfcwinds', f'{time.strftime("%Y%m%d_%H%M%S")}.png')
    if not path.exists(path_to_image):
        return blank
    plot = gv.RGB.load_image(path_to_image, bounds=bnds)
    print(f'Station plot done for {time}')
    return plot

def plot_satellite(time, channel_select, satellite_tick):
    bnds = (lim_mins[0], lim_mins[1]+.09, lim_maxs[0], lim_maxs[1])
    blank = gv.RGB((np.array([0, 1]), np.array([0, 1]), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))).opts(visible=False)
    if not satellite_tick:
        return blank
    path_to_image = path.join('/Volumes','LtgSSD','analysis', 'mpl-generated', 'sat', f'{channel_select}_{time.strftime("%Y%m%d_%H%M%S")}.png')
    if not path.exists(path_to_image):
        return blank
    plot = gv.RGB.load_image(path_to_image, bounds=bnds)
    print(f'Satellite plot done for {time}')
    return plot


def plot_radar(radar, time, radar_selector, radar_tick):
    bnds = (lim_mins[0], lim_mins[1]+.09, lim_maxs[0], lim_maxs[1])
    blank = gv.RGB((np.array([0, 1]), np.array([0, 1]), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))).opts(visible=False)
    if not radar_tick:
        return blank
    path_to_image = path.join('/Volumes','LtgSSD','analysis', 'mpl-generated', radar, f'{radar_selector.replace(" ", "")}_{time.strftime("%Y%m%d_%H%M%S")}.png')
    if not path.exists(path_to_image):
        return blank
    plot = gv.RGB.load_image(path_to_image, bounds=bnds)
    print(f'Radar plot done for {time}')
    return plot

def plot_seg_mask(dataset, time, seg_tick):
    if not seg_tick:
        return gv.QuadMesh(([0, 1], [0, 1], [[0, 1], [2, 3]]), kdims=['Longitude', 'Latitude'], vdims=['segmentation_mask']).opts(visible=False)
    print('Plotting segmentation')
    this_time = dataset.sel(time=time)
    lons = this_time.lon.data
    lats = this_time.lat.data
    seg_mask = this_time.segmentation_mask.data.compute()
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
    date_i_want = dt.strptime(sys.argv[1], '%Y-%m-%d')
    tfma_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges_augmented.zarr'
    if path.exists(tfma_path):
        tfm = xr.open_dataset(tfma_path, engine='zarr', chunks='auto')
    elif path.exists(tfma_path.replace('augmented.zarr', 'coordinated.zarr')):
        tfm = xr.open_dataset(tfma_path.replace('augmented.zarr', 'coordinated.zarr'), engine='zarr', chunks='auto')
    elif path.exists(tfma_path.replace('Track_features_merges_augmented.zarr', 'tfma_checkpoint.zarr')):
        tfm = xr.open_dataset(tfma_path.replace('Track_features_merges_augmented.zarr', 'tfma_checkpoint.zarr'), engine='zarr', chunks='auto')
    else:
        raise FileNotFoundError(f'No file found at {tfma_path}')
    unique_times = np.unique(tfm.time.data).astype('datetime64[us]').astype(dt).tolist()

    grid_max_lon = tfm.lon.max().compute()
    grid_min_lon = tfm.lon.min().compute()
    grid_max_lat = tfm.lat.max().compute()
    grid_min_lat = tfm.lat.min().compute()

    goes_time_range_start = tfm.time.data.astype('datetime64[us]').astype(dt).min()
    goes_time_range_end = tfm.time.data.astype('datetime64[us]').astype(dt).max()

    # lma = xr.open_dataset('/Volumes/LtgSSD/'+date_i_want.strftime('%B').lower()+date_i_want.strftime('/6sensor_minimum/LYLOUT_%y%m%d_000000_86400_map500m.nc'),
    #                       chunks='auto')
    
    
    sat_min_x = tfm.g16_scan_x.min().data.compute()
    sat_max_x = tfm.g16_scan_x.max().data.compute()
    sat_min_y = tfm.g16_scan_y.min().data.compute()
    sat_max_y = tfm.g16_scan_y.max().data.compute()

    date_slider = pn.widgets.Select(name='Date', options=unique_times, value=unique_times[0])
    channel_select = pn.widgets.Select(name='Satellite Channel', options=['CMI_C02', 'CMI_C13'], value='CMI_C02')
    satellite_tick = pn.widgets.Checkbox(name='Show Satellite')
    satellite = hv.DynamicMap(pn.bind(plot_satellite, date_slider, channel_select, satellite_tick))
    
    seg_tick = pn.widgets.Checkbox(name='Show Segmentation')
    seg_mask_func = partial(plot_seg_mask, dataset=tfm.copy())
    seg = hv.DynamicMap(pn.bind(seg_mask_func, time=date_slider, seg_tick=seg_tick))

    radar_sel = pn.widgets.Select(name='Radar Product', options=['Reflectivity', 'RhoHV', 'ZDR', 'Velocity', 'Spectrum Width'], value='Reflectivity')
    hgx_tick = pn.widgets.Checkbox(name='Show KHGX', value=False)
    radar_mesh_hgx = hv.DynamicMap(pn.bind(plot_radar, 'KHGX', date_slider, radar_sel, hgx_tick))
    lch_tick = pn.widgets.Checkbox(name='Show KLCH', value=False)
    radar_mesh_lch = hv.DynamicMap(pn.bind(plot_radar, 'KLCH', date_slider, radar_sel, lch_tick))
    iah_tick = pn.widgets.Checkbox(name='Show TIAH', value=False)
    radar_mesh_iah = hv.DynamicMap(pn.bind(plot_radar, 'TIAH', date_slider, radar_sel, iah_tick))
    hou_tick = pn.widgets.Checkbox(name='Show THOU', value=False)
    radar_mesh_hou = hv.DynamicMap(pn.bind(plot_radar, 'THOU', date_slider, radar_sel, hou_tick))

    stations_tick = pn.widgets.Checkbox(name='Show stations')
    stations = hv.DynamicMap(pn.bind(plot_stations, date_slider, stations_tick))

    div_tick = pn.widgets.Checkbox(name='Show divergence')
    divergence = hv.DynamicMap(pn.bind(plot_divergence, date_slider, div_tick))

    polygon = gv.Polygons([]).opts(fill_alpha=0.3, fill_color='black')
    sfc_lasso = hv.streams.PolyDraw(source=polygon, drag=False, num_objects=1)
    handle_with_time = partial(handle_lasso, time=date_slider)
    sfc_lasso.add_subscriber(handle_with_time)

    lower_limit_line = LineString([(-120, 26.5), (-91.5, 26.5)])
    lower_limit = gv.Shape(lower_limit_line).opts(color='red')
    eastern_limit_line = LineString([(-91.5, 26.5), (-91.5, 70)])
    eastern_limit = gv.Shape(eastern_limit_line).opts(color='red')

    write_btn = pn.widgets.Button(name='Write JSON')
    pn.bind(write_json, write_btn, watch=True)


    my_map = (gv.tile_sources.OSM *
              divergence.opts(alpha=0.85) *
              satellite.opts(alpha=0.85) *
              radar_mesh_hgx.opts(alpha=0.85) *
              radar_mesh_lch.opts(alpha=0.85) *
              radar_mesh_iah.opts(alpha=0.85) *
              radar_mesh_hou.opts(alpha=0.85) *
              seg.opts(alpha=0.5, tools=['hover']) *
              stations *
              polygon * lower_limit * eastern_limit
              ).opts(width=1300, height=825, xlim=(xmin, xmax), ylim=(ymin, ymax))
    control_column = pn.Column(date_slider, seg_tick, channel_select, satellite_tick, radar_sel, hgx_tick, lch_tick, iah_tick, hou_tick, stations_tick, div_tick, write_btn)
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