from os import path
from datetime import datetime as dt, timedelta

import holoviews as hv
import geoviews as gv
import panel as pn

import pandas as pd

import numpy as np
import xarray as xr
from dask import array as da

from goes2go import GOES

from pyxlma import coords
from glmtools.io.lightning_ellipse import lightning_ellipse_rev
import cmweather

from functools import partial

import warnings

import sys

hv.extension('bokeh')

selected_time = None
selected_idx = []

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


def plot_radar(dataset, time, radar_selector, z_selector, radar_tick):
    if not radar_tick:
        return gv.QuadMesh(([0, 1], [0, 1], [[0, 1], [2, 3]]), kdims=['Longitude', 'Latitude'], vdims=[radar_selector]).opts(visible=False)
    print('Plotting radar')
    this_time = dataset.sel(time=time, method='nearest')
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


def plot_sfc_obs(dataset, time, var_to_plot, sfc_tick):
    if not sfc_tick:
        return gv.Points(([0], [0], [0], [0]), kdims=['Longitude', 'Latitude'], vdims=[var_to_plot, 'ID']).opts(visible=False)
    print('Plotting obs')
    cells_of_interest = [2500, 3332, 3747]
    lower_time_bound = time - timedelta(hours=1)
    time = np.array([time]).astype('datetime64[us]')[0]
    lower_time_bound = np.array([lower_time_bound]).astype('datetime64[us]')[0]
    before = (dataset.observationTime.compute() <= time).compute()
    after = (dataset.observationTime.compute() >= lower_time_bound).compute()
    rec_i_want = dataset.where((before & after), drop=True)
    rec_i_want = rec_i_want.where(~np.isnan(rec_i_want[var_to_plot]).compute(), drop=True)
    # Trajectory based filtering
    # rec_mask = da.full_like(rec_i_want.recNum.data, False, dtype=bool)
    # for i, rec_num in enumerate(rec_i_want.recNum.data):
    #     station = rec_i_want.sel(recNum=rec_num)
    #     stat_lon = station.longitude.data
    #     stat_lat = station.latitude.data
    #     stat_speed = station.windSpeed.data
    #     if np.isnan(stat_speed):
    #         rec_mask[i] = False
    #         continue
    #     stat_u = station.u.data
    #     stat_v = station.v.data
    #     grid_idx_x = np.argmin(np.abs(tfm.lon.data - stat_lon))
    #     grid_idx_y = np.argmin(np.abs(tfm.lat.data - stat_lat))
    #     station_x, station_y = tfm_time.x.isel(x=int(grid_idx_x)).data, tfm_time.y.isel(y=int(grid_idx_y)).data
    #     hours_max = 3
    #     distance_max = stat_speed * hours_max * 60 * 60
    #     max_step = int(distance_max / stat_speed)
    #     steps = np.arange(0, max_step)

    #     u_norm = stat_u / stat_speed
    #     v_norm = stat_v / stat_speed
    #     xs = station_x + u_norm * steps
    #     xs = np.clip(xs, tfm.x.min().data, tfm.y.max().data)

    #     ys = station_y + v_norm * steps
    #     ys = np.clip(ys, tfm.y.min().data, tfm.y.max().data)


    #     too_close = (np.sqrt((tfm_time.x - station_x)**2 + (tfm_time.y - station_y)**2) < 3000)
    #     segmentation_along_trajectory = tfm_time.segmentation_mask_cell.where(~too_close, np.nan).sel(x=xs, y=ys, method='nearest')
    #     rec_mask[i] = np.any(np.isin(segmentation_along_trajectory.data, cells_of_interest))
    # rec_i_want = rec_i_want.where(rec_mask.compute(), drop=True)

    if var_to_plot == 'dewpoint':
        plot = gv.Points((rec_i_want.longitude, rec_i_want.latitude, ((9/5)*(rec_i_want.dewpoint-273.15)+32), rec_i_want.stationId),
                kdims=['longitude', 'latitude'], vdims=['dewpoint', 'ID']).opts(size=7, color='dewpoint', cmap='BrBG', clim=(60, 80), tools=['hover'], line_color='black', selection_line_color='red')
    elif var_to_plot == 'temperature':
        plot = gv.Points((rec_i_want.longitude, rec_i_want.latitude, ((9/5)*(rec_i_want.temperature-273.15)+32), rec_i_want.stationId),
                kdims=['longitude', 'latitude'], vdims=['temperature', 'ID']).opts(size=7, color='temperature', cmap='rainbow', clim=(50, 100), tools=['hover'], line_color='black', selection_line_color='red')
    elif var_to_plot == 'u':
        raise NotImplementedError('Cannot plot wind barbs in bokeh')
    if not sfc_tick:
        plot = plot.opts(visible=False)
    print(f'Obs plot done for {time}')
    return plot

def handle_sfc_sel(index, this_time):
    global selected_idx
    selected_idx = index
    global selected_time
    selected_time = this_time


def write_sfc_sel(_):
    if path.exists('20220602-stations.csv'):
        stations = pd.read_csv('20220602-stations.csv')
    else:
        stations = pd.DataFrame({'time': [''], 'index': ['']})
    selected_idx_str = '.'.join([str(i) for i in selected_idx])
    stations = pd.concat((stations, pd.DataFrame({'time': [selected_time.strftime('%Y%m%dT%H:%M:%S')], 'index': selected_idx_str})))
    stations.write_csv('20220602-stations.csv')



def plot_lma(dataset, time, var_to_plot, lma_tick):
    if not lma_tick:
        return gv.Points(([0], [0], [0]), kdims=['Longitude', 'Latitude'], vdims=[var_to_plot]).opts(visible=False)
    lower_time_bound = time - timedelta(minutes=10)
    time = np.array([time]).astype('datetime64[us]')[0]
    lower_time_bound = np.array([lower_time_bound]).astype('datetime64[us]')[0]
    lma_i_want = dataset.where(((dataset.event_time <= time) & (dataset.event_time >= lower_time_bound) & (dataset.event_chi2 <= 1)).compute(), drop=True)
    c_var = lma_i_want[var_to_plot].data.astype('float64')
    cmin = 0
    cmax = 1
    if len(c_var) > 0:
        cmin = c_var.min()
        c_var = (c_var - cmin)/1e9
        cmin = c_var.min()
        cmax = c_var.max()
    plot = gv.Points((lma_i_want.event_longitude.data, lma_i_want.event_latitude.data, c_var), kdims=['Longitude', 'Latitude'], vdims=[var_to_plot]).opts(cmap='magma', color=var_to_plot,
                                                                                                                                                          visible=lma_tick, clim=(cmin, cmax), tools=['hover'])
    print(f'LMA plot done for {time}')
    return plot


def plot_sounding_temperature(dataset, time, augmenter):
    time = np.array([time]).astype('datetime64[s]')[0]
    temps = dataset.skewed_T.data.copy()
    aug_temp = augmenter[augmenter.index == time]['temperature'].values
    if len(aug_temp) == 1:
        temps[0] = aug_temp[0]
    plot = hv.Curve((temps, dataset.pres.data), kdims=['Temperature'], vdims=['Pressure']).opts(ylabel='Pressure (hPa)', xlabel='Temperature (C)').opts(width=400, height=400, color='red')
    plot = plot * hv.Points((temps[0], dataset.pres.data[0])).opts(color='red', size=10, line_color='black')
    return plot


def plot_sounding_dewpoint(dataset, time, augmenter):
    time = np.array([time]).astype('datetime64[s]')[0]
    dews = dataset.skewed_Td.data.copy()
    aug_dew = augmenter[augmenter.index == time]['dewpoint'].values
    if len(aug_dew) == 1:
        dews[0] = aug_dew[0]
    plot = hv.Curve((dews, dataset.pres.data), kdims=['Temperature'], vdims=['Pressure']).opts(width=400, height=400, color='green')
    plot = plot * hv.Points((dews[0], dataset.pres.data[0])).opts(color='green', size=10, line_color='black')
    return plot

def plot_sounding_isotherms(dataset, values, temp_offset):
    plot = None
    for v in values:
        temps = np.full_like(dataset.pres.data, v)
        temps = temps + temp_offset
        if plot is None:
            plot = hv.Curve((temps, dataset.pres.data), kdims=['Temperature'], vdims=['Pressure']).opts(color='gray', alpha=0.5)
        else:
            plot = plot * hv.Curve((temps, dataset.pres.data), kdims=['Temperature'], vdims=['Pressure']).opts(color='gray', alpha=0.5)
    return plot

def avg_t_label(time, augmenter):
    time = np.array([time]).astype('datetime64[s]')[0]
    avg_temp = augmenter[augmenter.index == time]['temperature'].values
    avg_dew = augmenter[augmenter.index == time]['dewpoint'].values
    ml_cape = augmenter[augmenter.index == time]['mlcape'].values
    ml_cinh = augmenter[augmenter.index == time]['mlcinh'].values
    ml_ecape = augmenter[augmenter.index == time]['mlecape'].values
    sb_cape = augmenter[augmenter.index == time]['sbcape'].values
    sb_cinh = augmenter[augmenter.index == time]['sbcinh'].values
    sb_ecape = augmenter[augmenter.index == time]['sbecape'].values
    if len(avg_temp) == 1:
        label_str =  f'### Continental Stations\n ### Avg T: {avg_temp[0]:.2f} C Avg Td: {avg_dew[0]:.2f} C'
        label_str += f'\n ### Surface Based:\n### CAPE: {sb_cape[0]:.2f} J/kg CINH: {sb_cinh[0]:.2f} J/kg ECAPE: {sb_ecape[0]:.2f} J/kg'
        label_str += f'\n### Mixed Layer:\n### CAPE: {ml_cape[0]:.2f} J/kg CINH: {ml_cinh[0]:.2f} J/kg ECAPE: {ml_ecape[0]:.2f} J/kg'
        return label_str
    return '### Avg T: N/A Avg Td: N/A'

def cell_cloud_top_label(dataset, time):
    time_i_want = np.array([time]).astype('datetime64[ns]')[0]
    min_sats = dataset['min_L2-MCMIPC']
    min_sats_filt = min_sats.where(((min_sats.time == time_i_want)), drop=True)
    min_sats_singleval = min_sats_filt.mean().data.compute()
    return f'### Avg Cell Cloud Temperature: {min_sats_singleval:.2f} K'

def cell_cloud_top_plot(dataset, time):
    time_lower_bound = time - timedelta(hours=1)
    time_i_want = np.array([time]).astype('datetime64[ns]')[0]
    time_lower_bound = np.array([time_lower_bound]).astype('datetime64[ns]')[0]
    min_sats = dataset['min_L2-MCMIPC']
    min_sats_filt = min_sats.where(((min_sats.time >= time_lower_bound) & (min_sats.time <= time_i_want)), drop=True)
    feat_2d, time_2d = np.meshgrid(min_sats_filt.feature.data, min_sats_filt.time.data)
    plot = hv.Scatter((time_2d.T.flatten(), min_sats_filt.data.flatten(), feat_2d.T.flatten()),
           kdims=['Time'], vdims=['Channel 13 Brightness Minimum', 'Feature ID']).opts(color='Feature ID', cmap='viridis', ylim=(220, 310),
                                                                                       xlim=(time_lower_bound.astype('datetime64[s]'), time_i_want.astype('datetime64[s]')))
    return plot

def flash_count_label(lma_ds, time):
    lower_time_bound = time - timedelta(minutes=5)
    time = np.array([time]).astype('datetime64[us]')[0]
    lower_time_bound = np.array([lower_time_bound]).astype('datetime64[us]')[0]
    lma_i_want = lma_ds.where(((lma_ds.flash_time_start <= time) & (lma_ds.flash_time_start >= lower_time_bound)).compute(), drop=True)
    flash_count = lma_i_want.number_of_flashes.shape[0]
    return f'### 5-min Flash Count: {flash_count}'


# def flash_count_plot(dataset, time):
#     time_lower_bound = time - timedelta(minutes=10)
#     time_i_want = np.array([time]).astype('datetime64[ns]')[0]
#     time_lower_bound = np.array([time_lower_bound]).astype('datetime64[ns]')[0]
#     ds_filt = dataset.where(((dataset.time >= time_lower_bound) & (dataset.time <= time_i_want)).compute(), drop=True)
#     time_index_start = np.argmin(np.abs(dataset.time.data - time_lower_bound))
#     time_index_end = np.argmin(np.abs(dataset.time.data - time_i_want))
#     ds_filt = ds_filt.where((ds_filt.feature_time_index >= time_index_start).compute(), drop=True)
#     ds_filt = ds_filt.where((ds_filt.feature_time_index <= time_index_end).compute(), drop=True).compute()
#     unique_time_indices = np.unique(ds_filt.feature_time_index.data).astype(int)
#     unique_times = dataset.time.data[unique_time_indices]
#     flash_counts = np.zeros_like(unique_time_indices)
#     for i, time_idx in enumerate(np.unique(ds_filt.feature_time_index.data)):
#         flash_counts[i] = ds_filt.where((ds_filt.feature_time_index == time_idx).compute(), drop=True).feature_flash_count.sum().compute()
#     plot = hv.Scatter((unique_times, flash_counts), kdims=['Time'], vdims=['Flash Count']).opts(size=10, color='Flash Count', cmap='viridis')
#     return plot



if __name__ == '__main__':
    date_i_want = dt(2022, 6, 2, 0, 0, 0)
    tfm = xr.open_dataset(f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges_augmented2.zarr', engine='zarr', chunks='auto')
    sounding = xr.open_dataset('armdata/housondewnpnS3.b1.20220602.173000.cdf').isel(time=slice(-1))
    skew_angle = 30
    P_bottom = np.max(sounding.pres.data)
    temp_offset = 37*np.log10(P_bottom/sounding.pres.data)/np.tan(np.deg2rad(skew_angle))
    sounding['skewed_T'] = sounding.tdry.data + temp_offset
    sounding['skewed_Td'] = sounding.dp.data + temp_offset
    augment_data = pd.read_csv('20220602-augment.csv', parse_dates=['time'], index_col='time')

    grid_max_lon = tfm.lon.max().compute()
    grid_min_lon = tfm.lon.min().compute()
    grid_max_lat = tfm.lat.max().compute()
    grid_min_lat = tfm.lat.min().compute()

    goes_time_range_start = tfm.time.data.astype('datetime64[us]').astype(dt).min()
    goes_time_range_end = tfm.time.data.astype('datetime64[us]').astype(dt).max()

    radar = xr.open_mfdataset(date_i_want.strftime('/Volumes/LtgSSD/nexrad_zarr/%B/%Y%m%d/')+'*.zarr', engine='zarr', chunks='auto')
    radar = radar.isel(nradar=0)
    radar['lat'] = tfm.lat
    radar['lon'] = tfm.lon

    madis_file = path.join(path.sep, 'Volumes', 'LtgSSD', 'sfcdata_madis', date_i_want.strftime('%Y%m%d_*'))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        madis_ds = xr.open_mfdataset(madis_file, engine='netcdf4', chunks='auto', coords='minimal', concat_dim='recNum', combine='nested', compat='override')
    madis_ds = madis_ds.where(((madis_ds.longitude <= grid_max_lon) & (madis_ds.longitude >= grid_min_lon) & (madis_ds.latitude <= grid_max_lat) & (madis_ds.latitude >= grid_min_lat)).compute(), drop=True)
    madis_ds = madis_ds.where(np.logical_and(madis_ds.stationId != b'F5830', madis_ds.stationId != b'6114D').compute(), drop=True)
    dims_to_rm = list(madis_ds.dims)
    dims_to_rm.remove('recNum')
    madis_ds = madis_ds.drop_dims(dims_to_rm)


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

    radar_sel = pn.widgets.Select(name='Radar Product', options=['Reflectivity', 'RhoHV', 'ZDR'], value='Reflectivity')
    z_sel = pn.widgets.IntSlider(name='Radar Z Level', start=0, end=15000, step=500, value=0)
    radar_tick = pn.widgets.Checkbox(name='Show Radar', value=True)
    radar_mesh = hv.DynamicMap(pn.bind(plot_radar, radar, date_slider, radar_sel, z_sel, radar_tick))


    sfc_tick = pn.widgets.Checkbox(name='Show Surface Obs', value=True)
    sfc_sel = pn.widgets.Select(name='Surface Obs Variable', options=['temperature', 'dewpoint'], value='dewpoint')
    sfc = hv.DynamicMap(pn.bind(plot_sfc_obs, madis_ds, date_slider, sfc_sel, sfc_tick)).opts(tools=['lasso_select'])
    sfc_lasso = hv.streams.Selection1D(source=sfc)
    handle_sfc_sel_w_time = pn.bind(handle_sfc_sel, this_time=date_slider)
    sfc_lasso.add_subscriber(handle_sfc_sel_w_time)


    lma_tick = pn.widgets.Checkbox(name='Show LMA', value=True)
    lma_sel = pn.widgets.Select(name='LMA Variable', options=['event_time', 'event_power', 'event_altitude'], value='event_time')
    lma_points = hv.DynamicMap(pn.bind(plot_lma, lma, date_slider, lma_sel, lma_tick))

    sounding_T = hv.DynamicMap(pn.bind(plot_sounding_temperature, sounding, date_slider, augment_data))
    sounding_Td = hv.DynamicMap(pn.bind(plot_sounding_dewpoint, sounding, date_slider, augment_data))
    sounding_isotherms = plot_sounding_isotherms(sounding, np.arange(-110, 41, 10), temp_offset)
    sounding_label = pn.pane.Markdown(pn.bind(avg_t_label, date_slider, augment_data))

    cloud_top_plot = pn.pane.Markdown(pn.bind(cell_cloud_top_label, tfm, date_slider))

    flash_cnt_label = pn.pane.Markdown(pn.bind(flash_count_label, lma, date_slider))

    write_out_button = pn.widgets.Button(name='Write')
    pn.bind(write_sfc_sel, write_out_button, watch=True)

    lim_mins = (-98.6, 28)
    lim_maxs = (-93.2, 32.5)
    xmin, ymin = hv.util.transform.lon_lat_to_easting_northing(*lim_mins)
    xmax, ymax = hv.util.transform.lon_lat_to_easting_northing(*lim_maxs)

    my_map = (gv.tile_sources.OSM *
              satellite.opts(alpha=0.85) *
              radar_mesh.opts(alpha=0.85) *
              seg.opts(alpha=0.5, tools=['hover']) *
              sfc.opts(alpha=0.5, tools=['hover']) *
              lma_points.opts(tools=['hover'])
              ).opts(width=800, height=800, xlim=(xmin, xmax), ylim=(ymin, ymax))
    my_skewT = (sounding_isotherms * sounding_T * sounding_Td).opts(width=400, height=400, logy=True, xlim=(-40, 35), ylim=(1020, 100))
    control_column = pn.Column(date_slider, seg_sel, seg_tick, channel_select, satellite_tick, radar_sel, z_sel, radar_tick, sfc_sel, sfc_tick,
                                                           lma_sel, lma_tick, write_out_button)
    col = pn.Row(pn.Column(my_map), pn.Column(my_skewT, sounding_label, pn.Row(cloud_top_plot, flash_cnt_label)), control_column)
    if '--anim' not in sys.argv:
        pn.serve(col, port=5006, websocket_origin=['localhost:5006', '100.83.93.83:5006'])
    else:
        times_to_plot = augment_data.index
        unique_times_np = np.unique(tfm.time.data).astype('datetime64[us]')
        for i, time in enumerate(times_to_plot):
            time = np.array([time]).astype('datetime64[us]')[0]
            date_slider_index = np.argmin(np.abs(unique_times_np - time))
            date_slider.value = unique_times[date_slider_index]
            print(f'Plotting {date_slider.value}')
            this_time_plot = pn.Row(pn.Column(date_slider, my_map), pn.Column(my_skewT, sounding_label, pn.Row(cloud_top_plot, flash_cnt_label)))
            # save to PNG
            this_time_plot.save(f'20220602/{str(i+1).zfill(4)}.png')