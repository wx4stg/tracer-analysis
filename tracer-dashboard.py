from os import path
from datetime import datetime as dt, timedelta

import holoviews as hv
import geoviews as gv
import panel as pn

import numpy as np
import xarray as xr
from dask import array as da

from goes2go import GOES

from pyxlma import coords
from glmtools.io.lightning_ellipse import lightning_ellipse_rev
import cmweather

import warnings

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


def plot_sfc_obs(dataset, time, var_to_plot, sfc_tick, tfm):
    tfm_time = tfm.sel(time=time, method='nearest')
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
                kdims=['longitude', 'latitude'], vdims=['dewpoint', 'ID']).opts(size=7, color='dewpoint', cmap='BrBG', clim=(60, 80), tools=['hover'])
    elif var_to_plot == 'temperature':
        plot = gv.Points((rec_i_want.longitude, rec_i_want.latitude, ((9/5)*(rec_i_want.temperature-273.15)+32), rec_i_want.stationId),
                kdims=['longitude', 'latitude'], vdims=['temperature', 'ID']).opts(size=7, color='temperature', cmap='rainbow', clim=(50, 100), tools=['hover'])
    elif var_to_plot == 'u':
        raise NotImplementedError('Cannot plot wind barbs in bokeh')
    if not sfc_tick:
        plot = plot.opts(visible=False)
    return plot



def plot_lma(dataset, time, var_to_plot, lma_tick):
    lower_time_bound = time - timedelta(minutes=20)
    time = np.array([time]).astype('datetime64[us]')[0]
    lower_time_bound = np.array([lower_time_bound]).astype('datetime64[us]')[0]
    lma_i_want = dataset.where(((dataset.event_time <= time) & (dataset.event_time >= lower_time_bound) & (dataset.event_chi2 <= 1)).compute(), drop=True)
    plot = gv.Points((lma_i_want.event_longitude.data, lma_i_want.event_latitude.data, lma_i_want[var_to_plot].data), kdims=['Longitude', 'Latitude'], vdims=[var_to_plot]).opts(cmap='magma', visible=lma_tick)
    return plot

if __name__ == '__main__':
    date_i_want = dt(2022, 6, 2, 0, 0, 0)
    tfm = xr.open_dataset(f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges_augmented2.zarr', engine='zarr', chunks='auto')

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
    madis_ds['u'] = -madis_ds.windSpeed * np.sin(np.deg2rad(madis_ds.windDir))
    madis_ds['v'] = -madis_ds.windSpeed * np.cos(np.deg2rad(madis_ds.windDir))


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
    seg = hv.DynamicMap(pn.bind(plot_seg_mask, tfm, date_slider, seg_sel, seg_tick))

    radar_sel = pn.widgets.Select(name='Radar Product', options=['Reflectivity', 'RhoHV', 'ZDR'], value='Reflectivity')
    z_sel = pn.widgets.IntSlider(name='Radar Z Level', start=0, end=15000, step=500, value=0)
    radar_tick = pn.widgets.Checkbox(name='Show Radar')
    radar_mesh = hv.DynamicMap(pn.bind(plot_radar, radar, date_slider, radar_sel, z_sel, radar_tick))


    sfc_tick = pn.widgets.Checkbox(name='Show Surface Obs', value=True)
    sfc_sel = pn.widgets.Select(name='Surface Obs Variable', options=['temperature', 'dewpoint'], value='dewpoint')
    sfc = hv.DynamicMap(pn.bind(plot_sfc_obs, madis_ds, date_slider, sfc_sel, sfc_tick, tfm))


    lma_tick = pn.widgets.Checkbox(name='Show LMA')
    lma_sel = pn.widgets.Select(name='LMA Variable', options=['event_time', 'event_power', 'event_altitude'], value='event_time')
    lma = hv.DynamicMap(pn.bind(plot_lma, lma, date_slider, lma_sel, lma_tick))

    my_map = (gv.tile_sources.OSM *
              satellite.opts(alpha=0.85) *
              radar_mesh.opts(alpha=0.85) *
              seg.opts(alpha=0.5) *
              sfc.opts(alpha=0.5) *
              lma
              ).opts(width=800, height=800)
    col = pn.Row(pn.Column(date_slider, my_map), pn.Column(seg_sel, seg_tick, channel_select, satellite_tick, radar_sel, z_sel, radar_tick, sfc_sel, sfc_tick,
                                                           lma_sel, lma_tick))
    pn.serve(col, port=5006, websocket_origin=['localhost:5006', '100.83.93.83:5006'])