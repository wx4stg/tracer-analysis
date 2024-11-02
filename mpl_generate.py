from os import path, listdir, getcwd
from datetime import datetime as dt, timedelta
from pathlib import Path


from goes2go import GOES
from pyxlma import coords
from glmtools.io.lightning_ellipse import lightning_ellipse_rev


from dask.distributed import Client, wait, print
import numpy as np
import xarray as xr
import pyart
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from cartopy import crs as ccrs
from cartopy import feature as cfeat
import cmweather
from metpy.plots import USCOUNTIES


def plot_satellite(path_to_read, this_time, min_x, max_x, min_y, max_y, channel_select):
    print('Plotting satellite')
    mpl_use('agg')
    sat = xr.open_dataset(path_to_read)
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
        cmap_to_use = 'Greys_r'
    else:
        cmap_to_use = 'viridis_r'
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.epsg(3857))
    pcm = ax.pcolormesh(sat_lon, sat_lat, area_i_want[channel_select].data, cmap=cmap_to_use, transform=ccrs.PlateCarree())
    ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeat.COASTLINE.with_scale('10m'), edgecolor='gray', linewidth=0.5)
    lim_mins = (-98.3, 25.5)
    lim_maxs = (-91, 32)
    ax.set_extent([lim_mins[0], lim_maxs[0], lim_mins[1], lim_maxs[1]], crs=ccrs.PlateCarree())
    px = 1/plt.rcParams['figure.dpi']
    fig.set_size_inches(2048*px, 2048*px)
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    path_to_save = path.join('/Volumes','LtgSSD','analysis', 'mpl-generated', 'sat', f'{channel_select.replace(" ", "")}_{this_time.strftime("%Y%m%d_%H%M%S")}.png')
    Path(path.dirname(path_to_save)).mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_save, bbox_inches=extent, transparent=True)
    print(path_to_save)
    plt.close(fig)


def queue_satellite(time, dataset_time, min_x, max_x, min_y, max_y, client=None):
    res = []
    g16 = GOES(satellite=16, product='ABI-L2-MCMIPC')
    dl_res = g16.timerange(dataset_time, dataset_time + timedelta(days=1), max_cpus=12)
    dl_res['valid'] = dl_res[['start', 'end']].mean(axis=1)
    dl_res['path'] = '/Volumes/LtgSSD/' + dl_res['file'].values
    for this_time in time:
        closest_time = dl_res.iloc[(dl_res['valid'] - this_time).abs().argsort()[:1]]
        this_path = closest_time['path'].values[0]
        for channel_to_plot in ['CMI_C02', 'CMI_C13']:
            this_time = np.array([this_time]).astype('datetime64[s]').astype(dt)[0]
            if client is None:
                plot_satellite(this_path, this_time, min_x, max_x, min_y, max_y, channel_to_plot)
            else:
                t = client.submit(plot_satellite, this_path, this_time, min_x, max_x, min_y, max_y, channel_to_plot, pure=False)
                res.append(t)
    if client is not None:
        return res


def plot_radar(time, dataset_time, radar, var_to_plot):
    _ = cmweather.__version__
    mpl_use('agg')
    path_to_radar = path.join('/Volumes', 'LtgSSD', 'nexrad_l2', dataset_time.strftime('%Y%m%d'))
    radar_files = [path.join(path_to_radar, f) for f in sorted(listdir(path_to_radar)) if f.startswith(radar)]
    radar_times = [dt.strptime(path.basename(f)[4:-4], '%Y%m%d_%H%M%S') for f in radar_files]
    ds_timedeltas = [abs(time - t) for t in radar_times]
    if min(ds_timedeltas) > timedelta(minutes=15):
        print('fail')
        return 0
    rdr = pyart.io.read(radar_files[np.argmin(ds_timedeltas)])
    nyq = rdr.instrument_parameters['nyquist_velocity']['data'].max()
    if nyq == 0:
        maxr = rdr.instrument_parameters['unambiguous_range']['data'].min()
        prf = 2.998e8/(2*maxr)
        nyq = .05*prf/4
    if var_to_plot == 'Reflectivity':
        data2plot = 'reflectivity'
        cmap_to_use = 'ChaseSpectral'
        sweep_i_want = 0
        vmin = -10
        vmax = 80
    elif var_to_plot == 'RhoHV':
        data2plot = 'cross_correlation_ratio'
        cmap_to_use = 'plasmidis'
        sweep_i_want = 0
        vmin = 0
        vmax = 1
    elif var_to_plot == 'ZDR':
        data2plot = 'differential_reflectivity'
        cmap_to_use = 'turbone'
        sweep_i_want = 0
        vmin = -2
        vmax = 8
    elif var_to_plot == 'Velocity':
        data2plot = 'velocity'
        cmap_to_use = 'balance'
        sweep_i_want = 1
        vmin = -nyq/2
        vmax = nyq/2
    elif var_to_plot == 'Spectrum Width':
        data2plot = 'spectrum_width'
        cmap_to_use = 'cubehelix_r'
        sweep_i_want = 1
        vmin = 0
        vmax = nyq/2
    rdr = rdr.extract_sweeps([sweep_i_want])
    lats, lons, _ = rdr.get_gate_lat_lon_alt(0)
    data2plot = rdr.fields[data2plot]['data']
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.epsg(3857))
    pcm = ax.pcolormesh(lons, lats, data2plot, cmap=cmap_to_use, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeat.COASTLINE.with_scale('10m'), edgecolor='gray', linewidth=0.5)
    lim_mins = (-98.3, 25.5)
    lim_maxs = (-91, 32)
    ax.set_extent([lim_mins[0], lim_maxs[0], lim_mins[1], lim_maxs[1]], crs=ccrs.PlateCarree())
    px = 1/plt.rcParams['figure.dpi']
    fig.set_size_inches(2048*px, 2048*px)
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    path_to_save = path.join('/Volumes','LtgSSD','analysis', 'mpl-generated', radar, f'{var_to_plot.replace(" ", "")}_{time.strftime("%Y%m%d_%H%M%S")}.png')
    Path(path.dirname(path_to_save)).mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_save, bbox_inches=extent, transparent=True)
    print(path_to_save)
    plt.close(fig)
    return 1

def queue_radar(times, date_i_want, client=None):
    times = times.astype('datetime64[s]').astype(dt)
    radar = ['KHGX', 'TIAH', 'THOU', 'KLCH']
    all_vars = ['Reflectivity', 'Velocity', 'Spectrum Width']
    dualpol_vars = ['RhoHV', 'ZDR']
    res = []
    for time in times:
        for r in radar:
            for var in all_vars:
                if client is None:
                    plot_radar(time, date_i_want, r, var)
                else:
                    t = client.submit(plot_radar, time, date_i_want, r, var, pure=False)
                    res.append(t)
            if 'K' in r:
                for var in dualpol_vars:
                    if client is None:
                        plot_radar(time, date_i_want, r, var)
                    else:
                        t = client.submit(plot_radar, time, date_i_want, r, var, pure=False)
                        res.append(t)
    if client is not None:
        return res



if __name__ == '__main__':
    date_i_want = dt(2022, 6, 2, 0, 0, 0)
    client = Client('tcp://127.0.0.1:8786')
    tfm = xr.open_dataset(f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/Track_features_merges_augmented2.zarr',
                            engine='zarr', chunks='auto')
    times = tfm.time.data
    all_res = []

    radar_res = queue_radar(times, date_i_want, client)
    all_res.extend(radar_res)
    sat_min_x = tfm.g16_scan_x.min().data.compute()
    sat_max_x = tfm.g16_scan_x.max().data.compute()
    sat_min_y = tfm.g16_scan_y.min().data.compute()
    sat_max_y = tfm.g16_scan_y.max().data.compute()
    sat_res = queue_satellite(times, date_i_want, sat_min_x, sat_max_x, sat_min_y, sat_max_y, client)

    all_res.extend(sat_res)
    print('GATHERING!')
    client.gather(all_res)
