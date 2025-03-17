#!/usr/bin/env python


import xarray as xr
import pandas as pd
from act.io import read_icartt
from datetime import datetime as dt
from os import path, listdir
from pathlib import Path as pth
from shutil import rmtree
from glob import glob
import numpy as np
from metpy.interpolate import interpolate_1d
from metpy.units import units
from metpy import calc as mpcalc
from metpy.plots import USCOUNTIES, SkewT
from ecape.calc import calc_ecape
import sounderpy as spy
from scipy.interpolate import interp1d
from scipy.io import loadmat
import sys
import warnings
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib import use as mpluse
from cartopy import crs as ccrs
from cartopy import feature as cfeat
from matplotlib.path import Path
from numba import njit, prange

import pyart
from cmweather.cm_colorblind import ChaseSpectral, plasmidis, turbone
from cmocean.cm import phase

from pyxlma import coords

from track_features_merges_augmented import add_eet_to_radar_data, add_eet_to_tobac_data, add_goes_data_to_tobac_path

USE_DASK = True
if USE_DASK:
    from dask.distributed import Client


@njit(parallel=True)
def identify_side(dts, lons, lats, tfm_times, seabreeze, grid_lon, grid_lat):
    seabreezes = np.zeros(lons.shape)
    for i in prange(seabreezes.shape[0]):
        lon = lons[i]
        lat = lats[i]
        this_dt = dts[i]
        closest_time_idx = np.argmin(np.abs(tfm_times - this_dt))
        dist_idx_raveled = np.argmin(((grid_lon - lon)**2 + (grid_lat - lat)**2)**0.5)
        # dist_idx = np.unravel_index(distance.compute(), distance.shape)
        # Manually implement unravel_index since it isn't supported by numba
        closest_row_idx = dist_idx_raveled // grid_lon.shape[1]
        closest_col_idx = dist_idx_raveled % grid_lon.shape[1]
        closest_seabreeze = seabreeze[closest_time_idx, closest_row_idx, closest_col_idx]
        seabreezes[i] = closest_seabreeze
    return seabreezes


def interp_sounding_times(tfm_time, prev_idx, new_idx, data):
    last_profile = data[prev_idx, :, :].copy()
    new_profile = data[new_idx, :, :].copy()
    last_time = tfm_time[prev_idx].copy().astype(float)
    new_time = tfm_time[new_idx].copy().astype(float)
    times_between = tfm_time[prev_idx+1:new_idx].copy().astype(float)
    x_arr = np.array([last_time, new_time])
    y_arr = np.array([last_profile, new_profile])
    interper = interp1d(
        x_arr,
        y_arr,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
        axis=0
    )
    return interper(times_between)


def add_seabreeze_to_features(tfm, client=None, should_debug=False):
    def make_sbf_plot(time_idx):
        tfm_time = tfm.isel(time=time_idx)
        time = tfm_time.time.data.astype('datetime64[s]').astype('O').item()
        save_path = f'./debug-figs-{time.strftime("%Y%m%d")}/seabreeze/{time_idx}.png'
        if not path.exists(save_path):
            tfm_feat_mask = (tfm_time.feature_time_index == time_idx)
            tfm_feat_time = tfm_time.isel(feature=tfm_feat_mask)
            mpluse('Agg')
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.pcolormesh(tfm_feat_time.lon, tfm_feat_time.lat, tfm_feat_time.seabreeze.transpose(*tfm_feat_time.lon.dims), transform=ccrs.PlateCarree(), zorder=1, alpha=0.25, cmap='RdBu')
            ax.pcolormesh(tfm_feat_time.lon, tfm_feat_time.lat, tfm_feat_time.segmentation_mask, transform=ccrs.PlateCarree(), zorder=2, cmap='Greys', vmin=0, vmax=1, alpha=0.5)
            ax.scatter(tfm_feat_time.feature_lon, tfm_feat_time.feature_lat, c=tfm_feat_time.feature_seabreeze, transform=ccrs.PlateCarree(), zorder=3, s=2, cmap='RdBu')
            ax.set_title(f'Features + Area + Seabreeze\n{time.strftime("%Y-%m-%d %H:%M:%S")}')
            ax.add_feature(cfeat.STATES.with_scale('50m'))
            ax.add_feature(USCOUNTIES.with_scale('5m'))
            fig.savefig(save_path)
            plt.close(fig)
    feature_seabreezes = identify_side(tfm.feature_time.values.astype('datetime64[s]').astype(float), tfm.feature_lon.values, tfm.feature_lat.values, tfm.time.values.astype('datetime64[s]').astype(float), 
                                    tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.values, tfm.lat.values)
    tfm = tfm.assign({
        'feature_seabreeze' : (('feature',), feature_seabreezes)
    })
    if should_debug:
        if client is not None:
            res = client.map(make_sbf_plot, np.arange(tfm.time.shape[0]))
            res = client.gather(res)
        else:
            for i in np.arange(tfm.time.shape[0]):
                make_sbf_plot(i)
    return tfm


def identify_aircraft_below_feature(air_ds, tfm):
    in_feature_times = identify_side(air_ds.time.data.astype('datetime64[s]').astype(float), air_ds.lon.data, air_ds.lat.data, tfm.time.data.astype('datetime64[s]').astype(float),
              tfm.segmentation_mask.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.data, tfm.lat.data)
    air_ds['inside_feature'] = xr.DataArray(in_feature_times, dims='time')
    closest_to_tfm_time_indices = np.argmin(np.abs(air_ds.time.data[:, None] - tfm.time.data), axis=1)
    air_ds['closest_tfm_time'] = xr.DataArray(tfm.time.data[closest_to_tfm_time_indices], dims='time')
    features_to_select = np.unique(in_feature_times)
    features_to_select = features_to_select[~np.isnan(features_to_select)]
    cloud_passes = tfm.sel(feature=features_to_select)
    ccl_hpa = cloud_passes.feature_ccl
    indices_of_profile_ccl = xr.ufuncs.absolute(cloud_passes.feature_pressure_profile - ccl_hpa).argmin(dim='vertical_levels')
    ccl_z = cloud_passes.feature_msl_profile.isel(vertical_levels=indices_of_profile_ccl)
    below_feature = np.zeros(air_ds.inside_feature.data.shape, dtype=bool)
    for cap_to_check in ccl_z:
        spec_idx_inside_feat = np.where(air_ds.inside_feature.data == cap_to_check.feature.data)[0]
        below_feature_mask = air_ds.alt[spec_idx_inside_feat] < cap_to_check.data
        below_feature[spec_idx_inside_feat] = below_feature_mask
    air_ds['below_feature'] = xr.DataArray(below_feature, dims='time')
    return air_ds


def below_cloud_processing(tfm, date_i_want):
    spec_flight_dirs = sorted(glob(f'/Volumes/LtgSSD/air_SPEC_state/RF*_ict_{date_i_want.strftime("%Y%m%d")}*'))
    spec_ds = []
    for spec_flight_dir in spec_flight_dirs:
        page0_data_path = path.join(spec_flight_dir, f'ESCAPE-Page0_Learjet_{date_i_want.strftime("%Y%m%d")}_R0.ict')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = read_icartt(page0_data_path)
        
        rf_num = path.basename(spec_flight_dir).split('_')[0]
        spec_hires_position_dir = f'/Volumes/LtgSSD/AircraftTracks/Learjet/L_{rf_num}/'
        spec_hires_position_file = glob(spec_hires_position_dir+'*.txt')[0]
        nonpublic_df = pd.read_csv(spec_hires_position_file, sep='\\s+')
        nonpublic_df['pydatetime'] = pd.to_datetime(nonpublic_df['yyyy'].astype(str).str.zfill(4) +
                                                nonpublic_df['month'].astype(str).str.zfill(2) +
                                                nonpublic_df['day'].astype(str).str.zfill(2) +
                                                nonpublic_df['hh'].astype(str).str.zfill(2) +
                                                nonpublic_df['mm'].astype(str).str.zfill(2) +
                                                nonpublic_df['ss'].astype(str).str.zfill(2), format='%Y%m%d%H%M%S')
        trimmed_npdf = nonpublic_df[(nonpublic_df['pydatetime'] >= ds.time.data[0]) & (nonpublic_df['pydatetime'] <= ds.time.data[-1])]
        if np.all(trimmed_npdf['pydatetime'].values == ds.time.data):
            ds['lat'] = xr.DataArray(trimmed_npdf['Latitude(deg)'].values, dims='time')
            ds['lon'] = xr.DataArray(trimmed_npdf['Longitude(deg)'].values, dims='time')

        spec_ds.append(ds)
    if len(spec_ds) > 0:
        if len(spec_ds) > 1:
            spec_ds = xr.concat(spec_ds, dim='time')
        elif len(spec_ds) == 1:
            spec_ds = spec_ds[0]
        spec_alt_ft = spec_ds.Palt * units.feet
        spec_alt_m = spec_alt_ft.data.to('m').magnitude
        spec_ds['alt'] = xr.DataArray(spec_alt_m, dims='time')

        spec_ds = identify_aircraft_below_feature(spec_ds, tfm)
        spec_below_cloud = spec_ds.isel(time=spec_ds['below_feature'].data)
        spec_below_cloud.to_netcdf(f'/Volumes/LtgSSD/analysis/below_cloud/{date_i_want.strftime("%Y%m%d")}_spec_below_cloud.nc')



    nrc_flights = sorted(glob(f'/Volumes/LtgSSD/air_NRC_state/atmospheric-state_{date_i_want.strftime("%Y%m%d")}*.nc'))
    if len(nrc_flights) > 0:
        nrc_ds = xr.open_mfdataset(nrc_flights).load().isel(sps1=0)
        nrc_ds = identify_aircraft_below_feature(nrc_ds, tfm)
        nrc_below_cloud = nrc_ds.isel(time=nrc_ds['below_feature'].data)
        nrc_below_cloud.to_netcdf(f'/Volumes/LtgSSD/analysis/below_cloud/{date_i_want.strftime("%Y%m%d")}_nrc_below_cloud.nc')


def plot_radiosonde_data(this_sonde_data, this_pydt, this_lon, this_lat, sbf, tfm, save_path):
    if not path.exists(save_path):
        mpluse('Agg')
        p = this_sonde_data['pres'].values
        T = this_sonde_data['tdry'].values
        Td = this_sonde_data['dp'].values
        u = this_sonde_data['u_wind'].values
        v = this_sonde_data['v_wind'].values
        mask = mpcalc.resample_nn_1d(p,  np.logspace(4, 2))
        fig = plt.figure()
        if this_lon is None:
            skew = SkewT(fig, subplot=(1, 1, 1))
        else:
            skew = SkewT(fig, subplot=(1, 2, 1))
            ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        skew.plot(p, T, 'r')
        skew.plot(p, Td, 'lime')
        skew.ax.scatter([T[0]], [p[0]], color='r', s=10, edgecolors='k', zorder=5)
        skew.ax.scatter([Td[0]], [p[0]], color='lime', s=10, edgecolors='k', zorder=5)
        skew.plot_barbs(p[mask], u[mask], v[mask])

        skew.ax.set_title(f'{this_pydt.strftime("%Y-%m-%d %H:%M")}')
        if this_lon is not None:
            if sbf == -1:
                ax.scatter(this_lon, this_lat, c='blue', s=50, marker='*', edgecolors='k', zorder=5)
            elif sbf == -2:
                ax.scatter(this_lon, this_lat, c='red', s=50, marker='*', edgecolors='k', zorder=5)
            nearest_sbf_time_idx = np.argmin(np.abs(tfm.time.data.astype('datetime64[s]').astype('O') - this_pydt))
            this_sbf = tfm.seabreeze.isel(time=nearest_sbf_time_idx)
            ax.pcolormesh(tfm.lon, tfm.lat, this_sbf.transpose(*tfm.lon.dims), transform=ccrs.PlateCarree(), zorder=1, alpha=0.25, cmap='RdBu')
            ax.add_feature(cfeat.STATES.with_scale('50m'))
            ax.add_feature(USCOUNTIES.with_scale('5m'))
        fig.tight_layout()
        fig.savefig(save_path)


def add_radiosonde_data(tfm, n_sounding_levels=2000, should_debug=False):
    date_i_want = tfm.time.data[0].astype('datetime64[s]').astype(dt).replace(hour=0, minute=0, second=0, microsecond=0)
    time_start_this_day = np.min(tfm.time.data)
    time_end_this_day = np.max(tfm.time.data)

    # Load the ARM DOE sondes
    arm_sonde_path = '/Volumes/LtgSSD/arm-sondes/'
    arm_sonde_files = sorted(listdir(arm_sonde_path))
    arm_sonde_dts = np.array([dt.strptime(' '.join(f.split('.')[2:4]), '%Y%m%d %H%M%S') for f in arm_sonde_files]).astype('datetime64[s]')
    arm_sonde_files = np.array([path.join(arm_sonde_path, f) for f in arm_sonde_files])
    arm_day_filter = np.where((arm_sonde_dts >= time_start_this_day) & (arm_sonde_dts <= time_end_this_day))[0]
    arm_sonde_files_this_day = arm_sonde_files[arm_day_filter]
    if len(arm_sonde_files) > 0:
        arm_sonde_dts_this_day = arm_sonde_dts[arm_day_filter]
        arm_sonde_lons = []
        arm_sonde_lats = []

        for sonde_file in arm_sonde_files_this_day:
            tmp_sonde = xr.open_dataset(sonde_file)
            arm_sonde_lons.append(tmp_sonde.lon.data[0])
            arm_sonde_lats.append(tmp_sonde.lat.data[0])
            tmp_sonde.close()

        arm_sonde_lons = np.array(arm_sonde_lons)
        arm_sonde_lats = np.array(arm_sonde_lats)

        arm_sonde_sbf_side = identify_side(arm_sonde_dts_this_day.astype('datetime64[s]').astype(float), arm_sonde_lons, arm_sonde_lats, tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                                    tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
    else:
        print('Warning, no ARM sondes found!')
        arm_sonde_files_this_day = np.empty(0, dtype=str)
        arm_sonde_dts_this_day = np.empty(0, dtype='datetime64[s]')
        arm_sonde_sbf_side = np.empty(0, dtype=int)
        arm_sonde_lons = np.empty(0, dtype=float)
        arm_sonde_lats = np.empty(0, dtype=float)

    # Load the TAMU sondes
    tamu_sonde_path = '/Volumes/LtgSSD/TAMU_SONDES/'
    tamu_sonde_files = sorted(listdir(tamu_sonde_path))
    tamu_sonde_dts = np.array([dt.strptime('_'.join(f.split('_')[2:4]), '%Y%m%d_%H%M') for f in tamu_sonde_files]).astype('datetime64[s]')
    tamu_sonde_files = np.array([path.join(tamu_sonde_path, f) for f in tamu_sonde_files])
    tamu_day_filter = np.where((tamu_sonde_dts >= time_start_this_day) & (tamu_sonde_dts <= time_end_this_day))[0]
    tamu_sonde_files_this_day = tamu_sonde_files[tamu_day_filter]
    if len(tamu_sonde_files_this_day) > 0:
        tamu_sonde_dts_this_day = tamu_sonde_dts[tamu_day_filter]
        tamu_sonde_files_split = np.vstack(np.char.split(tamu_sonde_files_this_day, sep='_'))
        tamu_sonde_lons = tamu_sonde_files_split[:, -3]
        lon_negative = ((np.char.find(tamu_sonde_lons, 'W') >= 0).astype(int) - 0.5) * -2
        tamu_sonde_lons = np.char.replace(tamu_sonde_lons, 'W', '')
        tamu_sonde_lons = np.char.replace(tamu_sonde_lons, 'E', '')
        tamu_sonde_lons = tamu_sonde_lons.astype(float) * lon_negative

        tamu_sonde_lats = tamu_sonde_files_split[:, -2]
        lat_negative = ((np.char.find(tamu_sonde_lats, 'S') >= 0).astype(int) - 0.5) * -2
        tamu_sonde_lats = np.char.replace(tamu_sonde_lats, 'S', '')
        tamu_sonde_lats = np.char.replace(tamu_sonde_lats, 'N', '')
        tamu_sonde_lats = tamu_sonde_lats.astype(float) * lat_negative

        tamu_sonde_sbf_side = identify_side(tamu_sonde_dts_this_day.astype('datetime64[s]').astype(float), tamu_sonde_lons, tamu_sonde_lats, tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                                    tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
        
    else:
        print('Warning, no TAMU sondes found!')
        tamu_sonde_files_this_day = np.empty(0, dtype=str)
        tamu_sonde_dts_this_day = np.empty(0, dtype='datetime64[s]')
        tamu_sonde_sbf_side = np.empty(0, dtype=int)
        tamu_sonde_lons = np.empty(0, dtype=float)
        tamu_sonde_lats = np.empty(0, dtype=float)

    # Load the CMAS sondes
    CMAS_sonde_path = '/Volumes/LtgSSD/CMAS-sondes/'
    CMAS_sonde_files = [f for f in sorted(listdir(CMAS_sonde_path)) if f.startswith('GrawSonde') and f.endswith('.nc')]
    CMAS_sonde_dts = np.array([dt.strptime('_'.join(f.split('_')[3:5]), '%Y%m%d_%H%M%S.nc') for f in CMAS_sonde_files]).astype('datetime64[s]')
    CMAS_sonde_files = np.array([path.join(CMAS_sonde_path, f) for f in CMAS_sonde_files if f.startswith('GrawSonde') and f.endswith('.nc')])
    CMAS_day_filter = np.where((CMAS_sonde_dts >= time_start_this_day) & (CMAS_sonde_dts <= time_end_this_day))[0]
    CMAS_sonde_files_this_day = CMAS_sonde_files[CMAS_day_filter]
    if len(CMAS_sonde_files_this_day) > 0:
        CMAS_sonde_dts_this_day = CMAS_sonde_dts[CMAS_day_filter]
        CMAS_sonde_lons = []
        CMAS_sonde_lats = []

        for sonde_file in CMAS_sonde_files_this_day:
            tmp_sonde = xr.open_dataset(sonde_file)
            CMAS_sonde_lons.append(tmp_sonde.longitude.data[0])
            CMAS_sonde_lats.append(tmp_sonde.latitude.data[0])
            tmp_sonde.close()

        CMAS_sonde_lons = np.array(CMAS_sonde_lons)
        CMAS_sonde_lats = np.array(CMAS_sonde_lats)

        CMAS_sonde_sbf_side = identify_side(CMAS_sonde_dts_this_day.astype('datetime64[s]').astype(float), CMAS_sonde_lons, CMAS_sonde_lats, tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                                    tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
    else:
        print('Warning, no CMAS sondes found!')
        CMAS_sonde_files_this_day = np.empty(0, dtype=str)
        CMAS_sonde_dts_this_day = np.empty(0, dtype='datetime64[s]')
        CMAS_sonde_sbf_side = np.empty(0, dtype=int)
        CMAS_sonde_lons = np.empty(0, dtype=float)
        CMAS_sonde_lats = np.empty(0, dtype=float)

    # Load ACARS profiles
    acars_all_list = []
    airports_i_want = {'IAH' : {'latitude' : 29.9844353, 'longitude' : -95.3414425},
                    'HOU' : {'latitude' : 29.6457998, 'longitude' : -95.2772316},
                    'LBX' : {'latitude' : 29.1086389, 'longitude' : -95.4620833},
                    'GLS' : {'latitude' : 29.2653333, 'longitude' : -94.8604167},
                    'EFD' : {'latitude' : 29.6073333, 'longitude' : -95.1587500},
                    'IWS' : {'latitude' : 29.8181944, 'longitude' : -95.6726111},
                    'SGR' : {'latitude' : 29.6222486, 'longitude' : -95.6565342},
                    'DWH' : {'latitude' : 30.0617791, 'longitude' : -95.5527884},
                    'CXO' : {'latitude' : 30.3533955, 'longitude' : -95.4150819},
                    'HPY' : {'latitude' : 29.7860833, 'longitude' : -94.9526667}
                    }
    for hour_to_pull in range(24):
        acars_conn = spy.acars_data(str(date_i_want.year).zfill(4), str(date_i_want.month).zfill(2), str(date_i_want.day).zfill(2), str(hour_to_pull).zfill(2))
        try:
            this_hour_acars_list = [prof for prof in acars_conn.list_profiles() if prof[:3] in airports_i_want.keys()]
        except Exception as e:
            if 'HTTP Error' in str(e):
                continue
            else:
                raise e
        acars_all_list.extend(this_hour_acars_list)
    acars_profile_dts = []
    acars_profile_lons = []
    acars_profile_lats = []
    acars_profile_files = []
    for acars_loctime in acars_all_list:
        acars_profile_dts.append(date_i_want.replace(hour=int(acars_loctime[-4:-2]), minute=int(acars_loctime[-2:])))
        acars_profile_lats.append(airports_i_want[acars_loctime[:3]]['latitude'])
        acars_profile_lons.append(airports_i_want[acars_loctime[:3]]['longitude'])
        acars_profile_files.append('ACARS+'+acars_loctime)
    acars_profile_dts = np.array(acars_profile_dts).astype('datetime64[s]')
    acars_profile_sbf = identify_side(acars_profile_dts.astype(float), np.array(acars_profile_lons), np.array(acars_profile_lats), tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                        tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
    all_sonde_files = np.concatenate([arm_sonde_files_this_day, tamu_sonde_files_this_day, CMAS_sonde_files_this_day, acars_profile_files])
    all_sonde_dts = np.concatenate([arm_sonde_dts_this_day, tamu_sonde_dts_this_day, CMAS_sonde_dts_this_day, acars_profile_dts])
    all_sonde_sbf_side = np.concatenate([arm_sonde_sbf_side, tamu_sonde_sbf_side, CMAS_sonde_sbf_side, acars_profile_sbf])
    all_sonde_lons = np.concatenate([arm_sonde_lons, tamu_sonde_lons, CMAS_sonde_lons, acars_profile_lons])
    all_sonde_lats = np.concatenate([arm_sonde_lats, tamu_sonde_lats, CMAS_sonde_lats, acars_profile_lats])

    maritime_sonde_dts = all_sonde_dts[all_sonde_sbf_side == -1]
    maritime_sorting = np.argsort(maritime_sonde_dts)
    maritime_sonde_dts = maritime_sonde_dts[maritime_sorting]

    continental_sonde_dts = all_sonde_dts[all_sonde_sbf_side == -2]
    continental_sorting = np.argsort(continental_sonde_dts)
    continental_sonde_dts = continental_sonde_dts[continental_sorting]

    n_sounding_vars = 6 # pressure, temperature, dewpoint, u, v, z
    maritime_representative_profile = np.full((tfm.time.shape[0], n_sounding_levels, n_sounding_vars), -999, dtype=float)
    last_maritime_profile_time_index = -1

    continental_representative_profile = np.full((tfm.time.shape[0], n_sounding_levels, n_sounding_vars), -999, dtype=float)
    last_continental_profile_time_index = -1

    for f, this_dt, sbf, this_lon, this_lat in zip(all_sonde_files, all_sonde_dts, all_sonde_sbf_side, all_sonde_lons, all_sonde_lats):
        this_pydt = this_dt.astype(dt)
        if f.startswith('ACARS'):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                acars_conn = spy.acars_data(str(this_pydt.year).zfill(4), str(this_pydt.month).zfill(2), str(this_pydt.day).zfill(2), str(this_pydt.hour).zfill(2))
                try:
                    sounding = acars_conn.get_profile(f.replace('ACARS+', ''), hush=True, clean_it=True)
                except Exception as e:
                    if 'HTTP Error' in str(e):
                        continue
                    else:
                        raise e
            this_sonde_data = pd.DataFrame(
                {'pres' : sounding['p'].to('hPa').m,
                'tdry' : sounding['T'].to('degC').m,
                'dp' : sounding['Td'].to('degC').m,
                'u_wind' : sounding['u'].to('m/s').m,
                'v_wind' : sounding['v'].to('m/s').m,
                'alt' : sounding['z'].to('m').m}
            )
        elif f.endswith('.cdf'):
            this_sonde_data = xr.open_dataset(f)
        elif f.endswith('.nc'):
            this_sonde_data = xr.open_dataset(f)
            this_sonde_data['pres'] = this_sonde_data.pressure
            this_sonde_data['tdry'] = this_sonde_data.temperature
            this_sonde_data['dp'] = this_sonde_data.dewpoint_temperature
            u, v = mpcalc.wind_components(this_sonde_data.wind_speed * units.meter / units.second, this_sonde_data.wind_direction * units.degree)
            this_sonde_data['u_wind'] = u
            this_sonde_data['v_wind'] = v
            this_sonde_data['alt'] = this_sonde_data.geometric_height
        else:
            this_sonde_data = pd.read_csv(f, skiprows=28, encoding='latin1', sep='\\s+', names=[
                'FlightTime', 'pres', 'tdry', 'RH', 'WindSpeed', 'WindDirection', 'AGL', 'AGL2', 'alt', 'Longitude', 'Latitude', 'y', 'x', 'Tv', 'dp', 'rho',
                'e', 'v_wind', 'u_wind', 'range', 'rv', 'MSL2', 'UTC_DAY', 'UTC_TIME', 'UTC_AMPM', 'ELAPSED_TIME', 'ELAPSED_TIME2', 'ELAPSED_TIME3', 'FrostPoint']
                )
        if len(this_sonde_data.pres.values) < 2:
            continue

        if should_debug:
            save_path = f'./debug-figs-{this_pydt.strftime("%Y%m%d")}/profiles/{this_pydt.strftime("%Y%m%d_%H%M")}.png'
            plot_radiosonde_data(this_sonde_data, this_pydt, this_lon, this_lat, sbf, tfm, save_path)


        otp_check = np.array([-999])
        if sbf == -1:
            otp_check = maritime_representative_profile[last_maritime_profile_time_index, :, 0].copy()
        elif sbf == -2:
            otp_check = continental_representative_profile[last_continental_profile_time_index, :, 0].copy()
        otp_check[otp_check == -999] = np.nan
        old_top_pressure = np.nanmin(otp_check)
        new_top_pressure = np.min(this_sonde_data.pres.values)
        if np.isnan(old_top_pressure):
            n_levels_to_replace = n_sounding_levels
        else:
            if old_top_pressure < new_top_pressure:
                n_levels_to_replace = otp_check[otp_check > new_top_pressure].shape[0]
            else:
                n_levels_to_replace = n_sounding_levels

        new_pres = np.linspace(np.max(this_sonde_data.pres.values), new_top_pressure, n_levels_to_replace)
        new_t, new_dp, new_u, new_v, new_z = interpolate_1d(new_pres, this_sonde_data.pres.values, this_sonde_data.tdry.values,
                                this_sonde_data.dp.values, this_sonde_data.u_wind.values, this_sonde_data.v_wind.values,
                                this_sonde_data.alt.values)
        if n_levels_to_replace < n_sounding_levels:
            if sbf == -1:
                this_rep_profile = maritime_representative_profile[last_maritime_profile_time_index, :, :].copy()
                this_rep_profile[0:n_levels_to_replace, :] = np.vstack([new_pres, new_t, new_dp, new_u, new_v, new_z]).T
            else:
                this_rep_profile = continental_representative_profile[last_continental_profile_time_index, :, :].copy()
                this_rep_profile[0:n_levels_to_replace, :] = np.vstack([new_pres, new_t, new_dp, new_u, new_v, new_z]).T
        else:
            this_rep_profile = np.vstack([new_pres, new_t, new_dp, new_u, new_v, new_z]).T

        closest_time_index = np.argmin(np.abs(tfm.time.data - this_dt))
        if sbf == -1:
            # This is a maritime sounding
            maritime_representative_profile[closest_time_index, :, :] = this_rep_profile
            if last_maritime_profile_time_index != -1:
                maritime_representative_profile[last_maritime_profile_time_index+1:closest_time_index, :, :] = interp_sounding_times(tfm.time.data, last_maritime_profile_time_index, closest_time_index, maritime_representative_profile)
            else:
                maritime_representative_profile[0:closest_time_index, :, :] = this_rep_profile
            last_maritime_profile_time_index = closest_time_index
        elif sbf == -2:
            # This is a continental sounding
            continental_representative_profile[closest_time_index, :, :] = this_rep_profile
            if last_continental_profile_time_index != -1:
                continental_representative_profile[last_continental_profile_time_index+1:closest_time_index, :, :] = interp_sounding_times(tfm.time.data, last_continental_profile_time_index, closest_time_index, continental_representative_profile)
            else:
                last_continental_profile_time_index = closest_time_index
                continental_representative_profile[0:closest_time_index, :, :] = this_rep_profile
            last_continental_profile_time_index = closest_time_index
        
    continental_representative_profile[last_continental_profile_time_index+1:, :, :] = continental_representative_profile[last_continental_profile_time_index, :, :]
    maritime_representative_profile[last_maritime_profile_time_index+1:, :, :] = maritime_representative_profile[last_maritime_profile_time_index, :, :]

    new_maritime_vars = {'maritime_pressure_profile' : (('time', 'vertical_levels'), maritime_representative_profile[:, :, 0]),
        'maritime_temperature_profile' : (('time', 'vertical_levels'), maritime_representative_profile[:, :, 1]),
        'maritime_dewpoint_profile' : (('time', 'vertical_levels'), maritime_representative_profile[:, :, 2]),
        'maritime_u_profile' : (('time', 'vertical_levels'), maritime_representative_profile[:, :, 3]),
        'maritime_v_profile' : (('time', 'vertical_levels'), maritime_representative_profile[:, :, 4]),
        'maritime_msl_profile' : (('time', 'vertical_levels'), maritime_representative_profile[:, :, 5])
        }

    new_continental_vars = {
            'continental_pressure_profile' : (('time', 'vertical_levels'), continental_representative_profile[:, :, 0]),
            'continental_temperature_profile' : (('time', 'vertical_levels'), continental_representative_profile[:, :, 1]),
            'continental_dewpoint_profile' : (('time', 'vertical_levels'), continental_representative_profile[:, :, 2]),
            'continental_u_profile' : (('time', 'vertical_levels'), continental_representative_profile[:, :, 3]),
            'continental_v_profile' : (('time', 'vertical_levels'), continental_representative_profile[:, :, 4]),
            'continental_msl_profile' : (('time', 'vertical_levels'), continental_representative_profile[:, :, 5])
        }

    tfm_w_profiles = tfm.copy().assign_coords(vertical_levels=np.arange(n_sounding_levels)).assign(new_maritime_vars).assign(new_continental_vars)

    for var in new_maritime_vars.keys():
        tfm_w_profiles[var].attrs['units'] = 'hPa' if 'pressure' in var else 'C' if 'temperature' in var else 'm/s' if 'u' in var or 'v' in var else 'm'

    for var in new_continental_vars.keys():
        tfm_w_profiles[var].attrs['units'] = 'hPa' if 'pressure' in var else 'C' if 'temperature' in var else 'm/s' if 'u' in var or 'v' in var else 'm'

    tfm_w_profiles.attrs['soundings_used'] = [path.basename(f) for f in all_sonde_files]

    if should_debug:
        for side in ['continental', 'maritime']:
            save_path = f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/profiles/{side}_representative.png'
            if not path.exists(save_path):
                px = 1/plt.rcParams['figure.dpi']
                fig, axs = plt.subplots(6, 1)
                fig.set_size_inches(1800*px, 1800*px)
                hpa_handle = axs[0].pcolormesh(tfm_w_profiles.time.data, tfm_w_profiles.vertical_levels.data,
                                               tfm_w_profiles[f'{side}_pressure_profile'].T, cmap='viridis')
                axs[0].set_title('Pressure')
                temp_handle = axs[1].pcolormesh(tfm_w_profiles.time.data, tfm_w_profiles.vertical_levels.data,
                                                tfm_w_profiles[f'{side}_temperature_profile'].T, cmap='turbo')
                axs[1].set_title('Temperature')
                dew_handle = axs[2].pcolormesh(tfm_w_profiles.time.data, tfm_w_profiles.vertical_levels.data,
                                               tfm_w_profiles[f'{side}_dewpoint_profile'].T, cmap='BrBG')
                axs[2].set_title('Dewpoint')
                u_handle = axs[3].pcolormesh(tfm_w_profiles.time.data, tfm_w_profiles.vertical_levels.data,
                                             tfm_w_profiles[f'{side}_u_profile'].T, cmap='RdBu')
                axs[3].set_title('East component Wind')
                v_handle = axs[4].pcolormesh(tfm_w_profiles.time.data, tfm_w_profiles.vertical_levels.data,
                                             tfm_w_profiles[f'{side}_v_profile'].T, cmap='RdBu')
                axs[4].set_title('North component Wind')
                msl_handle = axs[5].pcolormesh(tfm_w_profiles.time.data, tfm_w_profiles.vertical_levels.data,
                                               tfm_w_profiles[f'{side}_msl_profile'].T, cmap='viridis')
                axs[5].set_title('Height')
                for ax in axs:
                    [ax.axvline(this_t, c='k', ls=':') for this_t in all_sonde_dts.astype('datetime64[s]').astype('O')]
                fig.colorbar(hpa_handle, ax=axs[0], label='hPa', orientation='vertical')
                fig.colorbar(temp_handle, ax=axs[1], label='°C', orientation='vertical')
                fig.colorbar(dew_handle, ax=axs[2], label='°C', orientation='vertical')
                fig.colorbar(u_handle, ax=axs[3], label='m/s', orientation='vertical')
                fig.colorbar(v_handle, ax=axs[4], label='m/s', orientation='vertical')
                fig.colorbar(msl_handle, ax=axs[5], label='m', orientation='vertical')
                fig.suptitle(f'{date_i_want.strftime("%Y-%m-%d")}\n{side.capitalize()} Representative Profiles')
                fig.savefig(save_path)
                plt.close(fig)

    return tfm_w_profiles


def identify_madis(tfmtime, madis_ds_temp, madis_ds_dew, madis_ds_time, madis_ds_lat, madis_ds_lon, polyline):
    maritime_temp = np.full(tfmtime.shape, np.nan)
    maritime_dew = np.full(tfmtime.shape, np.nan)
    continental_temp = np.full(tfmtime.shape, np.nan)
    continental_dew = np.full(tfmtime.shape, np.nan)
    for i in np.arange(tfmtime.shape[0]):
        time = tfmtime[i]
        if time not in polyline.index.values:
            maritime_temp[i] = np.nan
            maritime_dew[i] = np.nan
            continental_temp[i] = np.nan
            continental_dew[i] = np.nan
            continue
        lower_time_bound = time - 3600
        in_window = ((madis_ds_time <= time) & (madis_ds_time >= lower_time_bound))
        temp_in_window = madis_ds_temp[in_window]
        dew_in_window = madis_ds_dew[in_window]
        lat_in_window = madis_ds_lat[in_window]
        lon_in_window = madis_ds_lon[in_window]
        
        this_polyline = polyline.loc[time]['geometry']
        this_polyline_mpl = Path(np.array(this_polyline.exterior.coords))
        sbf_window = this_polyline_mpl.contains_points(np.array([lon_in_window.flatten(), lat_in_window.flatten()]).T).reshape(lon_in_window.shape).astype(int) - 2
        maritime_temp[i] = np.nanmean(temp_in_window[sbf_window == -1])
        maritime_dew[i] = np.nanmean(dew_in_window[sbf_window == -1])
        continental_temp[i] = np.nanmean(temp_in_window[sbf_window == -2])
        continental_dew[i] = np.nanmean(dew_in_window[sbf_window == -2])
    return maritime_temp, maritime_dew, continental_temp, continental_dew


def add_madis_data(tfm, should_debug=False, client=None):
    date_i_want = tfm.time.data[0].astype('datetime64[D]').astype(dt)
    grid_max_lon = tfm.lon.max().compute()
    grid_min_lon = tfm.lon.min().compute()
    grid_max_lat = tfm.lat.max().compute()
    grid_min_lat = tfm.lat.min().compute()
    madis_file = path.join(path.sep, 'Volumes', 'LtgSSD', 'sfcdata_madis', date_i_want.strftime('%Y%m%d_*'))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        madis_ds = xr.open_mfdataset(madis_file, engine='netcdf4', chunks='auto', coords='minimal', concat_dim='recNum', combine='nested', compat='override')
    madis_ds = madis_ds.where(((madis_ds.longitude <= grid_max_lon) & (madis_ds.longitude >= grid_min_lon) & (madis_ds.latitude <= grid_max_lat) & (madis_ds.latitude >= grid_min_lat)).compute(), drop=True)
    dims_to_rm = list(madis_ds.dims)
    dims_to_rm.remove('recNum')
    madis_ds = madis_ds.drop_dims(dims_to_rm)
    madis_ds_temp = madis_ds.temperature.data
    madis_ds_temp_qc = madis_ds.temperatureQCR.data

    madis_ds_dew = madis_ds.dewpoint.data
    madis_ds_dew_qc = madis_ds.dewpointQCR.data

    madis_ds_time = madis_ds.observationTime.data
    madis_ds_lat = madis_ds.latitude.data
    madis_ds_lon = madis_ds.longitude.data

    madis_ds_invalid = np.zeros_like(madis_ds_temp, dtype=bool)
    madis_ds_invalid[((madis_ds_temp_qc != 0) | (madis_ds_dew_qc != 0) | np.isnan(madis_ds_temp) | np.isnan(madis_ds_dew)).compute()] = True

    madis_ds_temp[madis_ds_invalid] = np.nan
    madis_ds_temp = madis_ds_temp.compute()
    madis_ds_dew[madis_ds_invalid] = np.nan
    madis_ds_dew = madis_ds_dew.compute()
    madis_ds_time[madis_ds_invalid] = np.datetime64('NaT')
    madis_ds_time = madis_ds_time.astype('datetime64[s]').compute()
    madis_ds_lat[madis_ds_invalid] = np.nan
    madis_ds_lat = madis_ds_lat.compute()
    madis_ds_lon[madis_ds_invalid] = np.nan
    madis_ds_lon = madis_ds_lon.compute()
    polyline = gpd.read_file(f'/Volumes/LtgSSD/analysis/sam_polyline/{date_i_want.strftime("%Y-%m-%d")}_interpolated.json').set_index('index')
    maritime_temp, maritime_dew, continental_temp, continental_dew = identify_madis(tfm.time.data.astype('datetime64[s]').astype(float), madis_ds_temp, madis_ds_dew,
               madis_ds_time.astype('datetime64[s]').astype(float), madis_ds_lat, madis_ds_lon, polyline)
    tfm_w_sfc = tfm.copy()
    tfm_w_sfc.maritime_dewpoint_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(maritime_dew)] = maritime_dew[~np.isnan(maritime_dew)] - 273.15
    tfm_w_sfc.maritime_temperature_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(maritime_temp)] = maritime_temp[~np.isnan(maritime_temp)] - 273.15
    tfm_w_sfc.continental_dewpoint_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(continental_dew)] = continental_dew[~np.isnan(continental_dew)] - 273.15
    tfm_w_sfc.continental_temperature_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(continental_temp)] = continental_temp[~np.isnan(continental_temp)] - 273.15
    if should_debug:
        timeseries_fig = plt.figure()
        ax = timeseries_fig.gca()
        ax.plot(tfm.time.data, tfm_w_sfc.maritime_temperature_profile.transpose('vertical_levels', 'time').data[0, :], label='Maritime Temperature', color='blue')
        ax.plot(tfm.time.data, tfm_w_sfc.continental_temperature_profile.transpose('vertical_levels', 'time').data[0, :], label='Continental Temperature', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.set_title(f'Temperature Time Series for {date_i_want.strftime("%Y-%m-%d")}')
        timeseries_fig.savefig(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/sfc/madis_temp_timeseries.png')
        plt.close(timeseries_fig)

    all_res = []
    if should_debug:
        for side, sidenum in zip(['continental', 'maritime'], [-2, -1]):
            for i in np.arange(tfm.time.shape[0]):
                this_profile = pd.DataFrame({
                    'pres' : tfm_w_sfc[f'{side}_pressure_profile'].compute().transpose('time', 'vertical_levels')[i, :],
                    'tdry' : tfm_w_sfc[f'{side}_temperature_profile'].compute().transpose('time', 'vertical_levels')[i, :],
                    'dp' : tfm_w_sfc[f'{side}_dewpoint_profile'].compute().transpose('time', 'vertical_levels')[i, :],
                    'u_wind' : tfm_w_sfc[f'{side}_u_profile'].compute().transpose('time', 'vertical_levels')[i, :],
                    'v_wind' : tfm_w_sfc[f'{side}_v_profile'].compute().transpose('time', 'vertical_levels')[i, :],
                    'alt' : tfm_w_sfc[f'{side}_msl_profile'].compute().transpose('time', 'vertical_levels')[i, :]
                })
                save_path = f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/sfc/{side}_{str(i).zfill(4)}.png'
                if client is not None:
                    all_res.append(client.submit(plot_radiosonde_data, this_profile, tfm.time.data.astype('datetime64[s]').astype('O')[i], None, None, sidenum, None, save_path))
                else:
                    plot_radiosonde_data(this_profile, tfm.time.data.astype('datetime64[s]').astype('O')[i], None, None, sidenum, None, save_path)
        if client is not None:
            client.gather(all_res)
    return tfm_w_sfc


def compute_sounding_stats(tfm):
    feature_pressure_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))
    feature_msl_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))
    feature_temp_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))
    feature_dew_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))
    feature_u_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))
    feature_v_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))
    feature_ccn_profile = np.zeros((tfm.feature.shape[0], tfm.vertical_levels.shape[0]))

    feature_mlcape = np.zeros((tfm.feature.shape[0]))
    feature_mlcin = np.zeros((tfm.feature.shape[0]))
    feature_mlecape = np.zeros((tfm.feature.shape[0]))
    feature_lcl = np.zeros((tfm.feature.shape[0]))
    feature_lfc = np.zeros((tfm.feature.shape[0]))
    feature_el = np.zeros((tfm.feature.shape[0]))
    feature_ccl = np.zeros((tfm.feature.shape[0]))

    for sidenum, side in enumerate(['continental', 'maritime']):
        sidenum -= 2
        temp = tfm[f'{side}_temperature_profile'].data * units.degC
        dew = tfm[f'{side}_dewpoint_profile'].data * units.degC
        pressure = tfm[f'{side}_pressure_profile'].data * units.hPa
        height = tfm[f'{side}_msl_profile'].data * units.m
        u = tfm[f'{side}_u_profile'].data * (units.m/units.s)
        v = tfm[f'{side}_v_profile'].data * (units.m/units.s)
        ccn = tfm[f'{side}_ccn_profile_0.6'].data

        mlcapes = np.zeros(tfm.time.shape[0])
        mlcins = np.zeros(tfm.time.shape[0])
        mlecapes = np.zeros(tfm.time.shape[0])
        lcls = np.zeros(tfm.time.shape[0])
        lfcs = np.zeros(tfm.time.shape[0])
        els = np.zeros(tfm.time.shape[0])
        ccls = np.zeros(tfm.time.shape[0])

        for i in range(tfm.time.shape[0]):
            temp_i = temp[i, :]
            dew_i = dew[i, :]
            pressure_i = pressure[i, :]
            height_i = height[i, :]
            u_i = u[i, :]
            v_i = v[i, :]

            if np.any(pressure_i.m == -999.):
                mlcapes[i] = np.nan
                mlcins[i] = np.nan
                mlecapes[i] = np.nan
                lcls[i] = np.nan
                lfcs[i] = np.nan
                els[i] = np.nan
                continue
            spc_hum = mpcalc.specific_humidity_from_dewpoint(pressure_i, dew_i)
            try:
                mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pressure_i, temp_i, dew_i)
                mlecape = calc_ecape(height_i, pressure_i, temp_i, spc_hum, u_i, v_i, cape_type='mixed_layer')
            except Exception:
                print(f'manually added EL to ecape calculation at time {tfm.time.data[i]}')
                temp_botch = temp_i.copy()
                temp_botch[-1] = 100 * units.degC
                dew_botch = dew_i.copy()
                dew_botch[-1] = -50 * units.degC
                spc_hum_botch = mpcalc.specific_humidity_from_dewpoint(pressure_i, dew_botch)
                try:
                    mlecape = calc_ecape(height_i, pressure_i, temp_botch, spc_hum_botch, u_i, v_i, cape_type='mixed_layer')
                    mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pressure_i, temp_botch, dew_botch)
                except Exception:
                    mlecape = 0 * units('J/kg')
                    mlcape = 0 * units('J/kg')
                    mlcin = 0 * units('J/kg')
            lcl, _ = mpcalc.lcl(pressure_i[0], temp_i[0], dew_i[0])
            lfc, _ = mpcalc.lfc(pressure_i, temp_i, dew_i)
            el, _ = mpcalc.el(pressure_i, temp_i, dew_i)
            try:
                ccl, _, _ = mpcalc.ccl(pressure_i, temp_i, dew_i)
            except IndexError:
                ccl = np.nan * units.m

            if type(mlecape) == int and mlecape == 0:
                mlecape = 0 * units('J/kg')
            mlcapes[i] = mlcape.magnitude
            mlcins[i] = mlcin.magnitude
            mlecapes[i] = mlecape.magnitude
            lcls[i] = lcl.magnitude
            lfcs[i] = lfc.magnitude
            els[i] = el.magnitude
            ccls[i] = ccl.magnitude

        features_matching = np.where(tfm.feature_seabreeze.data == sidenum)[0]
        for matching_feat_idx in features_matching:
            matching_time_idx = tfm.feature_time_index.data[matching_feat_idx]
            feature_pressure_profile[matching_feat_idx, :] = pressure[matching_time_idx, :]
            feature_msl_profile[matching_feat_idx, :] = height[matching_time_idx, :]
            feature_temp_profile[matching_feat_idx, :] = temp[matching_time_idx, :]
            feature_dew_profile[matching_feat_idx, :] = dew[matching_time_idx, :]
            feature_u_profile[matching_feat_idx, :] = u[matching_time_idx, :]
            feature_v_profile[matching_feat_idx, :] = v[matching_time_idx, :]
            feature_ccn_profile[matching_feat_idx, :] = ccn[matching_time_idx, :]

            feature_mlcape[matching_feat_idx] = mlcapes[matching_time_idx]
            feature_mlcin[matching_feat_idx] = mlcins[matching_time_idx]
            feature_mlecape[matching_feat_idx] = mlecapes[matching_time_idx]
            feature_lcl[matching_feat_idx] = lcls[matching_time_idx]
            feature_lfc[matching_feat_idx] = lfcs[matching_time_idx]
            feature_el[matching_feat_idx] = els[matching_time_idx]
            feature_ccl[matching_feat_idx] = ccls[matching_time_idx]

    tfm_stats = tfm.copy()
    tfm_stats = tfm_stats.assign({
        'feature_pressure_profile' : (('feature', 'vertical_levels'), feature_pressure_profile),
        'feature_msl_profile' : (('feature', 'vertical_levels'), feature_msl_profile),
        'feature_temp_profile' : (('feature', 'vertical_levels'), feature_temp_profile),
        'feature_dew_profile' : (('feature', 'vertical_levels'), feature_dew_profile),
        'feature_u_profile' : (('feature', 'vertical_levels'), feature_u_profile),
        'feature_v_profile' : (('feature', 'vertical_levels'), feature_v_profile),
        'feature_ccn_profile_0.6' : (('feature', 'vertical_levels'), feature_ccn_profile),
        'feature_mlcape' : (('feature',), feature_mlcape),
        'feature_mlcin' : (('feature',), feature_mlcin),
        'feature_mlecape' : (('feature',), feature_mlecape),
        'feature_lcl' : (('feature',), feature_lcl),
        'feature_lfc' : (('feature',), feature_lfc),
        'feature_el' : (('feature',), feature_el),
        'feature_ccl' : (('feature',), feature_ccl)
    })
    return tfm_stats


def add_sfc_aerosol_data(tfm, ss_lower_bound=0.6, ss_upper_bound=0.8, ss_target=0.6):
    date_i_want = tfm.time.data[0].astype('datetime64[D]').astype(dt)
    arm_ccn_path = '/Volumes/LtgSSD/arm-ccn-avg/'
    arm_ccn_files = glob(arm_ccn_path+date_i_want.strftime('*%Y%m%d*.nc'))
    maritime_ccn = []
    maritime_times = []
    continental_ccn = []
    continental_times = []
    if len(arm_ccn_files) == 1:
        arm_ccn_file = arm_ccn_files[0]
        arm_ccn = xr.open_dataset(arm_ccn_file)
        arm_ccn_ccn = arm_ccn.N_CCN.data
        arm_ccn_time = arm_ccn.time.data
        readings_in_window = ((arm_ccn.supersaturation_calculated >= ss_lower_bound) & (arm_ccn.supersaturation_calculated <= ss_upper_bound))
        arm_ccn_ccn_window = arm_ccn_ccn[readings_in_window]
        arm_ccn_time_window = arm_ccn_time[readings_in_window]
        arm_ccn_lon = np.full(arm_ccn_time_window.shape, arm_ccn.lon.data)
        arm_ccn_lat = np.full(arm_ccn_time_window.shape, arm_ccn.lat.data)
        arm_ccn_sbf = identify_side(arm_ccn_time_window.astype('datetime64[s]').astype(float), arm_ccn_lon, arm_ccn_lat, tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                            tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
        
        arm_ccn_maritime = arm_ccn_ccn_window[arm_ccn_sbf == -1]
        arm_maritime_time = arm_ccn_time_window[arm_ccn_sbf == -1]
        maritime_ccn.extend(arm_ccn_maritime.tolist())
        maritime_times.extend(arm_maritime_time.tolist())
        arm_ccn_continental = arm_ccn_ccn_window[arm_ccn_sbf == -2]
        arm_continental_time = arm_ccn_time_window[arm_ccn_sbf == -2]
        continental_ccn.extend(arm_ccn_continental.tolist())
        continental_times.extend(arm_continental_time.tolist())
    else:
        print(f'Warning, {len(arm_ccn_files)} ARM CCN files found!')
    tamu_ccn_path = '/Volumes/LtgSSD/brooks-ccn/'
    tamu_ccn_files = glob(tamu_ccn_path+date_i_want.strftime('*%y%m%d_ccn*.csv'))
    if len(tamu_ccn_files) == 1:
        tamu_ccn_file = tamu_ccn_files[0]
        tamu_ccn = pd.read_csv(tamu_ccn_file)
        tamu_ccn = tamu_ccn[tamu_ccn['SS'] == ss_target]
        tamu_times_window = pd.to_datetime(tamu_ccn.loc[:, 'Time'], format='%y%m%d %H:%M:%S').values
        tamu_ccn_window = tamu_ccn.loc[:, 'N_CCN'].values
        tamu_ccn_lon_window = tamu_ccn.loc[:, 'Longitude'].values
        tamu_ccn_lat_window = tamu_ccn.loc[:, 'Latitude'].values
        tamu_ccn_sbf = identify_side(tamu_times_window.astype('datetime64[s]').astype(float), tamu_ccn_lon_window, tamu_ccn_lat_window, tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                            tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
        tamu_ccn_maritime = tamu_ccn_window[tamu_ccn_sbf == -1]
        tamu_maritime_time = tamu_times_window[tamu_ccn_sbf == -1]
        maritime_ccn.extend(tamu_ccn_maritime.tolist())
        maritime_times.extend(tamu_maritime_time.tolist())
        tamu_ccn_continental = tamu_ccn_window[tamu_ccn_sbf == -2]
        tamu_continental_time = tamu_times_window[tamu_ccn_sbf == -2]
        continental_ccn.extend(tamu_ccn_continental.tolist())
        continental_times.extend(tamu_continental_time.tolist())
    else:
        print(f'Warning, {len(tamu_ccn_files)} TAMU CCN files found!')
    guy_ccn_path = '/Volumes/LtgSSD/guy-ccn/CCN_20220702to20220910_TRACER_IOP.mat'
    guy_ccn = loadmat(guy_ccn_path)
    varnames = guy_ccn['CCN'][0][0].dtype.names
    data_dict = {varname: guy_ccn['CCN'][0][0][varname].flatten() for varname in varnames}
    guy_ccn_data = pd.DataFrame(data_dict)
    guy_ccn_data['time'] = pd.to_datetime(guy_ccn_data['UTC']-719529, unit='D')
    guy_ccn_data = guy_ccn_data.set_index('time', drop=True)
    guy_ccn_this_day = guy_ccn_data.iloc[(guy_ccn_data.index.values >= tfm.time.data[0]) & (guy_ccn_data.index.values <= tfm.time.data[-1])]
    if len(guy_ccn_this_day) > 0:
        guy_ccn_this_day_this_SS = guy_ccn_this_day[(guy_ccn_this_day['SS_set'] >= ss_lower_bound) & (guy_ccn_this_day['SS_set'] <= ss_upper_bound)]
        guy_ccn_times = guy_ccn_this_day_this_SS.index.values
        guy_ccn_ccn = guy_ccn_this_day_this_SS['n0_all_ccn'].values
        guy_lat = np.full(guy_ccn_times.shape, 29.33)
        guy_lon = np.full(guy_ccn_times.shape, -95.74)
        guy_ccn_sbf = identify_side(guy_ccn_times.astype('datetime64[s]').astype(float), guy_lon, guy_lat, tfm.time.compute().data.astype('datetime64[s]').astype(float),
                                            tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.compute().data, tfm.lat.compute().data)
        guy_ccn_maritime = guy_ccn_ccn[guy_ccn_sbf == -1]
        guy_maritime_time = guy_ccn_times[guy_ccn_sbf == -1]
        maritime_ccn.extend(guy_ccn_maritime.tolist())
        maritime_times.extend(guy_maritime_time.tolist())
        guy_ccn_continental = guy_ccn_ccn[guy_ccn_sbf == -2]
        guy_continental_time = guy_ccn_times[guy_ccn_sbf == -2]
        continental_ccn.extend(guy_ccn_continental.tolist())
        continental_times.extend(guy_continental_time.tolist())
    continental_sorting = np.argsort(continental_times)
    continental_times = np.array(continental_times)[continental_sorting]
    continental_ccn = np.array(continental_ccn)[continental_sorting]
    maritime_sorting = np.argsort(maritime_times)
    maritime_times = np.array(maritime_times)[maritime_sorting]
    maritime_ccn = np.array(maritime_ccn)[maritime_sorting]
    if len(maritime_ccn) == 0:
        print('No maritime CCN data found!')
        maritime_ccn_vert = np.full((tfm.time.shape[0], tfm.vertical_levels.shape[0]), np.nan)
    else:
        maritime_ccn_interper = interp1d(maritime_times, maritime_ccn, kind='linear', bounds_error=False, fill_value=(maritime_ccn[0], maritime_ccn[-1]))
        maritime_ccn_interp = maritime_ccn_interper(tfm.time.data)
        maritime_ccn_vert = np.repeat(maritime_ccn_interp[:, np.newaxis], tfm.vertical_levels.shape[0], axis=1)

    if len(continental_ccn) == 0:
        print('No continental CCN data found!')
        continental_ccn_vert = np.full((tfm.time.shape[0], tfm.vertical_levels.shape[0]), np.nan)
    else:
        continental_ccn_interper = interp1d(continental_times, continental_ccn, kind='linear', bounds_error=False, fill_value=(continental_ccn[0], continental_ccn[-1]))
        continental_ccn_interp = continental_ccn_interper(tfm.time.data)
        continental_ccn_vert = np.repeat(continental_ccn_interp[:, np.newaxis], tfm.vertical_levels.shape[0], axis=1)

    tfm_w_aerosols = tfm.copy()
    tfm_w_aerosols = tfm_w_aerosols.assign({
        f'maritime_ccn_profile_{ss_target:.1f}' : (('time', 'vertical_levels'), maritime_ccn_vert),
        f'continental_ccn_profile_{ss_target:.1f}' : (('time', 'vertical_levels'), continental_ccn_vert)
    })

    return tfm_w_aerosols


def apply_coord_transforms(tfm, should_debug=False):
    tfm = tfm.copy()
    radar_lat, radar_lon = 29.47190094, -95.07873535
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=radar_lat, ctrLon=radar_lon, ctrAlt=0)
    
    x2d, y2d = np.meshgrid(tfm.x.data, tfm.y.data)
    grid_ecef_coords = tpcs.toECEF(x2d.flatten(), y2d.flatten(), np.zeros_like(x2d).flatten())
    
    geosys = coords.GeographicSystem()
    grid_lon, grid_lat, _ = geosys.fromECEF(*grid_ecef_coords)
    grid_lon = grid_lon.reshape(x2d.shape)
    grid_lat = grid_lat.reshape(x2d.shape)
    tfm = tfm.assign({'lat' : (('x', 'y'), grid_lat), 'lon' : (('x', 'y'), grid_lon)})


    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=-75., sweep_axis='x')#, ellipse=ltg_ell)
    grid_g16_scan_x, grid_g16_scan_y, _ = satsys.fromECEF(*grid_ecef_coords)
    grid_g16_scan_x = grid_g16_scan_x.reshape(x2d.shape)
    grid_g16_scan_y = grid_g16_scan_y.reshape(x2d.shape)
    tfm = tfm.assign({'g16_scan_x' : (('x', 'y'), grid_g16_scan_x), 'g16_scan_y' : (('x', 'y'), grid_g16_scan_y)})

    tfm.attrs['center_lat'] = radar_lat
    tfm.attrs['center_lon'] = radar_lon

    feat_x = tfm.feature_projection_x_coordinate.compute().data
    feat_y = tfm.feature_projection_y_coordinate.compute().data
    feat_z = np.zeros_like(feat_x)
    feature_ecef_coords = tpcs.toECEF(feat_x, feat_y, feat_z)
    feature_lon, feature_lat, _ = geosys.fromECEF(*feature_ecef_coords)
    tfm = tfm.assign({'feature_lat' : (('feature'), feature_lat), 'feature_lon' : (('feature'), feature_lon)})
    if should_debug:
        date_i_want = tfm.time.data[0].astype('datetime64[D]').astype(dt)
        ax_extent = [tfm.lon.min()-0.25, tfm.lon.max()+0.25, tfm.lat.min()-0.25, tfm.lat.max()+0.25]
        x2d, y2d = np.meshgrid(tfm.x.data, tfm.y.data)
        fig, axs = plt.subplots(2, 3, subplot_kw={'projection': ccrs.PlateCarree()})
        cartx = axs[0, 0].pcolormesh(tfm.lon, tfm.lat, x2d, transform=ccrs.PlateCarree(), cmap='managua')
        axs[0, 0].set_title('Cartesian - x')
        fig.colorbar(cartx, ax=axs[0, 0], orientation='horizontal', label='x')
        carty = axs[1, 0].pcolormesh(tfm.lon, tfm.lat, y2d, transform=ccrs.PlateCarree(), cmap='managua')
        axs[1, 0].set_title('Cartesian - y')
        fig.colorbar(carty, ax=axs[1, 0], orientation='horizontal', label='y')


        geolon = axs[0, 1].pcolormesh(tfm.lon, tfm.lat, tfm.lon, transform=ccrs.PlateCarree(), cmap='managua')
        axs[0, 1].set_title('Longitude')
        fig.colorbar(geolon, ax=axs[0, 1], orientation='horizontal', label='Longitude')
        geolat = axs[1, 1].pcolormesh(tfm.lon, tfm.lat, tfm.lat, transform=ccrs.PlateCarree(), cmap='managua')
        axs[1, 1].set_title('Latitude')
        fig.colorbar(geolat, ax=axs[1, 1], orientation='horizontal', label='Latitude')

        goesx = axs[0, 2].pcolormesh(tfm.lon, tfm.lat, tfm.g16_scan_x, transform=ccrs.PlateCarree(), cmap='managua')
        axs[0, 2].set_title('GOES x')
        fig.colorbar(goesx, ax=axs[0, 2], orientation='horizontal', label='GOES x')
        goesy = axs[1, 2].pcolormesh(tfm.lon, tfm.lat, tfm.g16_scan_y, transform=ccrs.PlateCarree(), cmap='managua')
        axs[1, 2].set_title('GOES y')
        fig.colorbar(goesy, ax=axs[1, 2], orientation='horizontal', label='GOES y')


        for i in range(2):
            for j in range(3):
                ax = axs[i, j]
                ax.add_feature(cfeat.COASTLINE.with_scale('50m'))
                ax.add_feature(cfeat.BORDERS.with_scale('50m'))
                ax.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='gray')
                ax.set_extent(ax_extent)
                ax.scatter(tfm.center_lon, tfm.center_lat, transform=ccrs.PlateCarree(), color='lime', s=1)

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig.set_size_inches(900*px, 600*px)
        fig.tight_layout()
        fig.savefig(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/coords/test.png')
    return tfm


def add_timeseries_data_to_toabc_path(tobac_data, date_i_want, client=None, should_debug=False):
    def make_debug_plot_for_featid(this_feat_time_idx, radar_time, meltlayer):
        mpluse('agg')
        px = 1/plt.rcParams['figure.dpi']
        radar_filepath = f'/Volumes/LtgSSD/nexrad_gridded/{radar_time.strftime("%B").upper()}/{radar_time.strftime("%Y%m%d")}/KHGX{radar_time.strftime("%Y%m%d_%H%M%S")}_V06_grid.nc'
        tfm_time = tfm.isel(time=this_feat_time_idx)
        tfm_time = tfm_time.isel(feature=(tfm_time.feature_time_index == this_feat_time_idx))
        this_time = tfm_time.time.data
        if this_feat_time_idx == 0:
            previous_time = tfm.time.data[this_feat_time_idx] - np.timedelta64(10, 'm')
        else:
            previous_time = tfm.time.data[this_feat_time_idx - 1]
        radar_time = this_time.astype('datetime64[s]').astype(dt).item()
        radar_filepath = f'/Volumes/LtgSSD/nexrad_gridded/{radar_time.strftime("%B").upper()}/{radar_time.strftime("%Y%m%d")}/KHGX{radar_time.strftime("%Y%m%d_%H%M%S")}_V06_grid.nc'
        radar_ds = xr.open_dataset(radar_filepath).isel(time=0)
        closest_to_melt_layer = np.argmin(np.abs(radar_ds.z.data - meltlayer))
        above_melt_layer = radar_ds.isel(z=slice(closest_to_melt_layer, None))
        above_melt_layer_zdr_thresholded = (above_melt_layer.differential_reflectivity > 1.0).sum(dim='z').data.astype(float)
        above_melt_layer_zdr_thresholded[above_melt_layer_zdr_thresholded == 0] = np.nan
        above_melt_layer_kdp_thresholded = (above_melt_layer.KDP_CSU > 0.75).sum(dim='z').data.astype(float)
        above_melt_layer_kdp_thresholded[above_melt_layer_kdp_thresholded == 0] = np.nan
        lightning_filepaths = glob(f'/Volumes/LtgSSD/{int(date_i_want.strftime("%m"))}/6sensor_minimum/LYLOUT_{date_i_want.strftime("%y%m%d")}*.nc')
        if len(lightning_filepaths) != 1:
            raise ValueError('Expected exactly one lightning data file, but found:', len(lightning_filepaths))
        lightning_filepath = lightning_filepaths[0]
        lightning = xr.open_dataset(lightning_filepath)
        lightning_data_at_time = lightning.sel(grid_time=slice(previous_time, this_time))
        flash_mask = (lightning_data_at_time.flash_time_start.data > previous_time) & (lightning_data_at_time.flash_time_end.data < this_time)
        event_mask = (lightning_data_at_time.event_time.data > previous_time) & (lightning_data_at_time.event_time.data < this_time)
        lightning_data_at_time = lightning_data_at_time.isel(number_of_flashes=flash_mask, number_of_events=event_mask)
        x2d, y2d = np.meshgrid(radar_ds.x.values, radar_ds.y.values)
        for feature_i_want in tfm_time.feature.data:
            tfm_feat_time = tfm_time.sel(feature=feature_i_want)
            path_to_save = f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/tobac_ts/{feature_i_want}.png'
            if path.exists(path_to_save):
                continue
            feature_segm_mask = tfm_feat_time.segmentation_mask == feature_i_want
            feature_composite_refl = radar_ds.reflectivity.max(dim='z').data
            feature_composite_refl[~feature_segm_mask] = np.nan
            ys_of_feature = tfm_feat_time.y.data * feature_segm_mask.max(dim='x')
            ys_of_feature[~feature_segm_mask.max(dim='x')] = np.nan
            feature_min_y = ys_of_feature.min().item()
            feature_max_y = ys_of_feature.max().item()

            xs_of_feature = tfm_feat_time.x.data * feature_segm_mask.max(dim='y')
            xs_of_feature[~feature_segm_mask.max(dim='y')] = np.nan
            feature_min_x = xs_of_feature.min().item()
            feature_max_x = xs_of_feature.max().item()
            if tfm_feat_time.feature_area == 0:
                feature_min_x = tfm_feat_time.feature_projection_x_coordinate.data.item()-1000
                feature_max_x = tfm_feat_time.feature_projection_x_coordinate.data.item()+1000
                feature_min_y = tfm_feat_time.feature_projection_y_coordinate.data.item()-1000
                feature_max_y = tfm_feat_time.feature_projection_y_coordinate.data.item()+1000
                zdr_thresh_vmax = 0
                kdp_thresh_vmax = 0
            else:
                zdr_thresh_vmax = np.nanmax(above_melt_layer_zdr_thresholded[feature_segm_mask])
                if zdr_thresh_vmax == 0 or np.isnan(zdr_thresh_vmax):
                    zdr_thresh_vmax = 1
                kdp_thresh_vmax = np.nanmax(above_melt_layer_kdp_thresholded[feature_segm_mask])
                if kdp_thresh_vmax == 0 or np.isnan(kdp_thresh_vmax):
                    kdp_thresh_vmax = 1

            fig, axs = plt.subplots(1, 4, figsize=(20, 7))
            feature_segm_mask_trasparent = feature_segm_mask.data.copy().astype(float)
            feature_segm_mask_trasparent[feature_segm_mask] = np.nan
            reflhandle = axs[0].pcolormesh(tfm.lon, tfm.lat, radar_ds.reflectivity.max(dim='z').data, cmap=ChaseSpectral, vmin=-10, vmax=80)
            axs[0].pcolormesh(tfm.lon, tfm.lat, feature_segm_mask_trasparent, cmap='Greys', alpha=0.75)
            axs[0].scatter(lightning_data_at_time.flash_center_longitude, lightning_data_at_time.flash_center_latitude, c=lightning_data_at_time.flash_id, cmap='tab20', s=10, linewidths=0.5, edgecolors='black')
            axs[0].set_xlabel('Longitude')
            axs[0].set_ylabel('Latitude')
            axs[0].set_title(f'Composite Z, segmask, flash centers\n{radar_time.strftime("%H:%M:%S")}')
            fig.colorbar(reflhandle, ax=axs[0], label='Composite Reflectivity (dBZ)', orientation='horizontal')
            axs[1].pcolormesh(x2d/1000, y2d/1000, feature_composite_refl, cmap=ChaseSpectral, vmin=-10, vmax=80)
            axs[1].set_xlim(feature_min_x/1000, feature_max_x/1000)
            axs[1].set_ylim(feature_min_y/1000, feature_max_y/1000)
            flash_handle = axs[1].scatter(lightning_data_at_time.event_x/1000, lightning_data_at_time.event_y/1000, c=lightning_data_at_time.event_parent_flash_id, cmap='tab20b', s=3, linewidths=0.5, edgecolors='black')
            axs[1].scatter(lightning_data_at_time.flash_ctr_x/1000, lightning_data_at_time.flash_ctr_y/1000, c=lightning_data_at_time.flash_id, cmap='tab20b', marker='*', s=50, linewidths=1, edgecolors='black')
            axs[1].scatter(lightning_data_at_time.flash_init_x/1000, lightning_data_at_time.flash_init_y/1000, c=lightning_data_at_time.flash_id, cmap='tab20b', marker='^', s=50, linewidths=1, edgecolors='black')
            axs[1].set_xlabel('East-West Distance (km)')
            axs[1].set_ylabel('North-South Distance (km)')
            previous_time_dt = previous_time.astype("datetime64[s]").astype(dt)
            axs[1].set_title(f'Composite Reflectivity and Lightning\n{previous_time_dt.strftime("%H:%M:%S")} through {radar_time.strftime("%H:%M:%S")}')
            fig.colorbar(flash_handle, ax=axs[1], label='Flash ID', orientation='horizontal')
            zdr_handle = axs[2].pcolormesh(radar_ds.x/1000, radar_ds.y/1000, above_melt_layer_zdr_thresholded, cmap='tab20b', vmin=0, vmax=zdr_thresh_vmax)
            axs[2].pcolormesh(tfm.x/1000, tfm.y/1000, feature_segm_mask_trasparent, cmap='Greys_r', alpha=0.75)
            axs[2].set_xlim(feature_min_x/1000, feature_max_x/1000)
            axs[2].set_ylim(feature_min_y/1000, feature_max_y/1000)
            axs[2].set_xlabel('East-West Distance (km)')
            axs[2].set_ylabel('North-South Distance (km)')
            axs[2].set_title(f'Grid cells with ZDR > 1.0 above {meltlayer} m AGL')
            fig.colorbar(zdr_handle, ax=axs[2], label='ZDR Volume', orientation='horizontal')
            kdp_handle = axs[3].pcolormesh(radar_ds.x/1000, radar_ds.y/1000, above_melt_layer_kdp_thresholded, cmap='tab20b', vmin=0, vmax=kdp_thresh_vmax)
            axs[3].pcolormesh(tfm.x/1000, tfm.y/1000, feature_segm_mask_trasparent, cmap='Greys_r', alpha=0.75)
            axs[3].set_xlim(feature_min_x/1000, feature_max_x/1000)
            axs[3].set_ylim(feature_min_y/1000, feature_max_y/1000)
            axs[3].set_xlabel('East-West Distance (km)')
            axs[3].set_ylabel('North-South Distance (km)')
            axs[3].set_title(f'Grid cells with KDP > 0.75 above {meltlayer} m AGL')
            fig.colorbar(kdp_handle, ax=axs[3], label='KDP Volume', orientation='horizontal')
            fig.suptitle(f'Feature ID: {feature_i_want} | Flash Count: {tfm_feat_time.feature_flash_count.data.item():.1f} | ZDR Volume: {tfm_feat_time.feature_zdrvol.data.item()} | KDP Volume: {tfm_feat_time.feature_kdpvol.data.item()} | Max Z: {tfm_feat_time.feature_maxrefl.data.item():.2f} | Area: {tfm_feat_time.feature_area.data.item()}')
            fig.tight_layout()
            fig.savefig(path_to_save)
            plt.close(fig)
        return 1
    tfm = tobac_data.copy()
    tobac_save_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime('%Y%m%d')}/'
    melt_layer = 0
    for f in listdir(tobac_save_path):
        if f.startswith('timeseries_data_melt') and f.endswith('.nc'):
            tobac_timeseries_path = path.join(tobac_save_path, f)
            melt_layer = int(f.replace('timeseries_data_melt', '').replace('.nc', ''))
            break
    else:
        raise ValueError('>>>>>>>Unable to find timeseries data...>>>>>>>')
    timeseries_data = xr.open_dataset(tobac_timeseries_path, chunks='auto')
    timeseries_data = timeseries_data.reindex(feature=tfm.feature.data, fill_value=np.nan)
    for dv in timeseries_data.data_vars:
        if dv not in tfm.data_vars:
            tfm[dv] = timeseries_data[dv].copy()

    if should_debug:
        if client is not None:
            tfm.feature_area.load()
            tfm.feature_grid_cell_count.load()
            tfm.feature_threshold_max.load()
            tfm.feature_maxrefl.load()
            for v in ['zdr', 'rhvdeficit', 'kdp']:
                tfm[f'feature_{v}vol'].load()
                tfm[f'feature_{v}col'].load()
                tfm[f'feature_{v}col_mean'].load()
                tfm[f'feature_{v}col_total'].load()
            tfm.feature_flash_count.load()
            tfm.feature_zdrwt_total.load()
            res = []
            for i, t in enumerate(tfm.time.data.astype('datetime64[s]').astype('O')):
                res.append(client.submit(make_debug_plot_for_featid, i, t, melt_layer))
            for promised_res in res:
                _ = promised_res.result()
        else:
            for this_itr in enumerate(tfm.time.data.astype('datetime64[s]').astype('O')):
                make_debug_plot_for_featid(*this_itr, melt_layer)
    return tfm


@njit(parallel=True)
def replace_values(seg_mask, cell_ids):
    seg_mask_flat = seg_mask.flatten()
    seg_mask_cell = np.full(seg_mask_flat.shape, np.nan, dtype=np.float32)
    indices = np.argwhere(~np.isnan(seg_mask_flat)).flatten()
    for i in indices:
        seg_mask_cell[i] = cell_ids[int(seg_mask_flat[i]) - 1]
    return seg_mask_cell.reshape(seg_mask.shape)


def generate_seg_mask_cell_track(tobac_data, convert_to='cell'):
    tobac_data = tobac_data.copy()
    print('-Overwriting 0 in segmask with nan')
    tobac_data['segmentation_mask'] = xr.where(tobac_data.segmentation_mask == 0, np.nan, tobac_data.segmentation_mask.astype(np.float32))
    feature_ids = tobac_data.feature.compute().data
    seg_data_feature = tobac_data.segmentation_mask.compute().data
    cell_ids = tobac_data[f'feature_parent_{convert_to}_id'].sel(feature=feature_ids).compute().data
    print('-Mapping')
    seg_data_cell = replace_values(seg_data_feature, cell_ids)
    print(f'-seg mask {convert_to} to xarray')
    tobac_data[f'segmentation_mask_{convert_to}'] = xr.DataArray(seg_data_cell, dims=('time', 'y', 'x'), coords={'time': tobac_data.time.data, 'y': tobac_data.y.data, 'x': tobac_data.x.data})
    return tobac_data


def convert_to_track_time(tfmo):
    for old_name in ['min_L2-MCMIPC', 'max_reflectivity']:
        new_name = 'feature_'+old_name
        new_name = new_name.replace('-', '_')
        ren = tfmo[old_name].rename(new_name)
        tfmo[new_name] = ren
        tfmo = tfmo.drop_vars(old_name)
        tfmo.drop_vars('feature_maxrefl')

    track_seabreezes = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_area = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), 0)
    track_echo_top = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_flash_count = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_large_flash_count = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_small_flash_count = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_KDP_volume = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_lat_ctr = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_lon_ctr = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_rhoHV_volume = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_ZDR_volume = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_min_L2_MCMIPC = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)

    track_pressure_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)
    track_height_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)
    track_temperature_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)
    track_dewpoint_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)
    track_u_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)
    track_v_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)
    track_ccn_profile = np.full((tfmo.track.shape[0], tfmo.time.shape[0], tfmo.vertical_levels.shape[0]), np.nan)

    track_mlcape = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_mlcin = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_mlecape = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_lcl = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_lfc = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_el = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)
    track_ccl = np.full((tfmo.track.shape[0], tfmo.time.shape[0]), np.nan)

    tfmo['feature_parent_track_id'] = tfmo.feature_parent_track_id.compute().astype('int32')
    vars_to_load_now = ['feature_time_index', 'feature_seabreeze', 'feature_area', 'feature_echotop', 'feature_flash_count', 'feature_flash_count_area_GT_4km',
                        'feature_flash_count_area_LE_4km', 'feature_kdpvol', 'feature_lat', 'feature_lon', 'feature_rhvdeficitvol', 'feature_zdrvol', 'feature_min_L2_MCMIPC',
                        'feature_pressure_profile', 'feature_msl_profile', 'feature_temp_profile', 'feature_dew_profile', 'feature_u_profile', 'feature_v_profile', 'feature_ccn_profile_0.6',
                        'feature_mlcape', 'feature_mlcin', 'feature_mlecape', 'feature_lcl', 'feature_lfc', 'feature_el', 'feature_ccl']
    for var in vars_to_load_now:
        tfmo[var] = tfmo[var].compute()
    features_with_parents = np.sort(np.where(tfmo.feature_parent_cell_id.compute().data != -1)[0])
    for i, feature_idx in enumerate(features_with_parents):
        parent_track = tfmo.feature_parent_track_id.data[feature_idx]
        time_idx = tfmo.feature_time_index.data[feature_idx]

        # Handle seabreeze (mean if already set)
        this_feature_seabreeze = tfmo.feature_seabreeze.data[feature_idx]
        previously_set_seabreeze = track_seabreezes[parent_track, time_idx]
        if np.isnan(previously_set_seabreeze):
            track_seabreezes[parent_track, time_idx] = this_feature_seabreeze
        elif previously_set_seabreeze != this_feature_seabreeze:
            track_seabreezes[parent_track, time_idx] = np.nanmean([previously_set_seabreeze, this_feature_seabreeze])


        # Handle feature area (sum if already set)
        this_feature_area = tfmo.feature_area.data[feature_idx]
        previously_set_area = track_area[parent_track, time_idx]
        if np.isnan(previously_set_area):
            track_area[parent_track, time_idx] = this_feature_area
        elif previously_set_area != this_feature_area:
            track_area[parent_track, time_idx] = previously_set_area + this_feature_area


        # Handle echo top (max if already set)
        this_feature_echo_top = tfmo.feature_echotop.data[feature_idx]
        previously_set_echo_top = track_echo_top[parent_track, time_idx]
        if np.isnan(previously_set_echo_top):
            track_echo_top[parent_track, time_idx] = this_feature_echo_top
        elif previously_set_echo_top != this_feature_echo_top:
            track_echo_top[parent_track, time_idx] = np.nanmax([previously_set_echo_top, this_feature_echo_top])


        # Handle feature flash count (sum if already set)
        this_feature_flash_count = tfmo.feature_flash_count.data[feature_idx]
        previously_set_flash_count = track_flash_count[parent_track, time_idx]
        if np.isnan(previously_set_flash_count):
            track_flash_count[parent_track, time_idx] = this_feature_flash_count
        elif previously_set_flash_count != this_feature_flash_count:
            track_flash_count[parent_track, time_idx] = np.nansum([previously_set_flash_count, this_feature_flash_count])


        # Handle feature large flash count (sum if already set)
        this_feature_large_flash_count = tfmo.feature_flash_count_area_GT_4km.data[feature_idx]
        previously_set_large_flash_count = track_large_flash_count[parent_track, time_idx]
        if np.isnan(previously_set_large_flash_count):
            track_large_flash_count[parent_track, time_idx] = this_feature_large_flash_count
        elif previously_set_large_flash_count != this_feature_large_flash_count:
            track_large_flash_count[parent_track, time_idx] = np.nansum([previously_set_large_flash_count, this_feature_large_flash_count])


        # Handle feature small flash count (sum if already set)
        this_feature_small_flash_count = tfmo.feature_flash_count_area_LE_4km.data[feature_idx]
        previously_set_small_flash_count = track_small_flash_count[parent_track, time_idx]
        if np.isnan(previously_set_small_flash_count):
            track_small_flash_count[parent_track, time_idx] = this_feature_small_flash_count
        elif previously_set_small_flash_count != this_feature_small_flash_count:
            track_small_flash_count[parent_track, time_idx] = np.nansum([previously_set_small_flash_count, this_feature_small_flash_count])


        # Handle KDP volume (sum if already set)
        this_feature_KDP_volume = tfmo.feature_kdpvol.data[feature_idx]
        previously_set_KDP_volume = track_KDP_volume[parent_track, time_idx]
        if np.isnan(previously_set_KDP_volume):
            track_KDP_volume[parent_track, time_idx] = this_feature_KDP_volume
        elif previously_set_KDP_volume != this_feature_KDP_volume:
            track_KDP_volume[parent_track, time_idx] = np.nansum([previously_set_KDP_volume, this_feature_KDP_volume])


        # Handle lat center (mean if already set)
        this_feature_lat_ctr = tfmo.feature_lat.data[feature_idx]
        previously_set_lat_ctr = track_lat_ctr[parent_track, time_idx]
        if np.isnan(previously_set_lat_ctr):
            track_lat_ctr[parent_track, time_idx] = this_feature_lat_ctr
        elif previously_set_lat_ctr != this_feature_lat_ctr:
            track_lat_ctr[parent_track, time_idx] = np.nanmean([previously_set_lat_ctr, this_feature_lat_ctr])


        # Handle lon center (mean if already set)
        this_feature_lon_ctr = tfmo.feature_lon.data[feature_idx]
        previously_set_lon_ctr = track_lon_ctr[parent_track, time_idx]
        if np.isnan(previously_set_lon_ctr):
            track_lon_ctr[parent_track, time_idx] = this_feature_lon_ctr
        elif previously_set_lon_ctr != this_feature_lon_ctr:
            track_lon_ctr[parent_track, time_idx] = np.nanmean([previously_set_lon_ctr, this_feature_lon_ctr])


        # Handle rhoHV deficit volume (sum if already set)
        this_feature_rhoHV_volume = tfmo.feature_rhvdeficitvol.data[feature_idx]
        previously_set_rhoHV_volume = track_rhoHV_volume[parent_track, time_idx]
        if np.isnan(previously_set_rhoHV_volume):
            track_rhoHV_volume[parent_track, time_idx] = this_feature_rhoHV_volume
        elif previously_set_rhoHV_volume != this_feature_rhoHV_volume:
            track_rhoHV_volume[parent_track, time_idx] = np.nansum([previously_set_rhoHV_volume, this_feature_rhoHV_volume])


        # Handle ZDR volume (sum if already set)
        this_feature_ZDR_volume = tfmo.feature_zdrvol.data[feature_idx]
        previously_set_ZDR_volume = track_ZDR_volume[parent_track, time_idx]
        if np.isnan(previously_set_ZDR_volume):
            track_ZDR_volume[parent_track, time_idx] = this_feature_ZDR_volume
        elif previously_set_ZDR_volume != this_feature_ZDR_volume:
            track_ZDR_volume[parent_track, time_idx] = np.nansum([previously_set_ZDR_volume, this_feature_ZDR_volume])



        # Handle minL2-MCMIPC (cloud top temperature) (min if already set)
        this_feature_min_L2_MCMIPC = tfmo.feature_min_L2_MCMIPC.data[feature_idx]
        previously_set_min_L2_MCMIPC = track_min_L2_MCMIPC[parent_track, time_idx]
        if np.isnan(previously_set_min_L2_MCMIPC):
            track_min_L2_MCMIPC[parent_track, time_idx] = this_feature_min_L2_MCMIPC
        elif previously_set_min_L2_MCMIPC != this_feature_min_L2_MCMIPC:
            track_min_L2_MCMIPC[parent_track, time_idx] = np.nanmin([previously_set_min_L2_MCMIPC, this_feature_min_L2_MCMIPC])


        # Handle pressure profile (mean if already set)
        this_feature_pressure_profile = tfmo.feature_pressure_profile.data[feature_idx, :]
        previously_set_pressure_profile = track_pressure_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_pressure_profile)):
            track_pressure_profile[parent_track, time_idx, :] = this_feature_pressure_profile
        elif not np.all(previously_set_pressure_profile == this_feature_pressure_profile):
            track_pressure_profile[parent_track, time_idx, :] = np.nanmean([previously_set_pressure_profile, this_feature_pressure_profile], axis=0)


        # Handle height profile (mean if already set)
        this_feature_height_profile = tfmo.feature_msl_profile.data[feature_idx, :]
        previously_set_height_profile = track_height_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_height_profile)):
            track_height_profile[parent_track, time_idx, :] = this_feature_height_profile
        elif not np.all(previously_set_height_profile == this_feature_height_profile):
            track_height_profile[parent_track, time_idx, :] = np.nanmean([previously_set_height_profile, this_feature_height_profile], axis=0)


        # Handle temperature profile (mean if already set)
        this_feature_temperature_profile = tfmo.feature_temp_profile.data[feature_idx, :]
        previously_set_temperature_profile = track_temperature_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_temperature_profile)):
            track_temperature_profile[parent_track, time_idx, :] = this_feature_temperature_profile
        elif not np.all(previously_set_temperature_profile == this_feature_temperature_profile):
            track_temperature_profile[parent_track, time_idx, :] = np.nanmean([previously_set_temperature_profile, this_feature_temperature_profile], axis=0)


        # Handle dewpoint profile (mean if already set)
        this_feature_dewpoint_profile = tfmo.feature_dew_profile.data[feature_idx, :]
        previously_set_dewpoint_profile = track_dewpoint_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_dewpoint_profile)):
            track_dewpoint_profile[parent_track, time_idx, :] = this_feature_dewpoint_profile
        elif not np.all(previously_set_dewpoint_profile == this_feature_dewpoint_profile):
            track_dewpoint_profile[parent_track, time_idx, :] = np.nanmean([previously_set_dewpoint_profile, this_feature_dewpoint_profile], axis=0)


        # Handle u profile (mean if already set)
        this_feature_u_profile = tfmo.feature_u_profile.data[feature_idx, :]
        previously_set_u_profile = track_u_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_u_profile)):
            track_u_profile[parent_track, time_idx, :] = this_feature_u_profile
        elif not np.all(previously_set_u_profile == this_feature_u_profile):
            track_u_profile[parent_track, time_idx, :] = np.nanmean([previously_set_u_profile, this_feature_u_profile], axis=0)


        # Handle v profile (mean if already set)
        this_feature_v_profile = tfmo.feature_v_profile.data[feature_idx, :]
        previously_set_v_profile = track_v_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_v_profile)):
            track_v_profile[parent_track, time_idx, :] = this_feature_v_profile
        elif not np.all(previously_set_v_profile == this_feature_v_profile):
            track_v_profile[parent_track, time_idx, :] = np.nanmean([previously_set_v_profile, this_feature_v_profile], axis=0)


        # Handle ccn profile (mean if already set)
        this_feature_ccn_profile = tfmo['feature_ccn_profile_0.6'].data[feature_idx, :]
        previously_set_ccn_profile = track_ccn_profile[parent_track, time_idx, :]
        if np.all(np.isnan(previously_set_ccn_profile)):
            track_ccn_profile[parent_track, time_idx, :] = this_feature_ccn_profile
        elif not np.all(previously_set_ccn_profile == this_feature_ccn_profile):
            track_ccn_profile[parent_track, time_idx, :] = np.nanmean([previously_set_ccn_profile, this_feature_ccn_profile], axis=0)


        # Handle mlcape (mean if already set)
        this_feature_mlcape = tfmo.feature_mlcape.data[feature_idx]
        previously_set_mlcape = track_mlcape[parent_track, time_idx]
        if np.isnan(previously_set_mlcape):
            track_mlcape[parent_track, time_idx] = this_feature_mlcape
        elif previously_set_mlcape != this_feature_mlcape:
            track_mlcape[parent_track, time_idx] = np.nanmean([previously_set_mlcape, this_feature_mlcape])
        
        # Handle mlcin (mean if already set)
        this_feature_mlcin = tfmo.feature_mlcin.data[feature_idx]
        previously_set_mlcin = track_mlcin[parent_track, time_idx]
        if np.isnan(previously_set_mlcin):
            track_mlcin[parent_track, time_idx] = this_feature_mlcin
        elif previously_set_mlcin != this_feature_mlcin:
            track_mlcin[parent_track, time_idx] = np.nanmean([previously_set_mlcin, this_feature_mlcin])

        # Handle mlecape (mean if already set)
        this_feature_mlecape = tfmo.feature_mlecape.data[feature_idx]
        previously_set_mlecape = track_mlecape[parent_track, time_idx]
        if np.isnan(previously_set_mlecape):
            track_mlecape[parent_track, time_idx] = this_feature_mlecape
        elif previously_set_mlecape != this_feature_mlecape:
            track_mlecape[parent_track, time_idx] = np.nanmean([previously_set_mlecape, this_feature_mlecape])

        # Handle lcl (min if already set)
        this_feature_lcl = tfmo.feature_lcl.data[feature_idx]
        previously_set_lcl = track_lcl[parent_track, time_idx]
        if np.isnan(previously_set_lcl):
            track_lcl[parent_track, time_idx] = this_feature_lcl
        elif previously_set_lcl != this_feature_lcl:
            track_lcl[parent_track, time_idx] = np.nanmin([previously_set_lcl, this_feature_lcl])
        
        # Handle lfc (min if already set)
        this_feature_lfc = tfmo.feature_lfc.data[feature_idx]
        previously_set_lfc = track_lfc[parent_track, time_idx]
        if np.isnan(previously_set_lfc):
            track_lfc[parent_track, time_idx] = this_feature_lfc
        elif previously_set_lfc != this_feature_lfc:
            track_lfc[parent_track, time_idx] = np.nanmin([previously_set_lfc, this_feature_lfc])
        
        # Handle el (max if already set)
        this_feature_el = tfmo.feature_el.data[feature_idx]
        previously_set_el = track_el[parent_track, time_idx]
        if np.isnan(previously_set_el):
            track_el[parent_track, time_idx] = this_feature_el
        elif previously_set_el != this_feature_el:
            track_el[parent_track, time_idx] = np.nanmax([previously_set_el, this_feature_el])
        
        # Handle ccl (min if already set)
        this_feature_ccl = tfmo.feature_ccl.data[feature_idx]
        previously_set_ccl = track_ccl[parent_track, time_idx]
        if np.isnan(previously_set_ccl):
            track_ccl[parent_track, time_idx] = this_feature_ccl
        elif previously_set_ccl != this_feature_ccl:
            track_ccl[parent_track, time_idx] = np.nanmin([previously_set_ccl, this_feature_ccl])


    tfmo = tfmo.assign({
        'track_seabreeze' : (('track', 'time'), track_seabreezes),
        'track_area' : (('track', 'time'), track_area),
        'track_echo_top' : (('track', 'time'), track_echo_top),
        'track_flash_count' : (('track', 'time'), track_flash_count),
        'track_flash_count_area_GT_4km' : (('track', 'time'), track_large_flash_count),
        'track_flash_count_area_LE_4km' : (('track', 'time'), track_small_flash_count),
        'track_kdpvol' : (('track', 'time'), track_KDP_volume),
        'track_lat' : (('track', 'time'), track_lat_ctr),
        'track_lon' : (('track', 'time'), track_lon_ctr),
        'track_rhvdeficitvol' : (('track', 'time'), track_rhoHV_volume),
        'track_zdrvol' : (('track', 'time'), track_ZDR_volume),
        'track_min_L2_MCMIPC' : (('track', 'time'), track_min_L2_MCMIPC),
        'track_pressure_profile' : (('track', 'time', 'vertical_levels'), track_pressure_profile),
        'track_msl_profile' : (('track', 'time', 'vertical_levels'), track_height_profile),
        'track_temp_profile' : (('track', 'time', 'vertical_levels'), track_temperature_profile),
        'track_dew_profile' : (('track', 'time', 'vertical_levels'), track_dewpoint_profile),
        'track_u_profile' : (('track', 'time', 'vertical_levels'), track_u_profile),
        'track_v_profile' : (('track', 'time', 'vertical_levels'), track_v_profile),
        'track_ccn_profile_0.6' : (('track', 'time', 'vertical_levels'), track_ccn_profile),
        'track_mlcape' : (('track', 'time'), track_mlcape),
        'track_mlcin' : (('track', 'time'), track_mlcin),
        'track_mlecape' : (('track', 'time'), track_mlecape),
        'track_lcl' : (('track', 'time'), track_lcl),
        'track_lfc' : (('track', 'time'), track_lfc),
        'track_el' : (('track', 'time'), track_el),
        'track_ccl' : (('track', 'time'), track_ccl)
    })
    return tfmo


if __name__ == '__main__':
    if USE_DASK:
        client = Client('tcp://127.0.0.1:8786')
    else:
        client = None
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    should_debug = ((len(sys.argv) > 2) and (sys.argv[2] == '--debug'))
    if should_debug:
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/coords').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/tobac_ts').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/eet').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/ctt').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/seabreeze').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/profiles').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/sfc').mkdir(parents=True, exist_ok=True)
        pth(f'./debug-figs-{date_i_want.strftime("%Y%m%d")}/aerosols').mkdir(parents=True, exist_ok=True)
    tfm_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/seabreeze.zarr'
    tfm = xr.open_dataset(tfm_path, engine='zarr')
    tfm_coord = apply_coord_transforms(tfm, should_debug=should_debug)
    if not path.exists(f'/Volumes/LtgSSD/nexrad_zarr/{date_i_want.strftime('%B').upper()}/{date_i_want.strftime('%Y%m%d')}'):
        print('I don\'t have EET for this day, computing it')
        add_eet_to_radar_data(date_i_want, client)
    tfm_eet = add_eet_to_tobac_data(tfm_coord, date_i_want, client, should_debug=should_debug)
    tfm_ts = add_timeseries_data_to_toabc_path(tfm_eet, date_i_want, client=client, should_debug=should_debug)
    print('Adding satellite data to tobac data')
    tfm_ctt = add_goes_data_to_tobac_path(tfm_ts, client, should_debug=should_debug)


    tfm_seabreeze = add_seabreeze_to_features(tfm_ctt, client, should_debug=should_debug)
    tfm_w_profiles = add_radiosonde_data(tfm_seabreeze, should_debug=should_debug)
    tfm_w_sfc = add_madis_data(tfm_w_profiles, should_debug=should_debug, client=client)
    tfm_w_aerosols = add_sfc_aerosol_data(tfm_w_sfc)
    # Compute aircraft aerosol passes here
    tfm_sounding_stats = compute_sounding_stats(tfm_w_aerosols)
    tfm_w_parents = generate_seg_mask_cell_track(generate_seg_mask_cell_track(tfm_sounding_stats, convert_to='track'), convert_to='cell')
    below_cloud_processing(tfm_w_parents, date_i_want)
    print('Converting to track time')
    tfm_obs = convert_to_track_time(tfm_w_parents)
    final_out_path = tfm_path.replace('.zarr', '-obs.zarr')
    tfm_obs = tfm_obs.drop_vars(['feature_time_str'], errors='ignore')
    client.close()
    try:
        for dv in tfm_obs.data_vars:
            if 'chunks' in tfm_obs[dv].encoding.keys():
                del tfm_obs[dv].encoding['chunks']
        if path.exists(final_out_path):
            rmtree(final_out_path)
        tfm_obs.chunk('auto').to_zarr(final_out_path)
    except TypeError:
        rmtree(final_out_path)
        tfm_obs.to_zarr(final_out_path, zarr_format=2)
