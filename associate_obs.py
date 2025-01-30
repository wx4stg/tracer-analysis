#!/usr/bin/env python


import xarray as xr
import pandas as pd
from datetime import datetime as dt
from os import path, listdir
from glob import glob
import numpy as np
from metpy.interpolate import interpolate_1d
from metpy.units import units
from metpy import calc as mpcalc
from ecape.calc import calc_ecape
from scipy.interpolate import interp1d
import sys
import warnings
import geopandas as gpd
from matplotlib.path import Path
from numba import njit

from glmtools.io.lightning_ellipse import lightning_ellipse_rev
from pyxlma import coords

from track_features_merges_augmented import add_eet_to_radar_data, add_eet_to_tobac_data, add_goes_data_to_tobac_path

USE_DASK = True
if USE_DASK:
    from dask.distributed import Client


@njit
def identify_side(dts, lons, lats, tfm_times, seabreeze, grid_lon, grid_lat):
    seabreezes = np.zeros(lons.shape)
    for i in np.arange(seabreezes.shape[0]):
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


def add_seabreeze_to_features(tfm):
    feature_seabreezes = identify_side(tfm.feature_time.values.astype('datetime64[s]').astype(float), tfm.feature_lon.values, tfm.feature_lat.values, tfm.time.values.astype('datetime64[s]').astype(float), 
                                    tfm.seabreeze.transpose('time', *tfm.lat.dims).compute().data, tfm.lon.values, tfm.lat.values)
    tfm = tfm.assign({
        'feature_seabreeze' : (('feature',), feature_seabreezes)
    })
    return tfm


def add_radiosonde_data(tfm, n_sounding_levels=2000):
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

    all_sonde_files = np.concatenate([arm_sonde_files_this_day, tamu_sonde_files_this_day])
    all_sonde_dts = np.concatenate([arm_sonde_dts_this_day, tamu_sonde_dts_this_day])
    all_sonde_sbf_side = np.concatenate([arm_sonde_sbf_side, tamu_sonde_sbf_side])

    maritime_sonde_dts = all_sonde_dts[all_sonde_sbf_side == -1]
    maritime_sorting = np.argsort(maritime_sonde_dts)
    maritime_sonde_dts = maritime_sonde_dts[maritime_sorting]

    continental_sonde_dts = all_sonde_dts[all_sonde_sbf_side == -2]
    continental_sorting = np.argsort(continental_sonde_dts)
    continental_sonde_dts = continental_sonde_dts[continental_sorting]

    n_sounding_vars = 6
    maritime_representative_profile = np.full((tfm.time.shape[0], n_sounding_levels, n_sounding_vars), -999, dtype=float)
    last_maritime_profile_time_index = -1

    continental_representative_profile = np.full((tfm.time.shape[0], n_sounding_levels, n_sounding_vars), -999, dtype=float)
    last_continental_profile_time_index = -1

    for f, this_dt, sbf in zip(all_sonde_files, all_sonde_dts, all_sonde_sbf_side):
        if f.endswith('.cdf'):
            this_sonde_data = xr.open_dataset(f)
        else:
            this_sonde_data = pd.read_csv('/Volumes/LtgSSD/TAMU_SONDES/TAMU_TRACER_20220602_2028_95.93W_30.07N_TSPOTINT.txt', skiprows=28, encoding='latin1', sep='\\s+', names=[
                'FlightTime', 'pres', 'tdry', 'RH', 'WindSpeed', 'WindDirection', 'AGL', 'AGL2', 'alt', 'Longitude', 'Latitude', 'y', 'x', 'Tv', 'dp', 'rho',
                'e', 'v_wind', 'u_wind', 'range', 'rv', 'MSL2', 'UTC_DAY', 'UTC_TIME', 'UTC_AMPM', 'ELAPSED_TIME', 'ELAPSED_TIME2', 'ELAPSED_TIME3', 'FrostPoint']
                )
        if len(this_sonde_data.pres.values) < 2:
            continue
        new_pres = np.linspace(np.max(this_sonde_data.pres.values), np.min(this_sonde_data.pres.values), n_sounding_levels)
        new_t, new_dp, new_u, new_v, new_z = interpolate_1d(new_pres, this_sonde_data.pres.values, this_sonde_data.tdry.values,
                                this_sonde_data.dp.values, this_sonde_data.u_wind.values, this_sonde_data.v_wind.values,
                                this_sonde_data.alt.values)
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


def add_madis_data(tfm):
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
    maritime_temp, maritime_dew, continental_temp, continental_dew = identify_madis(tfm.time.data.astype('datetime64[s]'), madis_ds_temp, madis_ds_dew,
               madis_ds_time.astype('datetime64[s]'), madis_ds_lat, madis_ds_lon, polyline)
    tfm_w_sfc = tfm.copy()
    tfm_w_sfc.maritime_dewpoint_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(maritime_dew)] = maritime_dew[~np.isnan(maritime_dew)] - 273.15
    tfm_w_sfc.maritime_temperature_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(maritime_temp)] = maritime_temp[~np.isnan(maritime_temp)] - 273.15
    tfm_w_sfc.continental_dewpoint_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(continental_dew)] = continental_dew[~np.isnan(continental_dew)] - 273.15
    tfm_w_sfc.continental_temperature_profile.transpose('vertical_levels', 'time').data[0, :][~np.isnan(continental_temp)] = continental_temp[~np.isnan(continental_temp)] - 273.15
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

    for sidenum, side in enumerate(['continental', 'maritime']):
        sidenum -= 2
        temp = tfm[f'{side}_temperature_profile'].data * units.degC
        dew = tfm[f'{side}_dewpoint_profile'].data * units.degC
        pressure = tfm[f'{side}_pressure_profile'].data * units.hPa
        height = tfm[f'{side}_msl_profile'].data * units.m
        u = tfm[f'{side}_u_profile'].data * (units.m/units.s)
        v = tfm[f'{side}_v_profile'].data * (units.m/units.s)
        ccn = tfm[f'{side}_ccn_profile'].data

        mlcapes = np.zeros(tfm.time.shape[0])
        mlcins = np.zeros(tfm.time.shape[0])
        mlecapes = np.zeros(tfm.time.shape[0])
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
                continue
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pressure_i, temp_i, dew_i)
            spc_hum = mpcalc.specific_humidity_from_dewpoint(pressure_i, dew_i)
            mlecape = calc_ecape(height_i, pressure_i, temp_i, spc_hum, u_i, v_i, cape_type='mixed_layer')

            mlcapes[i] = mlcape.magnitude
            mlcins[i] = mlcin.magnitude
            mlecapes[i] = mlecape.magnitude
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

    tfm_stats = tfm.copy()
    tfm_stats = tfm_stats.assign({
        'feature_pressure_profile' : (('feature', 'vertical_levels'), feature_pressure_profile),
        'feature_msl_profile' : (('feature', 'vertical_levels'), feature_msl_profile),
        'feature_temp_profile' : (('feature', 'vertical_levels'), feature_temp_profile),
        'feature_dew_profile' : (('feature', 'vertical_levels'), feature_dew_profile),
        'feature_u_profile' : (('feature', 'vertical_levels'), feature_u_profile),
        'feature_v_profile' : (('feature', 'vertical_levels'), feature_v_profile),
        'feature_ccn_profile' : (('feature', 'vertical_levels'), feature_ccn_profile),
        'feature_mlcape' : (('feature',), feature_mlcape),
        'feature_mlcin' : (('feature',), feature_mlcin),
        'feature_mlecape' : (('feature',), feature_mlecape)
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
        'maritime_ccn_profile' : (('time', 'vertical_levels'), maritime_ccn_vert),
        'continental_ccn_profile' : (('time', 'vertical_levels'), continental_ccn_vert)
    })

    return tfm_w_aerosols


def apply_coord_transforms(tfm):
    tfm = tfm.copy()
    radar_lat, radar_lon = 29.47, -95.08
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=radar_lat, ctrLon=radar_lon, ctrAlt=0)
    ltg_ell = lightning_ellipse_rev[1]
    
    x2d, y2d = np.meshgrid(tfm.x.data, tfm.y.data)
    grid_ecef_coords = tpcs.toECEF(x2d.flatten(), y2d.flatten(), np.zeros_like(x2d).flatten())
    
    geosys = coords.GeographicSystem()
    grid_lon, grid_lat, _ = geosys.fromECEF(*grid_ecef_coords)
    grid_lon = grid_lon.reshape(x2d.shape)
    grid_lat = grid_lat.reshape(x2d.shape)
    tfm = tfm.assign({'lat' : (('x', 'y'), grid_lat), 'lon' : (('x', 'y'), grid_lon)})


    satsys = coords.GeostationaryFixedGridSystem(subsat_lon=-75.19999694824219, sweep_axis='x')#, ellipse=ltg_ell)
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
    return tfm


def add_timeseries_data_to_toabc_path(tobac_data, date_i_want):
    tobac_data = tobac_data.copy()
    tobac_save_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime('%Y%m%d')}/'
    for f in listdir(tobac_save_path):
        if f.startswith('timeseries_data_melt') and f.endswith('.nc'):
            tobac_timeseries_path = path.join(tobac_save_path, f)
            break
    else:
        raise ValueError('>>>>>>>Unable to find timeseries data...>>>>>>>')
    timeseries_data = xr.open_dataset(tobac_timeseries_path, chunks='auto')
    timeseries_data = timeseries_data.reindex(feature=tobac_data.feature.data, fill_value=np.nan)
    for dv in timeseries_data.data_vars:
        if dv not in tobac_data.data_vars:
            tobac_data[dv] = timeseries_data[dv].copy()
    return tobac_data


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

    tfmo['feature_parent_track_id'] = tfmo.feature_parent_track_id.compute().astype('int32')
    vars_to_load_now = ['feature_time_index', 'feature_seabreeze', 'feature_area', 'feature_echotop', 'feature_flash_count', 'feature_flash_count_area_GT_4km',
                        'feature_flash_count_area_LE_4km', 'feature_kdpvol', 'feature_lat', 'feature_lon', 'feature_rhvdeficitvol', 'feature_zdrvol', 'feature_min_L2_MCMIPC',
                        'feature_pressure_profile', 'feature_msl_profile', 'feature_temp_profile', 'feature_dew_profile', 'feature_u_profile', 'feature_v_profile', 'feature_ccn_profile',
                        'feature_mlcape', 'feature_mlcin', 'feature_mlecape']
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
        this_feature_ccn_profile = tfmo.feature_ccn_profile.data[feature_idx, :]
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
        'track_ccn_profile' : (('track', 'time', 'vertical_levels'), track_ccn_profile),
        'track_mlcape' : (('track', 'time'), track_mlcape),
        'track_mlcin' : (('track', 'time'), track_mlcin),
        'track_mlecape' : (('track', 'time'), track_mlecape)
    })
    return tfmo


if __name__ == '__main__':
    if USE_DASK:
        client = Client('tcp://127.0.0.1:8786')
    else:
        client = None
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    tfm_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/seabreeze.zarr'
    tfm = xr.open_dataset(tfm_path, engine='zarr')
    tfm_coord = apply_coord_transforms(tfm)
    tfm_ts = add_timeseries_data_to_toabc_path(tfm_coord, date_i_want)
    if not path.exists(f'/Volumes/LtgSSD/nexrad_zarr/{date_i_want.strftime('%B').upper()}/{date_i_want.strftime('%Y%m%d')}'):
        print('I don\'t have EET for this day, computing it')
        add_eet_to_radar_data(date_i_want, client)
    tfm_eet = add_eet_to_tobac_data(tfm_ts, date_i_want, client)
    print('Adding satellite data to tobac data')
    tfm_ctt = add_goes_data_to_tobac_path(tfm_eet, client)
    tfm_seabreeze = add_seabreeze_to_features(tfm_ctt)
    tfm_w_profiles = add_radiosonde_data(tfm_seabreeze)
    tfm_w_sfc = add_madis_data(tfm_w_profiles)
    tfm_w_aerosols = add_sfc_aerosol_data(tfm_w_sfc)
    tfm_sounding_stats = compute_sounding_stats(tfm_w_aerosols)
    tfm_w_parents = generate_seg_mask_cell_track(generate_seg_mask_cell_track(tfm_sounding_stats, convert_to='track'), convert_to='cell')
    print('Converting to track time')
    tfm_obs = convert_to_track_time(tfm_w_parents)
    tfm_obs.to_zarr(tfm_path.replace('.zarr', '-obs.zarr'))
