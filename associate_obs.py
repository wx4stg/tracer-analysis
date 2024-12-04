#!/usr/bin/env python


import xarray as xr
import pandas as pd
from datetime import datetime as dt
from os import path, listdir
from glob import glob
import numpy as np
from metpy.interpolate import interpolate_1d
from scipy.interpolate import interp1d
import sys
import warnings
import geopandas as gpd
from matplotlib.path import Path
from numba import njit


@njit
def identify_side_jit(dts, lons, lats, tfm_times, seabreeze, grid_lon, grid_lat):
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


def identify_side(dts, lons, lats, tfm):
    seabreezes = []
    for lon, lat, this_dt in zip(lons, lats, dts):
        tfm_time = tfm.sel(time=this_dt, method='nearest')
        distance = (((tfm_time.lon - lon)**2 + (tfm_time.lat - lat)**2)**0.5)
        dist_idx = np.unravel_index(distance.argmin().compute(), distance.shape)
        closest_seabreeze = tfm_time.seabreeze.transpose(*tfm_time.lat.dims).data[dist_idx].compute()
        seabreezes.append(closest_seabreeze)
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

    arm_sonde_sbf_side = identify_side(arm_sonde_dts_this_day, arm_sonde_lons, arm_sonde_lats, tfm)
    # arm_sonde_sbf_side = identify_side_oneatatime(arm_sonde_dts_this_day, arm_sonde_lons, arm_sonde_lats, tfm.time.data.compute(),
    #                                               tfm.seabreeze.transpose(*tfm.lat.dims).data.compute(), tfm.lon.data.compute(), tfm.lat.data.compute())

    # Load the TAMU sondes
    tamu_sonde_path = '/Volumes/LtgSSD/TAMU_SONDES/'
    tamu_sonde_files = sorted(listdir(tamu_sonde_path))
    tamu_sonde_dts = np.array([dt.strptime('_'.join(f.split('_')[2:4]), '%Y%m%d_%H%M') for f in tamu_sonde_files]).astype('datetime64[s]')
    tamu_sonde_files = np.array([path.join(tamu_sonde_path, f) for f in tamu_sonde_files])
    tamu_day_filter = np.where((tamu_sonde_dts >= time_start_this_day) & (tamu_sonde_dts <= time_end_this_day))[0]
    tamu_sonde_files_this_day = tamu_sonde_files[tamu_day_filter]
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

    tamu_sonde_sbf_side = identify_side(tamu_sonde_dts_this_day, tamu_sonde_lons, tamu_sonde_lats, tfm)
    # tamu_sonde_sbf_side = identify_side_oneatatime(tamu_sonde_dts_this_day, tamu_sonde_lons, tamu_sonde_lats, tfm.time.data.compute(),
    #                                             tfm.seabreeze.transpose(*tfm.lat.dims).data.compute(), tfm.lon.data.compute(), tfm.lat.data.compute())

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
                last_maritime_profile_time_index = closest_time_index
                maritime_representative_profile[0:closest_time_index, :, :] = this_rep_profile
        elif sbf == -2:
            # This is a continental sounding
            continental_representative_profile[closest_time_index, :, :] = this_rep_profile
            if last_continental_profile_time_index != -1:
                continental_representative_profile[last_continental_profile_time_index+1:closest_time_index, :, :] = interp_sounding_times(tfm.time.data, last_continental_profile_time_index, closest_time_index, continental_representative_profile)
            else:
                last_continental_profile_time_index = closest_time_index
                continental_representative_profile[0:closest_time_index, :, :] = this_rep_profile
        else:
            raise NotImplementedError('Only maritime and continental sides are supported')
        
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

def add_sfc_aerosol_data(tfm):
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
        arm_ccn_aerosol = arm_ccn.aerosol_number_concentration
        arm_ccn_time = arm_ccn.time.data
        readings_in_window = ((arm_ccn.supersaturation_calculated >= 0.35) & (arm_ccn.supersaturation_calculated <= 0.55))
        arm_ccn_ccn_window = arm_ccn_ccn[readings_in_window]
        arm_ccn_time_window = arm_ccn_time[readings_in_window]
        arm_ccn_lon = np.full(arm_ccn_time_window.shape, arm_ccn.lon.data)
        arm_ccn_lat = np.full(arm_ccn_time_window.shape, arm_ccn.lat.data)
        arm_ccn_sbf = np.array(identify_side(arm_ccn_time_window, arm_ccn_lon, arm_ccn_lat, tfm))
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
        tamu_ccn = tamu_ccn[tamu_ccn['SS'] == 0.4]
        tamu_times_window = pd.to_datetime(tamu_ccn.loc[:, 'Time'], format='%y%m%d %H:%M:%S').values
        tamu_ccn_window = tamu_ccn.loc[:, 'N_CCN'].values
        tamu_ccn_lon_window = tamu_ccn.loc[:, 'Longitude'].values
        tamu_ccn_lat_window = tamu_ccn.loc[:, 'Latitude'].values
        tamu_ccn_sbf = identify_side(tamu_times_window, tamu_ccn_lon_window, tamu_ccn_lat_window, tfm)
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


if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    tfm_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/seabreeze.zarr'
    tfm = xr.open_dataset(tfm_path, engine='zarr', chunks='auto')
    tfm_w_profiles = add_radiosonde_data(tfm)
    tfm_w_sfc = add_madis_data(tfm_w_profiles)
    tfm_w_aerosols = add_sfc_aerosol_data(tfm_w_sfc)
    tfm_w_aerosols.to_zarr(tfm_path.replace('.zarr', '-obs.zarr'))