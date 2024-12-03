#!/usr/bin/env python


import xarray as xr
import pandas as pd
from datetime import datetime as dt
from os import path, listdir
import numpy as np
from pyxlma.coords import centers_to_edges
from metpy.interpolate import interpolate_1d
from scipy.interpolate import interp1d
import sys


def identify_side(dts, lons, lats, tfm):
    seabreezes = []
    for lon, lat, dt in zip(lons, lats, dts):
        tfm_time = tfm.sel(time=dt, method='nearest')
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


def add_radiosonde_data(tfm, date_i_want, n_sounding_levels=1000):
   

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

    all_sonde_files = np.concatenate([arm_sonde_files_this_day, tamu_sonde_files_this_day])
    all_sonde_dts = np.concatenate([arm_sonde_dts_this_day, tamu_sonde_dts_this_day])
    all_sonde_lons = np.concatenate([arm_sonde_lons, tamu_sonde_lons])
    all_sonde_lats = np.concatenate([arm_sonde_lats, tamu_sonde_lats])
    all_sonde_sbf_side = np.concatenate([arm_sonde_sbf_side, tamu_sonde_sbf_side])

    maritime_sonde_dts = all_sonde_dts[all_sonde_sbf_side == -1]
    maritime_sorting = np.argsort(maritime_sonde_dts)
    maritime_sonde_dts = maritime_sonde_dts[maritime_sorting]
    maritime_sonde_files = all_sonde_files[all_sonde_sbf_side == -1][maritime_sorting]
    maritime_sonde_lons = all_sonde_lons[all_sonde_sbf_side == -1][maritime_sorting]
    maritime_sonde_lats = all_sonde_lats[all_sonde_sbf_side == -1][maritime_sorting]

    continental_sonde_dts = all_sonde_dts[all_sonde_sbf_side == -2]
    continental_sorting = np.argsort(continental_sonde_dts)
    continental_sonde_dts = continental_sonde_dts[continental_sorting]
    continental_sonde_files = all_sonde_files[all_sonde_sbf_side == -2][continental_sorting]
    continental_sonde_lons = all_sonde_lons[all_sonde_sbf_side == -2][continental_sorting]
    continental_sonde_lats = all_sonde_lats[all_sonde_sbf_side == -2][continental_sorting]

    n_sounding_vars = 6
    maritime_representative_profile = np.full((tfm.time.shape[0], n_sounding_levels, n_sounding_vars), -999, dtype=float)
    last_maritime_profile = np.full((n_sounding_levels, n_sounding_vars), np.nan)
    last_maritime_profile_time_index = -1

    continental_representative_profile = np.full((tfm.time.shape[0], n_sounding_levels, n_sounding_vars), -999, dtype=float)
    last_continental_profile = np.full((n_sounding_levels), np.nan)
    last_continental_profile_time_index = -1

    for f, this_dt, lon, lat, sbf in zip(all_sonde_files, all_sonde_dts, all_sonde_lons, all_sonde_lats, all_sonde_sbf_side):
        if f.endswith('.cdf'):
            this_sonde_data = xr.open_dataset(f)
        else:
            this_sonde_data_TAMU = pd.read_csv('/Volumes/LtgSSD/TAMU_SONDES/TAMU_TRACER_20220602_2028_95.93W_30.07N_TSPOTINT.txt', skiprows=28, encoding='latin1', sep='\\s+', names=[
                'FlightTime', 'pres', 'tdry', 'RH', 'WindSpeed', 'WindDirection', 'AGL', 'AGL2', 'alt', 'Longitude', 'Latitude', 'y', 'x', 'Tv', 'dp', 'rho',
                'e', 'v_wind', 'u_wind', 'range', 'rv', 'MSL2', 'UTC_DAY', 'UTC_TIME', 'UTC_AMPM', 'ELAPSED_TIME', 'ELAPSED_TIME2', 'ELAPSED_TIME3', 'FrostPoint']
                )
        new_pres = np.linspace(np.max(this_sonde_data.pres.values), np.min(this_sonde_data.pres.values), 1000)
        new_t, new_dp, new_u, new_v, new_z = interpolate_1d(new_pres, this_sonde_data.pres.values, this_sonde_data.tdry.values,
                                this_sonde_data.dp.values, this_sonde_data.u_wind.values, this_sonde_data.v_wind.values,
                                this_sonde_data.alt.values)
        this_rep_profile = np.vstack([new_pres, new_t, new_dp, new_u, new_v, new_z]).T

        closest_time_index = np.argmin(np.abs(tfm.time.data - this_dt))
        closest_time = tfm.time.data[closest_time_index]
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



if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d')
    tfm_path = f'/Volumes/LtgSSD/tobac_saves/tobac_Save_{date_i_want.strftime("%Y%m%d")}/seabreeze.zarr'
    tfm = xr.open_dataset(tfm_path, engine='zarr', chunks='auto')
    tfm_w_profiles = add_radiosonde_data(tfm, date_i_want)
    tfm_w_profiles.to_zarr(tfm_path.replace('.zarr', '-obs.zarr'))