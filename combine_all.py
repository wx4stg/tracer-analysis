import xarray as xr
from glob import glob
from shutil import rmtree
import numpy as np
import warnings
from os import path
from numba import njit, prange
warnings.filterwarnings('ignore')

def drop_nontrack(ds):
    keep_vars = [v for v in ds.data_vars if 'track' in ds[v].dims]
    ds = ds[keep_vars]
    return ds

def prune_unnecessary_times(sbf_obs_paths):
    saved_paths = []
    timestep_max = 0
    for ds_path in sbf_obs_paths:
        print(f'Timing {ds_path}')
        ds = xr.open_dataset(ds_path, engine='zarr', decode_timedelta=True)
        ds = drop_nontrack(ds)
        track_present = ~np.isnan(ds.track_seabreeze)
        timestep_max = np.max([timestep_max, track_present.sum(dim='time').data.max()])
        ds.close()

    track_offset = 0
    ds_vars = None
    for ds_path in sbf_obs_paths:
        this_save_path = ds_path.replace('seabreeze-obs.zarr', 'tracks.zarr')
        if path.exists(this_save_path):
            saved_paths.append(this_save_path)
            continue
        ds = xr.open_dataset(ds_path, engine='zarr')
        ds = drop_nontrack(ds)
        track_present = ~np.isnan(ds.track_seabreeze)
        timestep = np.arange(timestep_max)

        first_timestep = track_present.argmax(dim='time').data
        last_timestep = first_timestep + track_present.sum(dim='time').data
        last_timestep[last_timestep == -1] = 0
        first_and_last = np.array([first_timestep, last_timestep]).T

        all_tracks = []
        max_track_id = first_and_last.shape[0] - 1
        for track_id, (first_ts, last_ts) in enumerate(first_and_last):
            print(f'Processing {ds_path}: {100*(track_id/max_track_id):.2f}%')
            if first_ts == 0 and last_ts == 0:
                continue
            if first_ts == last_ts:
                continue
            this_track = ds.isel(track=track_id, time=slice(first_ts, last_ts))
            track_size = this_track.time.size
            time_data = this_track.time.data
            this_track = this_track.assign(
                timestep = ('time', np.arange(track_size)),
            ).swap_dims({'time': 'timestep'}).drop_vars('time')
            this_track = this_track.reindex(timestep=timestep, fill_value=np.nan)
            new_times = np.full(timestep.size, np.nan, dtype='datetime64[ns]')
            new_times[0:track_size] = time_data
            this_track = this_track.assign(
                time = ('timestep', new_times)
            )
            all_tracks.append(this_track)
        all_tracks = xr.concat(all_tracks, dim='track')
        all_tracks['track'] = all_tracks['track'] + track_offset
        track_offset += all_tracks['track'].size
        all_tracks.to_zarr(this_save_path)
        ds_vars = list(all_tracks.data_vars)
        ds.close()
        saved_paths.append(this_save_path)
    if ds_vars is None:
        ds = xr.open_dataset(sbf_obs_paths[0], engine='zarr')
        ds = drop_nontrack(ds)
        ds_vars = list(ds.data_vars)
    return saved_paths, ds_vars


def combine_data_vars(track_dataset_paths, vars_to_combine):
    out_path = '/Volumes/LtgSSD/tobac_saves/all_tracks.zarr'
    if path.exists(out_path):
        return
    print('Combining data variables across all tracks...')
    max_var_idx = len(vars_to_combine) - 1
    for i, dv in enumerate(sorted(vars_to_combine)):
        print(f'{100*(i/max_var_idx):.2f}%: {dv}', end='... ')
        all_ds = [xr.open_dataset(f, engine='zarr') for f in track_dataset_paths]
        all_da = [ds[dv].to_dataset() for ds in all_ds]
        print('Merging', end='... ')
        all_da_merged = xr.concat(all_da, dim='track')
        print('Saving', end='... ')
        all_da_merged.to_zarr(f'/Volumes/LtgSSD/tobac_saves/tmp.zarr', mode='a', group=dv)
        print('Ceaning up', end='... ')
        [ds.close() for ds in all_ds]
        del all_da
        del all_da_merged
        print('Done')
    print('Actually actually combining')
    group_paths = glob('/Volumes/LtgSSD/tobac_saves/tmp.zarr/*')
    group_paths.remove('/Volumes/LtgSSD/tobac_saves/tmp.zarr/zarr.json')
    ds = xr.open_mfdataset(group_paths, engine='zarr')
    ds['track'] = np.arange(ds.track.shape[0])
    ds.to_zarr(out_path)
    rmtree('/Volumes/LtgSSD/tobac_saves/tmp.zarr', ignore_errors=True)


def remove_dropouts(x, y):
    x_sort_args = np.argsort(x)
    x = x[x_sort_args]
    y = y[x_sort_args]
    nans = np.isnan(y)
    y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    return y[x_sort_args]


@njit(parallel=True)
def vectorized_interpolation(regular_time, tracks_time, tracks_var, array_to_fill):
    for i in prange(len(tracks_time)):
        x = tracks_time[i]
        y = tracks_var[i]
        interpd_var = np.interp(regular_time, x, y)
        array_to_fill[i, :] = interpd_var
    return array_to_fill


def interpolate_var_to_reg_time_base(v, regular_time_base, all_tracks):
    var_reg_time_base = np.full((all_tracks.track.size, regular_time_base.size), np.nan, dtype=np.float64)
    this_var = all_tracks[v].transpose('track', 'timestep').compute()
    orig_dtype = this_var.data.dtype
    this_var = this_var.astype(np.float64)
    tracks_time = []
    tracks_var = []
    for tid in all_tracks.track.data:
        this_track_time = all_tracks.track_time_relative.sel(track=tid)[all_tracks.track_present.sel(track=tid)].compute()
        if this_track_time.size == 0:
            continue
        this_track_var = this_var.sel(track=tid)[all_tracks.track_present.sel(track=tid)]
        if np.all(np.isnan(this_track_var)):
            continue
        this_track_var_no_dropouts = remove_dropouts(this_track_time.data, this_track_var.data)
        tracks_time.append(this_track_time.data)
        tracks_var.append(this_track_var_no_dropouts)
    var_reg_time_base = vectorized_interpolation(regular_time_base, tracks_time, tracks_var, var_reg_time_base)
    return var_reg_time_base.astype(orig_dtype)


def interpolate_tracks_to_time_base():
    print('Interpolating tracks to regular time base...')
    all_tracks = xr.open_dataset('/Volumes/LtgSSD/tobac_saves/all_tracks.zarr', engine='zarr')
    track_present = ~np.isnan(all_tracks.track_seabreeze)
    all_tracks['track_present'] = track_present
    track_first_time = all_tracks.time.isel(timestep=0)
    ttr = ((all_tracks.time - track_first_time).astype(np.float64)/1e9).compute()
    all_tracks['track_time_relative'] = ttr
    all_tracks['track_time_relative'].attrs['units'] = 'seconds since first time in track'
    max_track_duration = np.nanmax(all_tracks.track_time_relative)
    regular_time_base = np.arange(0, max_track_duration, 10)
    regular_time_dataset = xr.Dataset()
    regular_time_dataset.assign_coords(
        time=('time', regular_time_base),
        track=('track', np.arange(all_tracks.track.size))
    )
    for v in all_tracks.data_vars:
        if v == 'track_time_relative':
            continue
        if all_tracks[v].dims != ('track', 'timestep'):
            continue
        interpd_var = interpolate_var_to_reg_time_base(v, regular_time_base, all_tracks)
        regular_time_dataset = regular_time_dataset.assign(
            **{v: (('track', 'time'), interpd_var)}
        )
    regular_time_dataset.to_zarr('/Volumes/LtgSSD/tobac_saves/regular_time_tracks.zarr') 



if __name__ == '__main__':
    all_res = sorted(glob('/Volumes/LtgSSD/tobac_saves/tobac_Save*/seabreeze-obs.zarr'))
    pruned_paths, vars_to_proc = prune_unnecessary_times(all_res)
    combine_data_vars(pruned_paths, vars_to_proc)
    interpolate_tracks_to_time_base()