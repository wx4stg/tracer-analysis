import xarray as xr
from glob import glob
from shutil import rmtree
import numpy as np
import warnings
from os import path
from shutil import rmtree
warnings.filterwarnings('ignore')

def drop_nontrack(ds):
    # Only keep variables with 'track' in their dims
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

    for ds_path in sbf_obs_paths:
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
        all_tracks = xr.concat([all_tracks, this_track], dim='track')
        this_save_path = ds_path.replace('seabreeze-obs.zarr', 'all_tracks.zarr')
        all_tracks.to_zarr(this_save_path)
        ds_vars = list(all_tracks.data_vars)
        ds.close()
        saved_paths.append(this_save_path)
        return saved_paths, ds_vars


def combine_data_vars(track_dataset_paths, vars_to_combine):
    pass

if __name__ == '__main__':
    all_res = sorted(glob('/Users/stgardner4/Desktop/tobac_saves_new/tobac_Save*/seabreeze-obs.zarr'))
    pruned_paths, vars_to_proc = prune_unnecessary_times(all_res)
    print(f'Pruned paths: {pruned_paths}')