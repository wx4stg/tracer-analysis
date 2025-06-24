import xarray as xr
from glob import glob
from shutil import rmtree
import numpy as np
import warnings
from dask.distributed import Client
from datetime import datetime as dt, UTC
from os import path
from shutil import rmtree
warnings.filterwarnings('ignore')

def drop_nontrack(ds):
    # Only keep variables with 'track' in their dims
    keep_vars = [v for v in ds.data_vars if 'track' in ds[v].dims]
    ds = ds[keep_vars]
    return ds


if __name__ == '__main__':
    # client = Client('tcp://127.0.0.1:8786')
    all_res = sorted(glob('/Users/stgardner4/Desktop/tobac_saves_new/tobac_Save*/seabreeze-obs.zarr'))
    track_offset = 0
    out_path = '/Users/stgardner4/Desktop/tobac_saves_new/all_tracks.zarr'
    unified_time = None  # Placeholder for unified time dimension
    unified_track = None

    if path.exists(out_path):
        rmtree(out_path)

    for ds_path in all_res:
        print(f'Timing {ds_path}')
        ds = xr.open_dataset(ds_path, engine='zarr', decode_timedelta=True)
        ds = drop_nontrack(ds)
        ds = ds.assign_coords(track=ds.track + track_offset)
        track_offset += ds.dims['track']

        # Update unified_time with unique values from the current dataset
        if unified_time is None:
            unified_time = ds.time.values
        else:
            unified_time = np.sort(np.unique(np.concatenate([unified_time, ds.time.values])))

        if unified_track is None:
            unified_track = ds.track.values
        else:
            unified_track = np.sort(np.unique(np.concatenate([unified_track, ds.track.values])))
        ds.close()

    track_offset = 0  # Reset track offset for the second pass
    for ds_path in all_res:
        print(f'Processing {ds_path}')
        ds = xr.open_dataset(ds_path, engine='zarr', decode_timedelta=True)
        ds = drop_nontrack(ds)
        ds = ds.assign_coords(track=ds.track + track_offset)
        track_offset += ds.dims['track']

        # Align the time dimension
        ds = ds.reindex({'time': unified_time, 'track': unified_track}, fill_value=np.nan)

        # Remove chunk encoding before saving and ensure compatibility
        for dv in ds.data_vars:
            if 'chunks' in ds[dv].encoding:
                del ds[dv].encoding['chunks']


        print(ds.time.shape)
        print(ds.track.shape)
        print(ds.vertical_levels.shape)
        # Explicitly rechunk the dataset to ensure compatibility
        ds = ds.chunk({'track': 12484*2, 'time': 6215*2, 'vertical_levels': 2000*2})

        if path.exists(out_path):
            print('Appending to existing output path')
            ds.to_zarr(out_path, mode='r+')
        else:
            print('Creating new output path')
            ds.to_zarr(out_path)