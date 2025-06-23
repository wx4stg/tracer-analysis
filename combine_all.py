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
    client = Client('tcp://127.0.0.1:8786')
    all_res = sorted(glob('/Users/stgardner4/Desktop/tobac_saves_new/tobac_Save*/seabreeze-obs.zarr'))
    track_offset = 0
    out_path = '/Users/stgardner4/Desktop/tobac_saves_new/all_tracks.zarr'
    unified_time = None  # Placeholder for unified time dimension

    if path.exists(out_path):
        rmtree(out_path)

    for ds_path in all_res:
        print(f'Timing {ds_path}')
        ds = xr.open_dataset(ds_path, engine='zarr', decode_timedelta=True, chunks='auto')
        ds = drop_nontrack(ds)

        # Update unified_time with unique values from the current dataset
        if unified_time is None:
            unified_time = ds.time.values
        else:
            unified_time = np.sort(np.unique(np.concatenate([unified_time, ds.time.values])))
        ds.close()

    for ds_path in all_res:
        print(f'Processing {ds_path}')
        ds = xr.open_dataset(ds_path, engine='zarr', decode_timedelta=True, chunks='auto')
        ds = drop_nontrack(ds)

        # Align the time dimension
        ds = ds.reindex({'time': unified_time}, fill_value=np.nan)

        # Offset track indices
        ds = ds.assign_coords(track=ds.track + track_offset)
        track_offset += ds.dims['track']

        # Remove chunk encoding before saving and ensure compatibility
        for dv in ds.data_vars:
            if 'chunks' in ds[dv].encoding:
                del ds[dv].encoding['chunks']

        # Explicitly rechunk the dataset to ensure compatibility
        ds = ds.chunk({'track': 128, 'time': 128, 'vertical_levels': 2000})

        if path.exists(out_path):
            print('Appending to existing output path')
            ds.to_zarr(out_path, mode='a', append_dim='track', safe_chunks=True)  # Enable safe_chunks to avoid corruption
        else:
            print('Creating new output path')
            ds.to_zarr(out_path)