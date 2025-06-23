import xarray as xr
from glob import glob
from shutil import rmtree
import numpy as np
import warnings
from dask.distributed import Client
from datetime import datetime as dt, UTC
warnings.filterwarnings('ignore')

def drop_nontrack(ds):
    # Only keep variables with 'track' in their dims
    keep_vars = [v for v in ds.data_vars if 'track' in ds[v].dims]
    ds = ds[keep_vars]
    return ds


if __name__ == '__main__':
    start = dt.now(UTC)
    client = Client('tcp://127.0.0.1:8786')
    all_res = sorted(glob('/Users/stgardner4/Desktop/tobac_saves_new/tobac_Save*/seabreeze-obs.zarr'))
    datasets = []
    track_offset = 0

    for path in all_res:
        print(f'Processing {path}')
        ds = xr.open_dataset(path, engine='zarr', decode_timedelta=True, chunks='auto')
        ds = drop_nontrack(ds)
        # Offset track indices
        ds = ds.assign_coords(track=ds.track + track_offset)
        datasets.append(ds)
        track_offset += ds.dims['track']

    print('Concatenating!')
    print(track_offset)
    # Use faster concat options and ensure chunking
    ds_combined = xr.concat(
        datasets,
        dim='track',
        fill_value=np.nan
    )
    print((dt.now(UTC) - start).total_seconds())
    print('Chunking!')
    ds_combined = ds_combined.chunk({'track': 128, 'time': 128, 'vertical_levels': 1000})

    # Remove chunk encoding before saving
    for dv in ds_combined.data_vars:
        if 'chunks' in ds_combined[dv].encoding:
            del ds_combined[dv].encoding['chunks']

    out_path = '/Users/stgardner4/Desktop/tobac_saves_new/all_tracks.zarr'
    print('Writing!')
    try:
        ds_combined.to_zarr(out_path)
    except TypeError:
        print('Writing with zarr format 2')
        rmtree(out_path)
        ds_combined.to_zarr(out_path, zarr_format=2)