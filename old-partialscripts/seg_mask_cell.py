#!/usr/bin/python3
import numpy as np
import xarray as xr



if __name__ == '__main__':
    print('Reading')
    tobac_data = xr.open_dataset('/Volumes/LtgSSD/tobac_saves/tobac_Save_20220602/Track_features_merges_augmented.nc')
    tobac_data['segmentation_mask'] = xr.where(tobac_data.segmentation_mask == 0, np.nan, tobac_data.segmentation_mask.astype(np.float32))
    feature_ids = tobac_data.feature.data
    seg_data_feature = tobac_data.segmentation_mask.data
    cell_ids = tobac_data.feature_parent_cell_id.sel(feature=feature_ids).data
    feature_to_cell_map = dict(zip(feature_ids, cell_ids))
    seg_data_cell = seg_data_feature.copy()
    print('Mapping')
    seg_data_cell = np.vectorize(feature_to_cell_map.get)(seg_data_cell)
    print('Filtering')
    seg_data_cell[seg_data_cell == None] = np.nan
    seg_data_cell[seg_data_cell == 0] = np.nan
    seg_data_cell[seg_data_cell == -1] = np.nan
    print('Converting')
    seg_data_cell = seg_data_cell.astype(np.float32)
    print('Saving')
    tobac_data['segmentation_mask_cell'] = xr.DataArray(seg_data_cell, dims=('time', 'y', 'x'), coords={'time': tobac_data.time.data, 'y': tobac_data.y.data, 'x': tobac_data.x.data})
    comp = dict(zlib=True, complevel=5)
    enc = {var: comp for var in tobac_data.data_vars if not np.issubdtype(tobac_data[var].dtype, str)}
    print('Writing')
    tobac_data.to_netcdf('/Volumes/LtgSSD/tobac_saves/tobac_Save_20220602/Track_features_merges_augmented_cell.nc', encoding=enc)