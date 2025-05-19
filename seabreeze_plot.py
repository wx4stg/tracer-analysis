#!/usr/bin/env python3

from matplotlib import pyplot as plt
import xarray as xr
from datetime import datetime as dt
from cartopy import crs as ccrs
from cartopy import feature as cfeat
import sys
from datetime import datetime as dt

if __name__ == '__main__':
    date_i_want = sys.argv[1]
    date_i_want = dt.strptime(date_i_want, '%Y-%m-%d').replace(hour=12)
    seabreeze = xr.open_dataset(f'./sam_sbf/{date_i_want.strftime('%Y-%m-%d')}_seabreeze.nc')
    
    i=0
    for time in seabreeze.time.data:
        if time.astype('datetime64[s]').astype(dt) < date_i_want:
            continue
        sbf = seabreeze.sel(time=time).seabreeze
        if sbf.data.max() == sbf.data.min():
            continue
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_extent([-97.75, -91.5, 26.5, 30])
        ax.pcolormesh(seabreeze.longitude, seabreeze.latitude, sbf, transform=ccrs.PlateCarree(), cmap='coolwarm_r', vmin=-2, vmax=-1)
        fig.suptitle(f'Subjectively Analyzed Seabreeze\n{time.astype('datetime64[s]').astype(dt).strftime('%Y-%m-%d %H:%M:%S')}')
        px = 1/plt.rcParams['figure.dpi']
        fig.tight_layout()
        fig.set_size_inches(600*px, 400*px)
        fig.savefig(f'./sam_sbf_img/{str(i+1).zfill(4)}.png')
        i += 1
        plt.close(fig)