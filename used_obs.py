#!/usr/bin/env python3

from glob import glob
from os import path
from sys import argv
from datetime import datetime as dt
from pathlib import Path

from geopandas import read_file
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import xarray as xr



def parse_sounding_paths(sounding_paths, sounding_lons, sounding_lats, sounding_times, base_date):
    soundings_dict = {}
    for sounding_index, sounding_path in enumerate(sounding_paths):
        this_sounding_origin = None
        this_sounding_lon = sounding_lons[sounding_index]
        this_sounding_lat = sounding_lats[sounding_index]
        if 'CMAS-sondes' in sounding_path:
            this_sounding_origin = 'CMAS mobile'
        elif 'housondewnpnS3' in sounding_path or 'housondewnpnS4' in sounding_path:
            this_sounding_origin = f'ARM Damon, TX ({float(this_sounding_lat):.2f}, {float(this_sounding_lon):.2f})'
        elif 'housondewnpnM1' in sounding_path or 'housondewnpnS1' in sounding_path:
            this_sounding_origin = f'ARM La Porte, TX ({float(this_sounding_lat):.2f}, {float(this_sounding_lon):.2f})'
        elif sounding_path.endswith('.csv'):
            if path.basename(sounding_path).startswith('HOU'):
                this_sounding_origin = f'ACARS, HOU ({float(this_sounding_lat):.2f}, {float(this_sounding_lon):.2f})'
            elif path.basename(sounding_path).startswith('IAH'):
                this_sounding_origin = f'ACARS, IAH ({float(this_sounding_lat):.2f}, {float(this_sounding_lon):.2f})'
        elif 'TAMU_SONDES' in sounding_path:
            this_sounding_origin = 'Texas A&M mobile'
        elif 'artificial' in sounding_path:
            continue
        else:
            raise ValueError(f'Unknown sounding origin for {sounding_path}!')
        this_sounding_time = sounding_times[sounding_index]
        this_sounding_dt = dt.strptime(base_date.strftime('%Y-%m-%d_')+this_sounding_time, '%Y-%m-%d_%H:%M:%S')
        fancy_string = f'{this_sounding_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}'
        if 'mobile' in this_sounding_origin:
            fancy_string += f' LOCATION:({float(this_sounding_lat):.2f}, {float(this_sounding_lon):.2f})'
        if this_sounding_origin not in soundings_dict.keys():
            soundings_dict[this_sounding_origin] = []
        soundings_dict[this_sounding_origin].append(fancy_string)
    return soundings_dict

def parse_ccn_obs(ccn_times, ccn_lon, ccn_lat):
    if np.all(np.array(ccn_lon).astype(float) == float(ccn_lon[0])):
        ccn_lon = ['' for _ in ccn_times]
        ccn_lat = ['' for _ in ccn_times]
    ccn_dict = {
        'times': [dt.strptime(t, '%Y-%m-%d %H:%M:%S') for t in ccn_times],
        'lons': ccn_lon,
        'lats': ccn_lat
    }
    return ccn_dict

if __name__ == '__main__':
    all_outs = sorted(glob('/Volumes/LtgSSD/tobac_saves/*/seabreeze-obs.zarr'))
    all_dates = [dt.strptime(out_file, '/Volumes/LtgSSD/tobac_saves/tobac_Save_%Y%m%d/seabreeze-obs.zarr') for out_file in all_outs]
    for this_out, this_date in zip(all_outs, all_dates):
        save_path = f'./thesis_figs/appendix/{this_date.strftime('%Y-%m-%d')}.png'
        if path.exists(save_path):
            continue
        this_ds = xr.open_dataset(this_out, engine='zarr')
        sbf_analysis_dts = [dt.strptime(this_date.strftime('%Y-%m-%d_')+t, '%Y-%m-%d_%H:%M:%S UTC') for t in this_ds.attrs['seabreeze_analysis_times'].split('\n')]
        maritime_sounding_paths = this_ds.attrs['maritime_soundings_used'].split('\n')
        maritime_sounding_lons = this_ds.attrs['maritime_soundings_lons'].split('\n')
        maritime_sounding_lats = this_ds.attrs['maritime_soundings_lats'].split('\n')
        maritime_sounding_times = this_ds.attrs['maritime_soundings_dts'].split('\n')
        maritime_soundings_dict = parse_sounding_paths(maritime_sounding_paths, maritime_sounding_lons, maritime_sounding_lats, maritime_sounding_times, this_date)

        continental_sounding_paths = this_ds.attrs['continental_soundings_used'].split('\n')
        continental_sounding_lons = this_ds.attrs['continental_soundings_lons'].split('\n')
        continental_sounding_lats = this_ds.attrs['continental_soundings_lats'].split('\n')
        continental_sounding_times = this_ds.attrs['continental_soundings_dts'].split('\n')
        continental_soundings_dict = parse_sounding_paths(continental_sounding_paths, continental_sounding_lons, continental_sounding_lats, continental_sounding_times, this_date)

        unique_ccn_counters = []
        for ccn_possible in this_ds.attrs.keys():
            if '_ccn_times' in ccn_possible:
                unique_ccn_counters.append(ccn_possible.replace('_times', '').replace('_maritime', '').replace('_continental', '').replace('_ccn', ''))
        unique_ccn_counters = np.unique(unique_ccn_counters)
        
        maritime_ccn_dict = {}
        continental_ccn_dict = {}
        for unique_loc in unique_ccn_counters:
            if len(this_ds.attrs[f'{unique_loc}_maritime_ccn_times']) > 0:
                lons = this_ds.attrs[f'{unique_loc}_maritime_ccn_lon'].split('\n')
                lats = this_ds.attrs[f'{unique_loc}_maritime_ccn_lat'].split('\n')
                this_maritime_ccn_loc = parse_ccn_obs(this_ds.attrs[f'{unique_loc}_maritime_ccn_times'].split('\n'), lons, lats)
                if this_maritime_ccn_loc['lons'] == ['' for _ in this_maritime_ccn_loc['lons']]:
                    unique_loc_new = unique_loc + f' ({lons[0]}, {lats[0]})'
                else:
                    unique_loc_new = unique_loc
                maritime_ccn_dict[unique_loc_new] = this_maritime_ccn_loc
            if len(this_ds.attrs[f'{unique_loc}_continental_ccn_times']) > 0:
                lons = this_ds.attrs[f'{unique_loc}_continental_ccn_lon'].split('\n')
                lats = this_ds.attrs[f'{unique_loc}_continental_ccn_lat'].split('\n')
                this_continental_ccn_loc = parse_ccn_obs(this_ds.attrs[f'{unique_loc}_continental_ccn_times'].split('\n'), lons, lats)
                if this_continental_ccn_loc['lons'] == ['' for _ in this_continental_ccn_loc['lons']]:
                    unique_loc_new = unique_loc + f' ({lons[0]}, {lats[0]})'
                else:
                    unique_loc_new = unique_loc
                continental_ccn_dict[unique_loc_new] = this_continental_ccn_loc
        unique_ccn_counters = np.unique(list(maritime_ccn_dict.keys()) + list(continental_ccn_dict.keys()))
        unique_locations = np.unique(list(maritime_soundings_dict.keys()) + list(continental_soundings_dict.keys()))
        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(nrows=1+1+len(unique_locations)+len(unique_ccn_counters), ncols=1, figure=fig, height_ratios=[1]+[0.1]+[1]*len(unique_locations)+[1]*len(unique_ccn_counters))
        fig.suptitle(f'Data sources for {this_date.strftime("%Y-%m-%d")}')
        axs = [fig.add_subplot(gs[i, 0]) for i in range(1+1+len(unique_locations)+len(unique_ccn_counters))]
        axs[0].set_title('Seabreeze analysis times')
        axs[0].scatter(sbf_analysis_dts, np.full(len(sbf_analysis_dts), 0.5), marker='*', color='goldenrod', s=20, edgecolor='k')
        axs[0].set_yticks([])
        axs[0].set_ylim(0, 1)
        
        for ax_index, this_location in enumerate(unique_locations):
            axs[ax_index+2].set_title(f'Profiles from {this_location}')
            maritime_times = []
            maritime_lons = []
            maritime_lats = []
            if this_location in maritime_soundings_dict.keys():
                if 'mobile' in this_location:
                    maritime_times += [dt.strptime(t.split('LOCATION')[0], '%Y-%m-%d %H:%M:%S UTC ') for t in maritime_soundings_dict[this_location]]
                    maritime_lons += [float(t.split('LOCATION:(')[1].split(',')[1].strip(')')) for t in maritime_soundings_dict[this_location]]
                    maritime_lats += [float(t.split('LOCATION:(')[1].split(',')[0]) for t in maritime_soundings_dict[this_location]]
                else:
                    maritime_times += [dt.strptime(t, '%Y-%m-%d %H:%M:%S UTC') for t in maritime_soundings_dict[this_location]]
                    maritime_lons += ['' for t in maritime_soundings_dict[this_location]]
                    maritime_lats += ['' for t in maritime_soundings_dict[this_location]]
            continental_times = []
            continental_lons = []
            continental_lats = []
            if this_location in continental_soundings_dict.keys():
                if 'mobile' in this_location:
                    continental_times += [dt.strptime(t.split('LOCATION')[0], '%Y-%m-%d %H:%M:%S UTC ') for t in continental_soundings_dict[this_location]]
                    continental_lons += [float(t.split('LOCATION:(')[1].split(',')[1].strip(')')) for t in continental_soundings_dict[this_location]]
                    continental_lats += [float(t.split('LOCATION:(')[1].split(',')[0]) for t in continental_soundings_dict[this_location]]
                else:
                    continental_times += [dt.strptime(t, '%Y-%m-%d %H:%M:%S UTC') for t in continental_soundings_dict[this_location]]
                    continental_lons += ['' for t in continental_soundings_dict[this_location]]
                    continental_lats += ['' for t in continental_soundings_dict[this_location]]
            maritime_handle = axs[ax_index+2].scatter(maritime_times, np.full(len(maritime_times), 0.5), marker='o', color='tab:blue', s=20, edgecolor='k', label='Maritime Profile')
            continental_handle = axs[ax_index+2].scatter(continental_times, np.full(len(continental_times), 0.5), marker='o', color='tab:red', s=20, edgecolor='k', label='Continental Profile')
            for i, maritime_time in enumerate(maritime_times):
                vertical_placement = 0.85 if (i % 2 == 0) else 0.15
                if maritime_lons[i] != '' and maritime_lats[i] != '':
                    axs[ax_index+2].text(maritime_time, vertical_placement, f'({maritime_lons[i]:.2f}, {maritime_lats[i]:.2f})', ha='center', va='top', fontsize=8)
            for i, continental_time in enumerate(continental_times):
                vertical_placement = 0.85 if (i % 2 == 0) else 0.15
                if continental_lons[i] != '' and continental_lats[i] != '':
                    axs[ax_index+2].text(continental_time, vertical_placement, f'({continental_lons[i]:.2f}, {continental_lats[i]:.2f})', ha='center', va='top', fontsize=8)
            axs[ax_index+2].set_yticks([])
            axs[ax_index+2].set_ylim(0, 1)
        for ax_index, this_ccn_loc in enumerate(unique_ccn_counters):
            ax = axs[ax_index+2+len(unique_locations)]
            ax.set_title(f'{this_ccn_loc.upper()} CCN observations')
            maritime_times = maritime_ccn_dict[this_ccn_loc]['times'] if this_ccn_loc in maritime_ccn_dict.keys() else []
            maritime_lons = maritime_ccn_dict[this_ccn_loc]['lons'] if this_ccn_loc in maritime_ccn_dict.keys() else []
            maritime_lats = maritime_ccn_dict[this_ccn_loc]['lats'] if this_ccn_loc in maritime_ccn_dict.keys() else []
            continental_times = continental_ccn_dict[this_ccn_loc]['times'] if this_ccn_loc in continental_ccn_dict.keys() else []
            continental_lons = continental_ccn_dict[this_ccn_loc]['lons'] if this_ccn_loc in continental_ccn_dict.keys() else []
            continental_lats = continental_ccn_dict[this_ccn_loc]['lats'] if this_ccn_loc in continental_ccn_dict.keys() else []
            if len(maritime_times) > 0:
                maritime_handle = ax.scatter(maritime_times, np.full(len(maritime_times), 0.5), color='tab:blue', s=20, edgecolor='k', label='Maritime CCN')
                for i, maritime_time in enumerate(maritime_times):
                    if maritime_lons[i] != '' and maritime_lats[i] != '':
                        ax.text(maritime_time, 0.45, f'{maritime_lons[i]}, {maritime_lats[i]}', ha='center', va='top', fontsize=8)
            if len(continental_times) > 0:
                continental_handle = ax.scatter(continental_times, np.full(len(continental_times), 0.5), color='tab:red', s=20, edgecolor='k', label='Continental CCN')
                for i, continental_time in enumerate(continental_times):
                    if continental_lons[i] != '' and continental_lats[i] != '':
                        ax.text(continental_time, 0.45, f'{continental_lons[i]}, {continental_lats[i]}', ha='center', va='top', fontsize=8)
            ax.set_yticks([])
            ax.set_ylim(0, 1)
        axs[1].legend(handles=[maritime_handle, continental_handle], loc='center', ncol=2)
        axs[1].axis('off')

        earliest_time = np.min([np.min(sbf_analysis_dts),
                                np.min(maritime_times) if len(maritime_times) > 0 else dt(this_date.year, this_date.month, this_date.day, 23, 59, 59),
                                np.min(continental_times) if len(continental_times) > 0 else dt(this_date.year, this_date.month, this_date.day, 23, 59, 59)])
        latest_time = np.max([np.max(sbf_analysis_dts),
                                np.max(maritime_times) if len(maritime_times) > 0 else dt(this_date.year, this_date.month, this_date.day, 0, 0, 0),
                                np.max(continental_times) if len(continental_times) > 0 else dt(this_date.year, this_date.month, this_date.day, 0, 0, 0)])
        [ax.set_xlim(earliest_time, latest_time) for ax in axs]
        [ax.xaxis.set_major_formatter(DateFormatter('%H:%M')) for ax in axs]
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    print('--------------')
    for date in all_dates:
        print('\\begin{figure}')
        print('    \\centering')
        print(f'    \\includegraphics[width=1\\linewidth]{{./figures/appendix/{date.strftime('%Y-%m-%d')}.png}}')
        print(f'    \\caption{{Data sources for {date.strftime('%Y-%m-%d')}}}')
        print(f'    \\label{{fig:appendix_{date.strftime('%Y-%m-%d')}}}')
        print('\\end{figure}')