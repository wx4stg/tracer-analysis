#!/usr/bin/env python3
# Sam's sounding plotter
# Created 17 May 2023 by Sam Gardner <sam@wx4stg.com>


import metpy.calc as mpcalc
from os import listdir, path
import numpy as np
from datetime import datetime as dt
import pandas as pd
import holoviews as hv
import panel as pn
from redacted import readSharppy, read_arm_doe
import sys

hv.extension('bokeh')

class ABL_analyzer:
    def __init__(self, dates_and_paths, source):
        self.dataset = None
        self.source = source
        self.selected_height_label = pn.pane.Markdown('')
        self.highlighted_height_label = pn.pane.Markdown('')
        if path.exists('abl_heights.csv'):
            self.abl_heights = pd.read_csv('abl_heights.csv')
        else:
            self.abl_heights = pd.DataFrame(columns=['filename', 'abl_height'])
        self.file_dropdown = pn.widgets.Select(name='Select a file', options=dates_and_paths)
        self.update_dataset(self.file_dropdown.value)
        self.kernel_size = pn.widgets.IntSlider(name='Kernel Size', start=1, end=10, value=5)
        self.data_panels = self.plot_ABL_analysis(self.file_dropdown, self.kernel_size)
        self.last_mouse_coord = [0, 0, 0, 0, 0, 0, 0, 0, -1]
        self.labels = pn.Column(self.selected_height_label, self.highlighted_height_label)


    def update_dataset(self, filename):
        if self.dataset is None or self.dataset.attrs['filename'] != filename:
            if self.source == 'tamu':
                ds, _, sounding_dt = readSharppy(filename)
            elif self.source == 'doe':
                ds, _, sounding_dt = read_arm_doe(filename)
            ds.attrs['filename'] = filename
            ds.attrs['datetime'] = sounding_dt
            sounding_dt = ds.attrs['datetime']
            self.dataset = ds
            self.last_mouse_coord = [0, 0, 0, 0, 0, 0, 0, 0, -1]
            return ds
        else:
            return self.dataset


    def plot_curve_theta(self, filename):
        dataset = self.update_dataset(filename)
        pot_temp = mpcalc.potential_temperature(dataset.LEVEL, dataset.TEMP)
        pot_temp_curve = hv.Curve((pot_temp, dataset.HGHT), kdims=["Potential Temperature (K)"], vdims=["Height (m)"], label="Potential Temperature").opts(tools=['hover'], color='purple')
        return pot_temp_curve

    def plot_curve_theta_deriv(self, filename, kernel_size):
        dataset = self.update_dataset(filename)
        pot_temp = mpcalc.potential_temperature(dataset.LEVEL, dataset.TEMP)
        pot_temp_deriv = np.gradient(pot_temp.data.magnitude, dataset.HGHT.data.magnitude)
        pot_temp_deriv_avg = np.convolve(pot_temp_deriv, np.ones(kernel_size)/kernel_size, mode='same')
        pot_temp_deriv_curve = hv.Curve((pot_temp_deriv_avg, dataset.HGHT), kdims=["Potential Temperature Derivative (K/m)"], vdims=["Height (m)"], label="Potential Temperature Derivative").opts(tools=['hover'], color='purple')
        return pot_temp_deriv_curve

    def plot_curve_wvmr(self, filename):
        dataset = self.update_dataset(filename)
        wvmr = mpcalc.mixing_ratio_from_relative_humidity(dataset.LEVEL, dataset.TEMP, dataset.RH)
        wvmr_curve = hv.Curve((wvmr, dataset.HGHT), kdims=["Water Vapor Mixing Ratio (g/kg)"], vdims=["Height (m)"], label="Water Vapor Mixing Ratio").opts(tools=['hover'], color='green')
        return wvmr_curve

    def plot_curve_wvmr_deriv(self, filename, kernel_size):
        dataset = self.update_dataset(filename)
        wvmr = mpcalc.mixing_ratio_from_relative_humidity(dataset.LEVEL, dataset.TEMP, dataset.RH)
        wvmr_deriv = np.gradient(wvmr.data.magnitude, dataset.HGHT.data.magnitude)
        wvmr_deriv_avg = np.convolve(wvmr_deriv, np.ones(kernel_size)/kernel_size, mode='same')
        wvmr_deriv_curve = hv.Curve((wvmr_deriv_avg, dataset.HGHT), kdims=["Water Vapor Mixing Ratio Derivative (g/kg/m)"], vdims=["Height (m)"], label="Water Vapor Mixing Ratio Derivative").opts(tools=['hover'], color='green')
        return wvmr_deriv_curve

    def plot_curve_thetav(self, filename):
        dataset = self.update_dataset(filename)
        wvmr = mpcalc.mixing_ratio_from_relative_humidity(dataset.LEVEL, dataset.TEMP, dataset.RH)
        thetav = mpcalc.virtual_potential_temperature(dataset.LEVEL, dataset.TEMP, wvmr)
        thetav_curve = hv.Curve((thetav, dataset.HGHT), kdims=["Virtual Potential Temperature (K)"], vdims=["Height (m)"], label="Virtual Potential Temperature").opts(tools=['hover'], color='orange')
        return thetav_curve

    def plot_curve_thetav_deriv(self, filename, kernel_size):
        dataset = self.update_dataset(filename)
        wvmr = mpcalc.mixing_ratio_from_relative_humidity(dataset.LEVEL, dataset.TEMP, dataset.RH)
        thetav = mpcalc.virtual_potential_temperature(dataset.LEVEL, dataset.TEMP, wvmr)
        thetav_deriv = np.gradient(thetav.data.magnitude, dataset.HGHT.data.magnitude)
        thetav_deriv_avg = np.convolve(thetav_deriv, np.ones(kernel_size)/kernel_size, mode='same')
        thetav_deriv_curve = hv.Curve((thetav_deriv_avg, dataset.HGHT), kdims=["Virtual Potential Temperature Derivative (K/m)"], vdims=["Height (m)"], label="Virtual Potential Temperature Derivative").opts(tools=['hover'], color='orange')
        return thetav_deriv_curve

    def plot_curve_wind_speed(self, filename):
        dataset = self.update_dataset(filename)
        wind_speed_curve = hv.Curve((dataset.WSPD, dataset.HGHT), kdims=["Wind Speed (kt)"], vdims=["Height (m)"], label="Wind Speed").opts(tools=['hover'], color='blue')
        return wind_speed_curve

    def plot_curve_wind_speed_deriv(self, filename, kernel_size):
        dataset = self.update_dataset(filename)
        wind_deriv = np.gradient(dataset.WSPD.data.magnitude, dataset.HGHT.data.magnitude)
        wind_deriv_avg = np.convolve(wind_deriv, np.ones(kernel_size)/kernel_size, mode='same')
        wind_speed_curve = hv.Curve((wind_deriv_avg, dataset.HGHT), kdims=["Wind Speed Derivative (kt/m)"], vdims=["Height (m)"], label="Wind Speed").opts(tools=['hover'], color='blue')
        return wind_speed_curve

    def get_datetime(self, filename):
        dataset = self.update_dataset(filename)
        return dataset.attrs['datetime'].strftime('Sounding from %Y-%m-%d %H:%M:%S')
    
    def highlight_vertical_position(self, y0, y1, y2, y3, y4, y5, y6, y7):
        coords = [y0, y1, y2, y3, y4, y5, y6, y7]
        crosshair = hv.HLine(0).opts(alpha=0)
        self.last_mouse_coord[-1] = -1
        for i in range(len(coords)):
            if coords[i] != self.last_mouse_coord[i]:
                self.last_mouse_coord[-1] = i
                crosshair = hv.HLine(coords[i]).opts(color='black')
                self.highlighted_height_label.object = f'Highlighted at {coords[i]} m'
                break
        self.last_mouse_coord[:-1] = coords
        return crosshair

    def select_abl_top(self, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7):
        ys = [y0, y1, y2, y3, y4, y5, y6, y7]
        crosshair = hv.HLine(0).opts(alpha=0)
        if self.last_mouse_coord[-1] != -1:
            height_of_top = ys[self.last_mouse_coord[-1]]
            crosshair = hv.HLine(height_of_top).opts(color='red', alpha=0.5)
            new_row = pd.DataFrame({'filename': path.basename(self.file_dropdown.value), 'abl_height': height_of_top}, index=[0])
            self.abl_heights = pd.concat([self.abl_heights, new_row], ignore_index=True)
            self.abl_heights.drop_duplicates(subset=['filename'], keep='last', inplace=True)
            self.abl_heights.sort_values(by='filename', inplace=True)
            self.abl_heights.to_csv('abl_heights.csv', index=False)
            self.selected_height_label.object = f'ABL top selected at {height_of_top} m'
        return crosshair

    def plot_ABL_top(self, filename):
        if path.basename(filename) in self.abl_heights['filename'].values:
            abl_height = self.abl_heights[self.abl_heights['filename'] == path.basename(filename)]['abl_height'].values[0]
            height_plotter_last = hv.HLine(abl_height).opts(color='green', alpha=0.5)
            self.selected_height_label.object = f'ABL top selected at {abl_height} m'
        else:
            height_plotter_last = hv.HLine(0).opts(alpha=0)
            self.selected_height_label.object = ''
        return height_plotter_last
    
    def plot_ABL_analysis(self, file_dropdown, kernel_size):
        all_curves = []
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_theta, filename=file_dropdown, watch=True)).opts(show_legend=False)) # potential temperature
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_theta_deriv, filename=file_dropdown, kernel_size=kernel_size, watch=True)).opts(show_legend=False)) # derivative of potential temperature
        
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_wvmr, filename=file_dropdown, watch=True)).opts(show_legend=False)) # water vapor mixing ratio
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_wvmr_deriv, filename=file_dropdown, kernel_size=kernel_size, watch=True)).opts(show_legend=False)) # derivative of water vapor mixing ratio
        
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_thetav, filename=file_dropdown, watch=True)).opts(show_legend=False)) # virtual potential temperature
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_thetav_deriv, filename=file_dropdown, kernel_size=kernel_size, watch=True)).opts(show_legend=False)) # derivative of virtual potential temperature

        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_wind_speed, filename=file_dropdown, watch=True)).opts(show_legend=False)) # wind speed
        all_curves.append(hv.DynamicMap(pn.bind(self.plot_curve_wind_speed_deriv, filename=file_dropdown, kernel_size=kernel_size, watch=True)).opts(show_legend=False)) # derivative of wind speed

        date_label = pn.pane.Markdown(pn.bind(self.get_datetime, filename=file_dropdown, watch=True))

        all_height_streams = [hv.streams.PointerY(source=all_curves[i], y=0).rename(y=f'y{i}') for i in range(len(all_curves))]
        height_plotter = hv.DynamicMap(self.highlight_vertical_position, streams=all_height_streams)
        all_tap_streams = [hv.streams.Tap(source=all_curves[i], x=0, y=0).rename(x=f'x{i}', y=f'y{i}')
                       for i in range(len(all_curves))]
        tap_plotter = hv.DynamicMap(self.select_abl_top, streams=all_tap_streams)

        height_plotter_last = hv.DynamicMap(pn.bind(self.plot_ABL_top, filename=file_dropdown, watch=True))

        all_curves = [curve * tap_plotter * height_plotter * height_plotter_last for curve in all_curves]
            
        pot_temp_pane = pn.pane.HoloViews(all_curves[0])
        pot_temp_deriv_pane = pn.pane.HoloViews(all_curves[1])
        wvmr_pane = pn.pane.HoloViews(all_curves[2])
        wvmr_deriv_pane = pn.pane.HoloViews(all_curves[3])
        thetav_pane = pn.pane.HoloViews(all_curves[4])
        thetav_deriv_pane = pn.pane.HoloViews(all_curves[5])
        wind_speed_pane = pn.pane.HoloViews(all_curves[6])
        wind_speed_deriv_pane = pn.pane.HoloViews(all_curves[7])

        thermo_panes = pn.Column(date_label, pn.Row(pn.Column(pot_temp_pane, pot_temp_deriv_pane), pn.Column(wvmr_pane, wvmr_deriv_pane), pn.Column(thetav_pane, thetav_deriv_pane), pn.Column(wind_speed_pane, wind_speed_deriv_pane)))

        return thermo_panes
    


if __name__ == "__main__":
    if sys.argv[1] == 'tamu':
        tamu_files = sorted(listdir('TAMU_TRACER_radiosonde_data_final/SHARPPY'))
        tamu_dates = [dt.strptime(file.split('_')[2]+file.split('_')[3], '%Y%m%d%H%M') for file in tamu_files]
        tamu_files = ['./TAMU_TRACER_radiosonde_data_final/SHARPPY/' + file for file in tamu_files]
        dates_and_paths = {date.strftime('%Y-%m-%d %H:%M:%S'): path for date, path in zip(tamu_dates, tamu_files)}
    elif sys.argv[1] == 'doe':
        doe_files = sorted(listdir('armdata'))
        laporte_files = [file for file in doe_files if 'M1' in file or 'S1' in file]
        laporte_dates = [dt.strptime(file.split('.')[2]+file.split('.')[3], '%Y%m%d%H%M%S') for file in laporte_files]
        laporte_files = ['./armdata/' + file for file in laporte_files]
        laporte_dates_and_paths = {date.strftime('LA PORTE: %Y-%m-%d %H:%M:%S'): path for date, path in zip(laporte_dates, laporte_files)}
        guy_sites = [file for file in doe_files if 'S3' in file or 'S4' in file]
        guy_dates = [dt.strptime(file.split('.')[2]+file.split('.')[3], '%Y%m%d%H%M%S') for file in guy_sites]
        guy_files = ['./armdata/' + file for file in guy_sites]
        guy_dates_and_paths = {date.strftime('GUY: %Y-%m-%d %H:%M:%S'): path for date, path in zip(guy_dates, guy_files)}
        dates_and_paths = {**laporte_dates_and_paths, **guy_dates_and_paths}


    my_abl = ABL_analyzer(dates_and_paths, sys.argv[1])
    pn.serve(pn.Column(my_abl.file_dropdown, my_abl.kernel_size, my_abl.data_panels, my_abl.labels))
