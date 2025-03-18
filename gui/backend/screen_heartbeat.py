from kivy.app import App
from kivy_garden.matplotlib import FigureCanvasKivyAgg
from kivy.uix.screenmanager import Screen

import numpy as np
import matplotlib.pyplot as plt
from sys_config import *
from user_config import *

########################################################################################################################################
# Layouts Screen
########################################################################################################################################       
from kivy.lang.builder import Builder

class ScreenHeartbeat(Screen):
    def __init__(self, **kwargs):
        self.app = App.get_running_app()        
        try:
            Builder.unload_file('gui/frontend/screen_heartbeat.kv')
        except:
            pass
        Builder.load_file('gui/frontend/screen_heartbeat.kv')

        super().__init__(**kwargs)
        self.show_graph()

    def show_graph(self):
        radar_time_array = np.linspace(-PLOTTING_WIN, 0, FPS*PLOTTING_WIN)
        ax_hbt = plt.subplot(211)
        ax_hbt.cla()
        ax_hbt.set_ylim(-30,30)

        try:
        
            if self.app.result_dict["presence_flag"][-1]:
                if USE_POLAR:
                    plt.plot(self.app.result_dict["x_polar_ecg_array"], self.app.result_dict["polar_ecg_new_data"], 'g', label="Polar")
                
                ax_hbt.set_title(f'Heartbeat Signal: Target at {round(self.app.result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
                ax_hbt.plot(radar_time_array[1:], self.app.result_dict["heartbeat_signal"], 'r', label="Radar")

            else:
                ax_hbt.set_title(f'Heartbeat Signal: No Detection !!!')
                ax_hbt.plot(np.zeros(PLOTTING_WIN), np.zeros(PLOTTING_WIN), 'r', label="Radar")

        except: pass


        ax_hbt.set_ylabel('Amplitude')
        ax_hbt.set_xlabel('Time (sec)')
        ax_hbt.legend(loc='lower left')

        
        
        # Plotting Respiration rate:
        ax_hr = plt.subplot(212)
        ax_hr.cla()
        ax_hr.set_ylim(0,30)

        # try:
        
        if self.app.result_dict["presence_flag"][-1]:
            if USE_POLAR:
                # TODO: Add Polar HR
                pass
            ax_hr.set_title(f'Heart Rate: Target at {round(self.app.result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            
            print(len(self.app.result_dict["heartbeat_bpm_fft"]))
            ax_hr.plot(np.linspace(-PLOTTING_WIN, 0, PLOTTING_WIN), self.app.result_dict["heartbeat_bpm_fft"], 'r', label="Radar")

        else:
            ax_hr.set_title(f'Heart Rate: No Detection !!!')
            ax_hr.plot(np.zeros(PLOTTING_WIN), np.zeros(PLOTTING_WIN), 'r', label="Radar")

        # except: pass

        ax_hr.set_ylabel('BPM')
        ax_hr.set_xlabel('Time (sec)')
        ax_hr.legend(loc='lower left')

        try:
            self.ids.graph_holder.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        except: pass