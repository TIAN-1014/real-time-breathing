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

class ScreenRespiration(Screen):
    def __init__(self, **kwargs):
        self.app = App.get_running_app()        
        try:
            Builder.unload_file('gui/frontend/screen_respiration.kv')
        except:
            pass
        Builder.load_file('gui/frontend/screen_respiration.kv')

        super().__init__(**kwargs)
        self.show_graph()

    def show_graph(self):
        radar_time_array = np.linspace(-PLOTTING_WIN, 0, FPS*PLOTTING_WIN)

        # Plotting Respiration Signal
        ax_resp = plt.subplot(211)
        ax_resp.cla()
        ax_resp.set_ylim(-30,30)

        try:
        
            if self.app.result_dict["presence_flag"][-1]:
                if USE_VERNIER:
                    ax_resp.plot(self.app.result_dict["x_vernier_array"], self.app.result_dict["vernier_new_data"], 'g', label="Vernier")
                
                ax_resp.set_title(f'Respiration Signal: Target at {round(self.app.result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
                ax_resp.plot(radar_time_array[1:], self.app.result_dict["respiration_signal"], 'r', label="Radar")

                # Highlight Hold period in plot:
                if len(self.app.result_dict["respiration_pattern"])>0:
                    for pattern in self.app.result_dict["respiration_pattern"]:
                        ax_resp.axvspan(radar_time_array[pattern[0]], radar_time_array[pattern[1]], facecolor='red', alpha=0.2)
                        ax_resp.text(radar_time_array[pattern[0]]+(radar_time_array[pattern[1]]-radar_time_array[pattern[0]])//2, 20, 'HOLD')

            else:
                ax_resp.set_title(f'Respiration Signal: No Detection !!!')
                ax_resp.plot(np.zeros(PLOTTING_WIN), np.zeros(PLOTTING_WIN), 'r', label="Radar")

        except: pass

        ax_resp.set_ylabel('Amplitude')
        ax_resp.set_xlabel('Time (sec)')
        ax_resp.legend(loc='lower left')

        # Plotting Respiration rate:
        ax_rr = plt.subplot(212)
        ax_rr.cla()
        ax_rr.set_ylim(0,30)

        try:
        
            if self.app.result_dict["presence_flag"][-1]:
                if USE_VERNIER:
                    # TODO: Add Vernier Respiration Rate
                    pass
                ax_rr.set_title(f'Respiration Rate: Target at {round(self.app.result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
                ax_rr.plot(np.linspace(-PLOTTING_WIN, 0, PLOTTING_WIN), self.app.result_dict["respiration_bpm_fft"], 'r', label="Radar")

            else:
                ax_rr.set_title(f'Respiration Rate: No Detection !!!')
                ax_rr.plot(np.zeros(PLOTTING_WIN), np.zeros(PLOTTING_WIN), 'r', label="Radar")

        except: pass

        ax_rr.set_ylabel('BPM')
        ax_rr.set_xlabel('Time (sec)')
        ax_rr.legend(loc='lower left')

        try:
            self.ids.graph_holder.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        except: pass
