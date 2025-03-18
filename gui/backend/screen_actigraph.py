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
        ax = plt.subplot(111)
        ax.cla()
        ax.set_ylim(-30,30)

        try:
        
            if self.app.result_dict["presence_flag"]:
                if USE_VERNIER:
                    plt.plot(self.app.result_dict["x_vernier_array"], self.app.result_dict["vernier_new_data"], 'g', label="Vernier")
                
                ax.set_title(f'Respiration Signal: Target at {round(self.app.result_dict["target_bin"]*RANGE_RESOLUTION,2)} m')
                ax.plot(self.app.result_dict["radar_time_array"][1:], self.app.result_dict["respiration_signal"], 'r', label="Radar")

            else:
                ax.set_title(f'Respiration Signal: No Detection !!!')
                plt.plot(np.zeros(30), np.zeros(30), 'r', label="Radar")

        except: pass


        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (sec)')
        ax.legend(loc='lower left')

        ################## Breathing #################################################################################
        # ax4 = plt.subplot(212)
        # ax4.cla()
        # ax4.set_ylim(0,40)


        # ax4.set_xlabel('Time (sec)')
        # ax4.set_ylabel('Breathing Rate')
        # ax4.set_title('Breathing Rate Signal')
        # ax4.legend(loc='lower left')
        self.ids.graph_holder.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        
