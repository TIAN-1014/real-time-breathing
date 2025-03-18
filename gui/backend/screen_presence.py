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

class ScreenPresence(Screen):
    def __init__(self, **kwargs):
        self.app = App.get_running_app()
        
        try:
            Builder.unload_file('gui/frontend/screen_presence.kv')
        except:
            pass
        Builder.load_file('gui/frontend/screen_presence.kv')
        super().__init__(**kwargs)
        self.show_graph()

    def show_graph(self):
        # ax_rtm = plt.subplot(111)
        # ax_rtm.cla()

        # try:
        #     ax_rtm.imshow(self.app.result_dict["magn_data"], cmap='viridis', aspect='auto', extent=[0, MAX_DISTANCE*RANGE_RESOLUTION, 0, -PLOTTING_WIN])
        # except: pass
        
        # ax_rtm.set_ylabel('Time (sec)')
        # ax_rtm.set_xlabel('Range (m)')
        # ax_rtm.set_title('Range Heatmap')
        # # ax1.legend(loc='lower left')

        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_title('Radar Range Heatmap')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Range (m)')

        plt.imshow(self.app.result_dict["magn_data"].T, cmap='viridis', aspect='auto', origin="lower", extent=[-PLOTTING_WIN, 0,  0, MAX_DISTANCE*RANGE_RESOLUTION])

        self.ids.graph_holder.clear_widgets()

        try:
            self.ids.graph_holder.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        except: pass
