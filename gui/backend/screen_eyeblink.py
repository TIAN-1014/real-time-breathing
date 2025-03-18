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

class ScreenEyeblink(Screen):
    def __init__(self, **kwargs):
        self.app = App.get_running_app()        
        try:
            Builder.unload_file('gui/frontend/screen_eyeblink.kv')
        except:
            pass
        Builder.load_file('gui/frontend/screen_eyeblink.kv')

        super().__init__(**kwargs)
        self.show_graph()

    def show_graph(self):
        ax = plt.subplot(111)
        ax.cla()
        ax.set_ylim(-30,30)

        try:
        
           pass

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
        
