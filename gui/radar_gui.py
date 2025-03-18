from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.iconfonts import *
from kivy.uix.screenmanager import ScreenManager, NoTransition, CardTransition, SwapTransition
from kivy.uix.label import Label 
from kivy.lang import Builder
from kivy.clock import Clock

from os.path import join, dirname
from sys_config import *
from user_config import *
from libs.utils import connectToMainDB

class RadarGuiWindow(BoxLayout):
	pass

class RadarGuiApp(App):
    def __init__(self, **kwargs):
        self.results_dict_q = kwargs.get("results_dict_q")
        self.gui_quit_q = kwargs.get("gui_quit_q")
        super().__init__()
        
    def build(self):
        self.screen_manager = ScreenManager(transition = SwapTransition())
        self.title = 'Ma-Lab: Radar-Based Activity/Health Parameters Monitoring'
        self.root = RadarGuiWindow()
        self.root.ids.screens_holder.add_widget(self.screen_manager)
        self.result_dict = {}
        
        self.show_welcome_screen()  
        Clock.schedule_interval(self.check_refresh, 0.05)

    def show_welcome_screen(self):
        '''Display vitals Screen'''
        for child in self.screen_manager.children:
            self.screen_manager.remove_widget(child)

        from gui.backend.screen_welcome import ScreenWelcome
        # print('showing welcome screen')
        self.screen_welcome = ScreenWelcome(name='welcome')
        self.screen_manager.add_widget(self.screen_welcome)
        self.screen_manager.current = 'welcome'

    def show_presence_screen(self):
        '''Display presence analysis Screen'''
        for child in self.screen_manager.children:
            self.screen_manager.remove_widget(child)

        from gui.backend.screen_presence import ScreenPresence
        # print('showing presence screen')
        self.screen_presence = ScreenPresence(name='presence')
        self.screen_manager.add_widget(self.screen_presence)
        self.screen_manager.current = 'presence'

    def show_respiration_screen(self):
        '''Display respiration analysis Screen'''
        for child in self.screen_manager.children:
            self.screen_manager.remove_widget(child)

        from gui.backend.screen_respiration import ScreenRespiration
        # print('showing respiration screen')
        self.screen_respiration = ScreenRespiration(name='respiration')
        self.screen_manager.add_widget(self.screen_respiration)
        self.screen_manager.current = 'respiration'

    def show_heartbeat_screen(self):
        '''Display heartbeat analysis Screen'''
        for child in self.screen_manager.children:
            self.screen_manager.remove_widget(child)
            
        from gui.backend.screen_heartbeat import ScreenHeartbeat
        # print('showing heartbeat screen')
        self.screen_heartbeat = ScreenHeartbeat(name='heartbeat')
        self.screen_manager.add_widget(self.screen_heartbeat)
        self.screen_manager.current = 'heartbeat'

    def check_refresh(self, dt):
        # Get latest Results:
        if self.results_dict_q.qsize():
            for i in range(self.results_dict_q.qsize()):
                self.result_dict = self.results_dict_q.get()

            # Update Stats
            self.root.ids.presence_var.text = "Subject" if self.result_dict["presence_flag"][-1] else "No One here !!!"
            self.root.ids.presence_var.color = "green" if self.result_dict["presence_flag"][-1] else "red"

            if self.result_dict["presence_flag"][-1]:
                self.root.ids.range_var.text = f'{round(self.result_dict["target_bin"][-1][0]*RANGE_RESOLUTION*100,2)} cm' 
                self.root.ids.range_var.color = 'black'
                self.root.ids.motion_var.color = 'green' if self.result_dict["Motion_State_bin"]=="Static" else "red"
                self.root.ids.motion_var.text = self.result_dict["Motion_State_bin"]

                # self.root.ids.pattern_var.text = self.result_dict["respiration_pattern"]
                # self.root.ids.pattern_var.color = 'green' if self.result_dict["respiration_pattern"]=="NORMAL" else 'red'
                try:
                    self.root.ids.rr_var.text = f'{self.result_dict["respiration_bpm_fft"][-1]} bpm'
                    self.root.ids.rr_var.color = 'black'
                    self.root.ids.hr_var.text = f'{self.result_dict["heartbeat_bpm_fft"][-1]} bpm'
                    self.root.ids.hr_var.color = 'black'
                except: print('stat display error !!!')
            else:
                self.root.ids.range_var.text = 'N/A' 
                self.root.ids.range_var.color = 'grey'

                self.root.ids.pattern_var.text = 'N/A'
                self.root.ids.pattern_var.color = 'grey' 

                self.root.ids.rr_var.text = 'N/A'
                self.root.ids.rr_var.color = 'grey'

                self.root.ids.hr_var.text = 'N/A'
                self.root.ids.hr_var.color = 'grey'

                self.root.ids.motion_var.text = "N/A"
                self.root.ids.motion_var.color = 'grey'

            # Refresh Graphs:
            current_screen = self.screen_manager.current
            for scr in self.screen_manager.children:
                if scr.name == current_screen:
                    if current_screen == 'welcome':
                        self.show_welcome_screen()
                    if current_screen == 'presence':
                        self.show_presence_screen()
                    if current_screen == 'respiration':
                        self.show_respiration_screen()
                    if current_screen == 'heartbeat':
                        self.show_heartbeat_screen()
                else:
                    self.screen_manager.remove_widget(scr) 

            if USE_SERVER:
                # Update DB: For Mobile/Web App
                try: 
                
                    conn, curs = connectToMainDB()
                    curs.execute('''UPDATE vitals SET hr=%s, br=%s, distance=%s, angle=%s, presence=%s, motion=%s WHERE id_record=%s''', (
                        float(self.result_dict["heartbeat_bpm_fft"][-1]), self.result_dict["respiration_bpm_fft"][-1], float(round(self.result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)), 0, 1 if self.result_dict["presence_flag"][-1] else 0, 0 if self.result_dict["Motion_State_bin"]=="Static" else 1, 1 
                    ))
                    conn.commit()
                    conn.close()   
                except Exception as e: print(style.RED + f'>>> radar_gui.py: refresh fct: Update DB --> {e}' + style.RESET)

    def quit_gui(self):
        self.gui_quit_q.put(True)
        print(style.RED + "\nApp is Exiting ...\n" + style.RESET)


register('default_font', 'gui/assets/Fonts/Material-Design-Iconic-Font.ttf', join(dirname(__file__),'assets/Fonts/zmd.fontd'))

# from kivy.core.window import Window
# Window. size = (1920, 1000)
# Window.top = 30
# Window.left = 0
# # Window.fullscreen = True