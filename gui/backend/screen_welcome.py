from kivy.uix.screenmanager import Screen

########################################################################################################################################
# Layouts Screen
########################################################################################################################################       
from kivy.lang.builder import Builder
Builder.load_file('gui/frontend/screen_welcome.kv')

class ScreenWelcome(Screen):
    pass