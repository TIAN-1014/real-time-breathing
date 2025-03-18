from gui.radar_gui import RadarGuiApp

def run_kivy_app(dict_result, gui_quit_q):
    RadarGuiApp(results_dict_q=dict_result, gui_quit_q=gui_quit_q).run()