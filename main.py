import os
from sys_config import *
from user_config import *

if not KIVY_LOG:
    os.environ["KIVY_NO_CONSOLELOG"] = "1"

from kivy.config import Config
Config.set('graphics', 'width', f'{GUI_SCREEN_SIZE[0]}')
Config.set('graphics', 'height', f'{GUI_SCREEN_SIZE[1]}')

Config.set('graphics', 'position', 'custom')
Config.set('graphics', 'top', '50') 
Config.set('graphics', 'left', '50') 

import datetime
import keyboard
from time import sleep
from multiprocessing import Process 

from libs.run_gui import run_kivy_app
from libs.process_radar_data import RadarProcessor
from libs.record_reference_data import ScreenRecorder, PolarController, VernierBeltController, VernierSpirometerController, HATController, WitMotionController, M5StickCController

if VERSION == "8_ANT":
    from libs.radar_controller_8ant import RadarController
    
elif VERSION == "1_ANT":
    from libs.radar_controller import RadarController
    
from libs.cache_data import save_1dfft, save_ref_data, save_to_excel

from libs.data_recording import start_timed_recording, control_recording

from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()    
scheduler.start()

if __name__ == "__main__":
    process_list = []

    if USE_RADAR:
        print(style.YELLOW + f"\nCurrent Radar Image: {VERSION}\n" + style.RESET)
        process_list.append(RadarController(state_q,
                            out_fft_array_q=out_fft_array_q,
                            out_fft_phase_q=out_fft_phase_q,
                            calculation_status=calculation_status,
                            start_radar_flag = START_RADAR_FLAG,
                            save_fft_array_q=save_fft_array_q,
                            pt_fft_q=pt_fft_q))
    
    if RECORD_VIDEO:
        # Camera process:
        # process_list.append(CameraController(CAMERA_INDEX, start_video_q=START_VIDEO_FLAG))    # --> Record from camera
        process_list.append(ScreenRecorder(CAMERA_INDEX, start_video_q=START_VIDEO_FLAG))    # --> Record from camera

    if USE_VERNIER_BELT:
        process_list.append(VernierBeltController(vernier_belt_ref_q=VERNIER_BELT_FORCE_DATA, 
                                            vernier_belt_connection_q=VERNIER_BELT_CONNECTED,
                                            start_vernier_belt_q=START_VERNIER_BELT_FLAG,
                                            vernier_belt_realtime_q=VERNIER_BELT_FORCE_DATA_REALTIME))
    
    if USE_VERNIER_SPIRO:
        process_list.append(VernierSpirometerController(vernier_spiro_ref_q=VERNIER_SPIRO_FORCE_DATA, 
                                            vernier_spiro_connection_q=VERNIER_SPIRO_CONNECTED,
                                            start_spiro_vernier_q=START_VERNIER_SPIRO_FLAG,
                                            vernier_spiro_realtime_q=VERNIER_SPIRO_FORCE_DATA_REALTIME))
        
    if USE_POLAR:
        process_list.append(PolarController(polar_mac_addr= POLAR_MAC,
                                         start_polar_q = START_POLAR_FLAG,
                                         polar_ecg_ref_q=POLAR_ECG_DATA, 
                                         polar_ecg_realtime_q = POLAR_ECG_DATA_REALTIME,
                                         polar_hr_realtime_q=POLAR_HR_DATA_REALTIME,
                                         polar_hr_ref_q=POLAR_HR_DATA, 
                                         polar_rri_ref_q=POLAR_RRI_DATA, 
                                         polar_acc_ref_q=POLAR_ACC_DATA,
                                         polar_connection_q=POLAR_CONNECTED))
        
    if USE_WITMOTION1:
        process_list.append(WitMotionController(WITMOTION1_MAC, WITMOTION1_DATA, START_WITMOTION1_FLAG, WITMOTION1_DATA_REALTIME))

    if USE_WITMOTION2:
        process_list.append(WitMotionController(WITMOTION2_MAC, WITMOTION2_DATA, START_WITMOTION2_FLAG, WITMOTION2_DATA_REALTIME))

    if USE_HAT:
        process_list.append(HATController(mac_address=HAT_MAC, m5stickc_data_q = HAT_DATA, start_m5stickc_q=START_HAT_FLAG, m5stickc_data_realtime_q= HAT_DATA_REALTIME))

    if USE_M5STICKC_MARKER:
        process_list.append(M5StickCController(mac_address=M5STICKC_MARKER_MAC, m5stickc_data_q = M5STICKC_MARKER_DATA, start_m5stickc_q=START_M5STICKC_MARKER_FLAG, m5stick_realtime_q=M5STICKC_MARKER_DATA_REALTIME, role="Marker"))

    if REALTIME_PLOT:
        process_list.append(RadarProcessor( in_q=out_fft_array_q,
                                            out_range_profile_q=out_range_profile_q,
                                            calculation_status=calculation_status,
                                            range_data_q = range_data_q,
                                            phase_data_q = phase_data_q,
                                            vernier_belt_realtime_q=VERNIER_BELT_FORCE_DATA_REALTIME,
                                            vernier_spiro_realtime_q=VERNIER_SPIRO_FORCE_DATA_REALTIME,
                                            witmotion1_realtime_q=WITMOTION1_DATA_REALTIME,
                                            witmotion2_realtime_q=WITMOTION2_DATA_REALTIME,
                                            polar_ecg_realtime_q = POLAR_ECG_DATA_REALTIME,
                                            polar_hr_realtime_q=POLAR_HR_DATA_REALTIME,
                                            calibration_flag = CALIBRATION_FLAG, 
                                            results_dict_q=RESULTS_DICT_Q,
                                            m5stickc_marker_realtime_q = M5STICKC_MARKER_DATA_REALTIME,
                                            hat_realtime_q = HAT_DATA_REALTIME
                                            ))
    
    if USE_MATPLOTLIB: # TODO: Switch to process-based
        # process_list.append(Plot_Results(results_dict_q=RESULTS_DICT_Q))
        pass
    else:
        process_list.append(Process(target=run_kivy_app, args=(RESULTS_DICT_Q, GUI_QUIT_Q)))

    # Start processes
    try:
        for process in process_list:
            process.daemon = True
            process.start()
    except KeyboardInterrupt:
        for process in process_list:
            process.terminate()
            process.join()

    # Wait for devices connection process: # TODO: add conditional check
    print(style.BLUE + f"\nWaiting for all devices to connect ... {WAIT_CONNECTION} sec\n" + style.RESET)
    sleep(WAIT_CONNECTION)   # Wait for different sensors to connect

    # Wait for calibration process:
    print(style.GREEN + "\nCalibration finished. Start recording ...\n" + style.RESET)

    # Run Recording
    if EXPERIMENT_MODE:
        print("Experiment Mode ...")
        start_timed_recording()
        print(f"Recording Finished ;) at {datetime.datetime.now()}") 

    else:
        print("Free Mode ... (Press 'Q' to stop)")
        control_recording(True)
        # exit rules
        while True:  # Keep main loop alive
            if keyboard.is_pressed("q") or GUI_QUIT_Q.qsize():  # "q" --> Key press
                control_recording(False)   # end recording
                sleep(1)                   # time for final processings
                print(f"Recording Finished ;) at {datetime.datetime.now()}") 
                break

    # Save Data    
    if SAVE_FFT_DATA or SAVE_REF_DATA:     
        RECORDED_DATA = {}   # Data holder
        if USE_RADAR and SAVE_FFT_DATA:  # save radar data
            print("\nSaving radar data ...")
            save_1dfft(data_dict = RECORDED_DATA, save_fft_array_q=save_fft_array_q)
            print("FFT saving finished :)")
        
        if SAVE_REF_DATA:  # save reference data
            print("Saving ground truth data ...")

            save_ref_data(
                data_dict = RECORDED_DATA,
                polar_acc_ref_q=POLAR_ACC_DATA, 
                polar_ecg_ref_q=POLAR_ECG_DATA, 
                polar_hr_ref_q=POLAR_HR_DATA, 
                polar_rri_ref_q=POLAR_RRI_DATA, 
                vernier_belt_ref_q=VERNIER_BELT_FORCE_DATA, 
                vernier_spiro_ref_q=VERNIER_SPIRO_FORCE_DATA, 
                m5stickc_acc1_ref_q = M5STICKC_ACC1_DATA, 
                m5stickc_acc2_ref_q = M5STICKC_ACC2_DATA,
                m5stickc_marker_ref_q = M5STICKC_MARKER_DATA,
                witmotion1_ref_data_q = WITMOTION1_DATA,
                witmotion2_ref_data_q = WITMOTION2_DATA,
                hat_ref_q = HAT_DATA
            )
        
        if SAVE_FFT_DATA or SAVE_REF_DATA:
            save_to_excel(RECORDED_DATA)

    # Check video recording:
    if RECORD_VIDEO:
        while 1:
            if START_VIDEO_FLAG.get():
                break

    sleep(2)
    print("Good Bye :)")
        




