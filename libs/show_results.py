from sys_config import *
from user_config import *
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os

from libs.utils import get_Spectrum, signal_padding
from libs.data_processing.respiration import cal_time # add by nakajima


matplotlib.use('TkAgg')
plt.rcParams["figure.figsize"] = MATPLOTLIB_FIGURE_SIZE
plt.rcParams["font.family"] = MATPLOTLIB_PARAM["font_family"]
plt.rcParams.update({'font.size': MATPLOTLIB_PARAM["font_size"]})
plt.rcParams['axes.xmargin'] = 0
props = dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.15)  # bbox features

def matplotlib_plot(result_dict=None):
    plt.figure('Realtime Monitoring Results')

    for sub in range(1, N_SUBPLOTS+1):
        plt.subplot(N_SUBPLOTS, 1, sub)
        plt.cla()
        plt.ion()
        for sig in PLOTS:
            if PLOTS[sig][0] and PLOTS[sig][1]==sub:
                plot_signal(sig, result_dict)
        plt.legend(loc='upper left')
    # plt.subplots_adjust(right=0.70)
    plt.tight_layout()
    plt.show()
    plt.pause(0.2)

def plot_signal(signal, result_dict):
    if signal == "PLOT_RTM":
        plt.title('Radar Range Heatmap')
        plt.xlabel('Time (sec)')
        plt.ylabel('Range (m)')
        plt.imshow(result_dict["magn_data"].T, cmap='viridis', vmax=RTM_MAX_MAG, aspect='auto', origin="lower", extent=[-PLOTTING_WIN, 0,  0, MAX_DISTANCE*RANGE_RESOLUTION])
    
    elif signal == "PLOT_RAM":
        plt.title('Radar Range-Angle Heatmap')
        plt.xlabel('Angle (deg)')
        plt.ylabel('Range (m)')
        plt.imshow(result_dict["angle_info"][0].T, cmap='viridis', vmax=RAM_MAX_MAG, aspect='auto', origin="lower", extent=[np.min(SEARCHING_AZIMUTH_DOA_RANGE), np.max(SEARCHING_AZIMUTH_DOA_RANGE), 0, MAX_DISTANCE*RANGE_RESOLUTION])
    
    elif signal == "PLOT_MAGNITUDE":
        plt.xlabel('Time (sec)')
        plt.ylabel('Magnitude')
        if result_dict["presence_flag"][-1]:
            plt.title(f'Magnitude Signal at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            plt.plot(result_dict["x_radar_array"], result_dict["magn_data"][:, result_dict["target_bin"][-1]])
        else:
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])))
            plt.title(f"Magnitude Signal: No Subject Detected")
        
    elif signal == "PLOT_PHASE":
        plt.ylim(0, 1)
        plt.xlabel('Time (sec)')
        # plt.ylabel('Angle')
        plt.ylabel('ABs&Norm')
        plt.grid(linestyle=':')
        if result_dict["presence_flag"][-1]:
            # plt.plot(result_dict["x_radar_array"], result_dict["target_data"])
            # plt.title(f"Unwrapped Phase Signal at {round(result_dict['target_bin'][-1][0]*RANGE_RESOLUTION,2)} m")

            plt.title(f'ABS & Normalization of Respiration Signal')
            plt.plot(result_dict["x_radar_array"][1:], result_dict["respiration_derivative"], 'b', label="ABS&Norm")
            plt.hlines(0.05, -30, 0, color='r', label='threshold')
        else:
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])))
            plt.title(f"Unwrapped Phase Signal: No Subject Detected")
    
    elif signal == "PLOT_RESP_WAVE":
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.grid(linestyle=':')
        plt.ylim(-30, 30)
        # plt.ylim(0, 1)
        
        if result_dict["presence_flag"][-1]: # target in range
            plt.title(f'Respiration Signal: Target at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            plt.plot(result_dict["x_radar_array"][1:], result_dict["respiration_signal"], 'r', label="Radar")
            # plt.plot(result_dict["x_radar_array"][1:], result_dict["respiration_derivative"], 'b', label="Radar")

            # Highlight Hold period in plot:
            if len(result_dict["respiration_pattern"])>0:
                for pattern in result_dict["respiration_pattern"]:
                    plt.axvspan(result_dict["x_radar_array"][pattern[0]], result_dict["x_radar_array"][pattern[1]], facecolor='red', alpha=0.2)
                    # plt.text(result_dict["x_radar_array"][pattern[0]]+(result_dict["x_radar_array"][pattern[1]]-result_dict["x_radar_array"][pattern[0]])//2, 20, 'HOLD')
        else:  # No one is Here !!!
            plt.title(f'Respiration Signal: No Detection !!!')
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])), 'r', label="Radar")
    
    elif signal == "PLOT_HEARTBEAT_WAVE":
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        if result_dict["presence_flag"][-1]:
            plt.title(f'Heartbeat Signal: Target at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            plt.plot(result_dict["x_radar_array"][1:], result_dict["heartbeat_signal"], 'r', label="Radar")

        else:  # No one is Here !!!
            plt.title(f'Heartbeat Signal: No Detection !!!')
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])), 'r', label="Radar")
    
    elif signal == "PLOT_IQ":
        plt.title(f'I-Q signal at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')

        plt.plot(result_dict["x_radar_array"], np.real(result_dict['1dfft_IQ_data']), 'b', label="(I)")
        plt.plot(result_dict["x_radar_array"], np.imag(result_dict['1dfft_IQ_data']), 'r', label="(Q)")

    elif signal == "SCATTER_IQ":
        plt.title(f'I-Q signal at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.ylim(-2000, 2000)
        plt.xlim(-2000, 2000)
        plt.scatter(np.real(result_dict['1dfft_IQ_data'])[-FPS*5:], np.imag(result_dict['1dfft_IQ_data'][-FPS*5:]))
    
    elif signal == "PLOT_VERNIER_BELT_FORCE":
        data_size = True
        plt.ylim(-10, 10)
        # check if correct amount of data otherwise perform padding
        if len(result_dict['vernier_belt_new_data'])<VERNIER_BELT_FPS*PLOTTING_WIN:
            required_samples = VERNIER_BELT_FPS*PLOTTING_WIN - len(result_dict['vernier_belt_new_data'])
            result_dict['vernier_belt_new_data'] = signal_padding(result_dict['vernier_belt_new_data'], length=required_samples)
        if len(result_dict['processed_vernier_new_data'])<VERNIER_BELT_FPS*PLOTTING_WIN:
            required_samples = VERNIER_BELT_FPS*PLOTTING_WIN - len(result_dict['processed_vernier_new_data'])
            result_dict['processed_vernier_new_data'] = signal_padding(result_dict['processed_vernier_new_data'], length=required_samples)
            data_size = False


        # plot data
        # plt.plot(result_dict["x_vernier_belt_array"], result_dict["vernier_belt_new_data"], 'g', label="Vernier Belt") #not processed data
        plt.plot(result_dict["x_vernier_belt_array"], result_dict["processed_vernier_new_data"], 'g', label="Vernier")
        plt.plot(result_dict["x_vernier_thresh_array"], result_dict["vernier_thresh"], 'c', label='threshold')
        # plt.plot(result_dict["x_vernier_mean_array"], result_dict["vernier_mean_list"], 'r', label='mean') # for check mean value
        plt.plot(result_dict["x_vernier_std_array"], result_dict["vernier_std"], 'm', label='std_w5')

        # if data_size:
        #     for cp in result_dict["vernier_change_point"]:
        #         plt.axvline(cp, color='r', linestyle='dashed', linewidth=1)

        if len(result_dict["vernier_respiration_pattern"])>0:
            for pattern in result_dict["vernier_respiration_pattern"]:
                # print(pattern)
                p1 = result_dict["x_vernier_belt_array"][pattern[0]]
                p2 = result_dict["x_vernier_belt_array"][pattern[1]]
                # print(f"p1={p1}, p2={p2}")
                plt.axvspan(p1, p2, facecolor='red', alpha=0.2)
                # plt.text(p1, -2, cal_time(pattern[0], fs = VERNIER_BELT_FPS), c='g')
                # plt.text(p2, -2, cal_time(pattern[1], fs = VERNIER_BELT_FPS), c='g')
                plt.text(p1, -2, str(p1), c='g')
                plt.text(p2, -2, str(p2), c='g')

            
    elif signal == "PLOT_POLAR_H10_ECG":
        # check if correct amount of data otherwise perform padding
        if len(result_dict['polar_ecg_new_data'])<POLAR_ECG_FPS*PLOTTING_WIN:
            required_samples = POLAR_ECG_FPS*PLOTTING_WIN - len(result_dict['polar_ecg_new_data'])
            result_dict['polar_ecg_new_data'] = signal_padding(result_dict['polar_ecg_new_data'], length=required_samples)

        plt.plot(result_dict["x_polar_ecg_array"], POLAR_ECG_SCALE * np.asarray(result_dict['polar_ecg_new_data'], dtype=float), 'g', label="Polar ECG")

    elif signal == "PLOT_MARKER":
        # check if correct amount of data otherwise perform padding
        if len(result_dict['marker_new_data'])<M5STICKC_MARKER_FPS*PLOTTING_WIN:
            required_samples = M5STICKC_MARKER_FPS*PLOTTING_WIN - len(result_dict['marker_new_data'])
            result_dict['marker_new_data'] = signal_padding(result_dict['marker_new_data'], padding_value=48 ,length=required_samples)
        
        plt.plot(result_dict["x_marker_array"], M5STICKC_MARKER_SCALER*(np.asarray(result_dict['marker_new_data'], dtype=int)-48), 'c', label="Marker")
        
        timestamp = np.where(np.asarray(result_dict['marker_new_data']) == 49)[0]
        if np.size(timestamp) != 0:
            for point in timestamp:
                # time = cal_time(point, fs = M5STICKC_MARKER_FPS)
                # plt.text(result_dict["x_marker_array"][point], -1, time, c='c')
                plt.text(result_dict["x_marker_array"][point], -1, str(result_dict["x_marker_array"][point]), c='c')


    elif signal == "PLOT_WITMOTION_1":
        required_samples = 0
        if len(result_dict['witmotion1_new_data'])<WITMOTION1_FPS*PLOTTING_WIN:
            required_samples = WITMOTION1_FPS*PLOTTING_WIN - len(result_dict['witmotion1_new_data'])

        if WITMOTION1_CHANNEL == "ACC":
            plt.xlabel('Time (sec)')
            plt.ylabel('Acceleration (g)')

            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_accX_data = signal_padding(result_dict['witmotion1_new_data'][:,0], length=required_samples)
            else:
                wit1_accX_data = result_dict['witmotion1_new_data'][:,0]
            plt.plot(result_dict["x_witmotion1_array"], wit1_accX_data, 'b', label=f"Wit-{WITMOTION1_LOCATION}-Acc_X")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_accY_data = signal_padding(result_dict['witmotion1_new_data'][:,1], length=required_samples)
            else:
                wit1_accY_data = result_dict['witmotion1_new_data'][:,1]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_accY_data, 'g', label=f"Wit-{WITMOTION1_LOCATION}-Acc_Y")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_accZ_data = signal_padding(result_dict['witmotion1_new_data'][:,2], length=required_samples)
            else:
                wit1_accZ_data = result_dict['witmotion1_new_data'][:,2]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_accZ_data, 'r', label=f"Wit-{WITMOTION1_LOCATION}-Acc_Z")

        elif WITMOTION1_CHANNEL == "GYR":
            plt.xlabel('Time (sec)')
            plt.ylabel('Gyro (deg/sec)')

            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_gyrX_data = signal_padding(result_dict['witmotion1_new_data'][:,3], length=required_samples)
            else:
                wit1_gyrX_data = result_dict['witmotion1_new_data'][:,3]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_gyrX_data, 'b', label=f"Wit-{WITMOTION1_LOCATION}-Gyr_X")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_gyrY_data = signal_padding(result_dict['witmotion1_new_data'][:,4], length=required_samples)
            else:
                wit1_gyrY_data = result_dict['witmotion1_new_data'][:,4]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_gyrY_data, 'g', label=f"Wit-{WITMOTION1_LOCATION}-Gyr_Y")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_gyrZ_data = signal_padding(result_dict['witmotion1_new_data'][:,5], length=required_samples)
            else:
                wit1_gyrZ_data = result_dict['witmotion1_new_data'][:,5]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_gyrZ_data, 'r', label=f"Wit-{WITMOTION1_LOCATION}-Gyr_Z")

        elif WITMOTION1_CHANNEL == "ANG":
            plt.xlabel('Time (sec)')
            plt.ylabel('Angle (deg)')
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_angX_data = signal_padding(result_dict['witmotion1_new_data'][:,6], length=required_samples)
            else:
                wit1_angX_data = result_dict['witmotion1_new_data'][:,6]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_angX_data, 'b', label=f"Wit-{WITMOTION1_LOCATION}-Ang_X")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_angY_data = signal_padding(result_dict['witmotion1_new_data'][:,7], length=required_samples)
            else:
                wit1_angY_data = result_dict['witmotion1_new_data'][:,7]

            plt.plot(result_dict["x_witmotion1_array"], wit1_angY_data, 'g', label=f"Wit-{WITMOTION1_LOCATION}-Ang_Y")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit1_angZ_data = signal_padding(result_dict['witmotion1_new_data'][:,8], length=required_samples)
            else:
                wit1_angZ_data = result_dict['witmotion1_new_data'][:,8]
            
            plt.plot(result_dict["x_witmotion1_array"], wit1_angZ_data, 'r', label=f"Wit-{WITMOTION1_LOCATION}-Ang_Z")

    elif signal == "PLOT_WITMOTION_2":
        required_samples = 0
        if len(result_dict['witmotion2_new_data'])<WITMOTION2_FPS*PLOTTING_WIN:
            required_samples = WITMOTION2_FPS*PLOTTING_WIN - len(result_dict['witmotion2_new_data'])

        if WITMOTION1_CHANNEL == "ACC":
            plt.xlabel('Time (sec)')
            plt.ylabel('Acceleration (g)')
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_accX_data = signal_padding(result_dict['witmotion2_new_data'][:,0], length=required_samples)
            else:
                wit2_accX_data = result_dict['witmotion2_new_data'][:,0]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_accX_data, 'b', label=f"Wit-{WITMOTION2_LOCATION}-Acc_X")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_accY_data = signal_padding(result_dict['witmotion2_new_data'][:,1], length=required_samples)
            else:
                wit2_accY_data = result_dict['witmotion2_new_data'][:,1]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_accY_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Acc_Y")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_accZ_data = signal_padding(result_dict['witmotion2_new_data'][:,2], length=required_samples)
            else:
                wit2_accZ_data = result_dict['witmotion2_new_data'][:,2]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_accZ_data, 'r', label=f"Wit-{WITMOTION2_LOCATION}-Acc_Z")

        elif WITMOTION1_CHANNEL == "GYR":
            plt.xlabel('Time (sec)')
            plt.ylabel('Gyro (deg/sec)')
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_gyrX_data = signal_padding(result_dict['witmotion2_new_data'][:,3], length=required_samples)
            else:
                wit2_gyrX_data = result_dict['witmotion2_new_data'][:,3]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_gyrX_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Gyr_X")
            
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_gyrY_data = signal_padding(result_dict['witmotion2_new_data'][:,4], length=required_samples)
            else:
                wit2_gyrY_data = result_dict['witmotion2_new_data'][:,4]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_gyrY_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Gyr_Y")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_gyrZ_data = signal_padding(result_dict['witmotion2_new_data'][:,5], length=required_samples)
            else:
                wit2_gyrZ_data = result_dict['witmotion2_new_data'][:,5]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_gyrZ_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Gyr_Z")

        elif WITMOTION1_CHANNEL == "ANG":
            plt.xlabel('Time (sec)')
            plt.ylabel('Angle (deg)')
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_angX_data = signal_padding(result_dict['witmotion2_new_data'][:,6], length=required_samples)
            else:
                wit2_angX_data = result_dict['witmotion2_new_data'][:,6]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_angX_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Ang_X")
            
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_angY_data = signal_padding(result_dict['witmotion2_new_data'][:,7], length=required_samples)
            else:
                wit2_angY_data = result_dict['witmotion2_new_data'][:,7]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_angY_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Ang_Y")
            
            # check if correct amount of data otherwise perform padding
            if required_samples:
                wit2_angZ_data = signal_padding(result_dict['witmotion2_new_data'][:,8], length=required_samples)
            else:
                wit2_angZ_data = result_dict['witmotion2_new_data'][:,8]
            
            plt.plot(result_dict["x_witmotion2_array"], wit2_angZ_data, 'g', label=f"Wit-{WITMOTION2_LOCATION}-Ang_Z")
    
    elif signal == "PLOT_HAT_EOG":
        if len(result_dict['hat_new_data'])<HAT_FPS*PLOTTING_WIN:
            required_samples = HAT_FPS*PLOTTING_WIN - len(result_dict['hat_new_data'])
            eog_data = signal_padding(result_dict["hat_new_data"][:,-1], length=required_samples)
        else:
            eog_data = result_dict["hat_new_data"][:,-1]
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.plot(result_dict["x_hat_array"], eog_data, label="EOG")

    elif signal == "PLOT_HAT_RAW_EYE":
        if len(result_dict['hat_new_data'])<HAT_FPS*PLOTTING_WIN:
            required_samples = HAT_FPS*PLOTTING_WIN - len(result_dict['hat_new_data'])
            eog_raw_data = signal_padding(result_dict["hat_new_data"][:,-2], length=required_samples)
        else:
            eog_raw_data = result_dict["hat_new_data"][:,-2]

        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.plot(result_dict["x_hat_array"], eog_raw_data, label="RAW EOG")

    elif signal == "PLOT_HAT_ACC":
        if len(result_dict['hat_new_data'])<HAT_FPS*PLOTTING_WIN:
            required_samples = HAT_FPS*PLOTTING_WIN - len(result_dict['hat_new_data'])
            hat_x_data = signal_padding(result_dict["hat_new_data"][:,0], length=required_samples)
            hat_y_data = signal_padding(result_dict["hat_new_data"][:,1], length=required_samples)
            hat_z_data = signal_padding(result_dict["hat_new_data"][:,2], length=required_samples)
        else:
            hat_x_data = result_dict["hat_new_data"][:,0]
            hat_y_data = result_dict["hat_new_data"][:,1]
            hat_z_data = result_dict["hat_new_data"][:,2]
        plt.xlabel('Time (sec)')
        plt.ylabel('Acceleration (g)')
        plt.plot(result_dict["x_hat_array"], hat_x_data, 'b', label="Hat-AccX")
        plt.plot(result_dict["x_hat_array"], hat_y_data, 'g', label="Hat-AccX")
        plt.plot(result_dict["x_hat_array"], hat_z_data, 'r', label="Hat-AccX")


























##########################################################
############### Other Plotting tests ###################

def matplotlib_plot1(result_dict=None):
    plt.figure('Realtime Monitoring Results')
    # 1/ Range-Heatmap
    if PLOTS["PLOT_RTM"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_RTM"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_title('Radar Range Heatmap')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Range (m)')

        plt.imshow(result_dict["magn_data"].T, cmap='viridis', aspect='auto', origin="lower", extent=[-PLOTTING_WIN, 0,  0, MAX_DISTANCE*RANGE_RESOLUTION])

        if result_dict["presence_flag"][-1]:
            boxtext = f"Radar Used: {RADAR_DEVICE} \n\nFrequency: {START_FREQUENCY} Hz \n\nSampling Frequency: {FPS} Hz \n\nRange Resolution: {RANGE_RESOLUTION*100} cm \n\nCurrent Bin: {result_dict['target_bin'][-1][0]} \n\nCurrent Power: {round(np.max(result_dict['magn_data']),2)}"
        else:
            boxtext = f"Radar Used: {RADAR_DEVICE} \n\nFrequency: {START_FREQUENCY} Hz \n\nSampling Frequency: {FPS} Hz \n\nRange Resolution: {RANGE_RESOLUTION*100} cm \n\nCurrent Bin: N/A \nCurrent Power: {round(np.max(result_dict['magn_data']),2)}"

        an = ax.text(1.03, 0.98, boxtext, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)        

    if PLOTS["PLOT_RAM"][0]:
        if VERSION == "8_ANT":
            plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_RAM"][1])
            plt.cla()
            plt.ion()
            ax = plt.gca()
            ax.set_title('Radar Range-Angle Heatmap')
            ax.set_xlabel('Angle (deg)')
            ax.set_ylabel('Range (m)')
            plt.imshow(result_dict["angle_info"][0].T, cmap='viridis', norm = matplotlib.colors.Normalize(), aspect='auto', origin="lower", extent=[np.min(SEARCHING_AZIMUTH_DOA_RANGE), np.max(SEARCHING_AZIMUTH_DOA_RANGE), 0, MAX_DISTANCE*RANGE_RESOLUTION])
        else: 
            print("Number of antenna used is not sufficient for RAM plot ...")

    if PLOTS["PLOT_PHASE"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_PHASE"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Angle')
        plt.grid(linestyle=':')
        if result_dict["presence_flag"][-1]:
            plt.plot(result_dict["x_radar_array"], result_dict["target_data"])
            ax.set_title(f"Unwrapped Phase Signal at {round(result_dict['target_bin'][-1][0]*RANGE_RESOLUTION,2)} m")
        else:
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])))
            ax.set_title(f"Unwrapped Phase Signal: No Subject Detected")

    if PLOTS["PLOT_MAGNITUDE"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_MAGNITUDE"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Magnitude')
        if result_dict["presence_flag"][-1]:
            ax.set_title(f'Magnitude Signal at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            plt.plot(result_dict["x_radar_array"], result_dict["magn_data"][:, result_dict["target_bin"][-1]])
        else:
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])))
            ax.set_title(f"Magnitude Signal: No Subject Detected")

    if PLOTS["PLOT_RESP_WAVE"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_RESP_WAVE"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-30, 30)
        
        if result_dict["presence_flag"][-1]:
            ax.set_title(f'Respiration Signal: Target at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            plt.plot(result_dict["x_radar_array"][1:], result_dict["respiration_signal"], 'r', label="Radar")

            # Highlight Hold period in plot:
            if len(result_dict["respiration_pattern"])>0:
                for pattern in result_dict["respiration_pattern"]:
                    ax.axvspan(result_dict["x_radar_array"][pattern[0]], result_dict["x_radar_array"][pattern[1]], facecolor='red', alpha=0.2)
                    ax.text(result_dict["x_radar_array"][pattern[0]]+(result_dict["x_radar_array"][pattern[1]]-result_dict["x_radar_array"][pattern[0]])//2, 20, 'HOLD')
            
            an = ax.text(1.03, 0.68, 'Pattern: ', transform=ax.transAxes, verticalalignment='top', bbox=props)
            an.set_in_layout(False)

            respiration_bpm_fft_textstr = f'Current RR: {result_dict["respiration_bpm_fft"][-1]} bpm'
            an = ax.text(1.03, 0.98, respiration_bpm_fft_textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
            an.set_in_layout(False)

        else:  # No one is Here !!!
            ax.set_title(f'Respiration Signal: No Detection !!!')
            plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])), 'r', label="Radar")
                            
            respiration_bpm_fft_textstr = 'Current RR: N/A'
            an = ax.text(1.03, 0.98, respiration_bpm_fft_textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
            an.set_in_layout(False)

            respiration_pattern_value = f'N/A'
            color = 'grey'
            an = ax.text(1.03, 0.68, respiration_pattern_value, color=color, transform=ax.transAxes, verticalalignment='top', bbox=props)
            an.set_in_layout(False)
        
        try:
            plt.legend(loc='upper left')
        except: pass

    if PLOTS["PLOT_VERNIER_FORCE"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_VERNIER_FORCE"][1])
        if USE_VERNIER:
            plt.plot(result_dict["x_vernier_array"], result_dict["vernier_new_data"], 'g', label="Vernier")
    
    if PLOTS["PLOT_HEARTBEAT_WAVE"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_HEARTBEAT_WAVE"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-10, 10)
        plt.grid(linestyle=':')

        if result_dict["presence_flag"][-1]:
            ax.set_title(f'Heartbeat Signal: Target at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
            
            plt.plot(result_dict["x_radar_array"][1:], result_dict["heartbeat_signal"], 'r', label="Radar")
            heartbeat_bpm_fft_textstr = f'Current HR: {result_dict["heartbeat_bpm_fft"][-1]} bpm'
            an = ax.text(1.03, 0.98, heartbeat_bpm_fft_textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
            an.set_in_layout(False)

        else:  # No one is Here !!!
            try:
                ax.set_title(f'Heartbeat Signal: No Detection !!!')
                plt.plot(result_dict["x_radar_array"], np.zeros(len(result_dict["x_radar_array"])), 'r', label="Radar")

            except: pass
        try:
            plt.legend(loc='upper left')
        except: pass
    
    if PLOTS["PLOT_POLAR_H10_ECG"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_POLAR_H10_ECG"][1])
        if USE_POLAR:
            if POLAR_ECG_DATA_FLAG:
                plt.plot(result_dict["x_polar_ecg_array"], POLAR_ECG_SCALE * result_dict["polar_ecg_new_data"], 'g', label="Polar ECG")
            if POLAR_HR_DATA_FLAG:
                an = ax.text(1.03, 0.78, f"Polar HR: {np.round(np.mean(result_dict['polar_hr_new_data'][-1]),2)} bpm", transform=ax.transAxes, verticalalignment='top', bbox=props)
                an.set_in_layout(False)           
    
    if PLOTS["PLOT_IQ"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_IQ"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_title(f'I-Q signal at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude')

        plt.plot(result_dict["x_radar_array"], np.real(result_dict['1dfft_IQ_data']), 'b', label="(I)")
        plt.plot(result_dict["x_radar_array"], np.imag(result_dict['1dfft_IQ_data']), 'r', label="(Q)")

        plt.legend(loc='upper left')#, bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)

        presence_box_label = "Presence: "
        an = ax.text(1.03, -0.9, presence_box_label, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)

        presence_box_state = "Subject" if result_dict["presence_flag"][-1] else "No One here !!!"
        color = "green" if result_dict["presence_flag"][-1] else "red"
        an = ax.text(1.2, -0.9, presence_box_state, color=color, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)

        macro_motion_box_label = f"Macro Motion: "
        color = "black"
        an = ax.text(1.03, -1.2, macro_motion_box_label, color = color, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)

        if result_dict["presence_flag"][-1]:
            motion_box_state = result_dict["Motion_State_bin"]
            if motion_box_state == "Static":
                color = "green"
            else:
                color = "red"
        else:
            motion_box_state = "N/A"
            color = "grey"
        an = ax.text(1.2, -1.2, motion_box_state, color = color, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)

        motion_box_label = f"Micro Motion:"
        color = "black"
        an = ax.text(1.03, -1.5, motion_box_label, color = color, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)

        if result_dict["presence_flag"][-1]:
            motion_box_state = "TBD"
            if motion_box_state == "IDLE":
                color = "green"
            else:
                color = "red"
        else:
            motion_box_state = "N/A"
            color = "grey"
        an = ax.text(1.2, -1.5, motion_box_state, color = color, transform=ax.transAxes, verticalalignment='top', bbox=props)
        an.set_in_layout(False)


    if PLOTS["SCATTER_IQ"][0]:
        plt.subplot(N_SUBPLOTS,1,PLOTS["SCATTER_IQ"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_title(f'I-Q signal at {round(result_dict["target_bin"][-1][0]*RANGE_RESOLUTION,2)} m')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_ylim(-2000, 2000)
        ax.set_xlim(-2000, 2000)
        ax.scatter(np.real(result_dict['1dfft_IQ_data'])[-FPS*5:], np.imag(result_dict['1dfft_IQ_data'][-FPS*5:]))

    if PLOTS["PLOT_MARKER"][0]:
        # check if correct amount of data otherwise perform padding
        if len(result_dict['marker_new_data'])<M5STICKC_MARKER_FPS*PLOTTING_WIN:
            required_samples = M5STICKC_MARKER_FPS*PLOTTING_WIN - len(result_dict['marker_new_data'])
            print(required_samples)
            result_dict['marker_new_data'] = signal_padding(result_dict['marker_new_data'], padding_value=48 ,length=required_samples)

        plt.subplot(N_SUBPLOTS,1,PLOTS["PLOT_MARKER"][1])
        plt.cla()
        plt.ion()
        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.plot(result_dict["x_marker_array"], np.asarray(result_dict['marker_new_data'], dtype=int)-48, 'c', label="Marker")
        ax.legend()

    plt.tight_layout()
    # plt.subplots_adjust(right=0.70)
    plt.show()
    plt.pause(0.2)





def matplotlib_plot2(result_dict=None):
    
    result_dict["x_radar_array"] = np.linspace(-PLOTTING_WIN, 0, FPS*PLOTTING_WIN)
    plt.figure('Realtime Monitoring Results')
    
    # 1/ Range-Heatmap
    # plt.subplot(411)
    # plt.cla()
    # plt.ion()
    # ax = plt.gca()
    # ax.set_title('Radar Range Heatmap')
    # ax.set_xlabel('Range (m)')
    # ax.set_ylabel('Time (sec)')

    # plt.imshow(result_dict["magn_data"], cmap='viridis', aspect='auto', extent=[0, MAX_DISTANCE*RANGE_RESOLUTION, 0, -PLOTTING_WIN])


    # 2/ Bin Power
    plt.subplot(412)
    plt.cla()
    plt.ion()
    ax = plt.gca()
    ax.set_title(f'Avg. power change per bin for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE}')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Power/Magnitude')
    for bin in range(result_dict["energy_array"].shape[1]):
        plt.plot(np.linspace(-PLOTTING_WIN, 0, PLOTTING_WIN), result_dict["energy_array"][:, bin][-PLOTTING_WIN:], label=f"bin {bin+PRESENCE_DETECTION_RANGE[0]}")
   
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=result_dict["energy_array"].shape[1])


    # 3/ Magnitude
    plt.subplot(413)
    plt.cla()
    plt.ion()
    ax = plt.gca()
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Magnitude Signal for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE}')
    plt.plot(result_dict["x_radar_array"], result_dict["magn_data"][:, result_dict["target_bin"][-1]])

    
    # 4/ Phase:
    plt.subplot(414)
    plt.cla()
    plt.ion()
    ax = plt.gca()
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Angle')
    plt.grid(linestyle=':')
    plt.plot(result_dict["x_radar_array"], result_dict["target_data"])
    ax.set_title(f"Unwrapped Phase Signal for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE} at bin '{result_dict['target_bin'][-1]}'")

    plt.tight_layout()
    # plt.subplots_adjust(right=0.70)
    try:
        plt.legend()
    except: pass
    plt.show()
    # print("passed heartbeat")
    
    # elapsed = time.time() - t
    # print(f"refresh time: {elapsed}")
    plt.pause(0.2)
    

def test_results(result_dict=None):
    """
    Used for testing/analysis purposes only 
    Save the plot of the last "window" of data
    """
    plt.rcParams["figure.figsize"] = (15,10)
    # Check if end of recording

    # Time axis:
    result_dict["x_radar_array"] = np.linspace(-PLOTTING_WIN, 0, PLOTTING_WIN*FPS)

    
    fig = plt.figure()

    # Plot Heatmap
    plt.subplot(411)
    
    ax = plt.gca()
    ax.set_title('Radar Range Heatmap')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Time (sec)')
    plt.imshow(result_dict["magn_data"], cmap='viridis', aspect='auto', extent=[0, MAX_DISTANCE*RANGE_RESOLUTION, 0, -PLOTTING_WIN])

    # Plot Bin Power selection and Power change
    plt.subplot(412)
    for bin in range(result_dict["energy_array"].shape[1]):
        plt.plot(np.linspace(-PLOTTING_WIN, 0, PLOTTING_WIN), result_dict["energy_array"][:, bin][-PLOTTING_WIN:], label=f"bin {bin+PRESENCE_DETECTION_RANGE[0]}")
    plt.xlabel("Samples")
    plt.ylabel("Power/Magnitude")
    plt.title(f"Avg. power change per bin for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE}")    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=result_dict["energy_array"].shape[1])

    # Plot Magnitude signal at bin
    plt.subplot(413)
    plt.plot(result_dict["x_radar_array"], result_dict["magn_data"][:, result_dict["target_bin"]])
    plt.xlabel("Time (sec)")
    plt.ylabel("Power/Magnitude")
    plt.title(f"Magnitude Signal for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE} at bin '{result_dict['target_bin']}'")    

    # Plot Phase signal at current bin
    plt.subplot(414)
    plt.plot(result_dict["x_radar_array"], result_dict["target_data"])
    plt.xlabel("Time (sec)")
    plt.ylabel("Angle")
    plt.title(f"Unwrapped Phase Signal for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE} at bin '{result_dict['target_bin']}'")    

    plt.tight_layout()

    if not os.path.exists(LOCAL_CACHE_PATH):
        os.makedirs(LOCAL_CACHE_PATH)
    fig.savefig(f"{LOCAL_CACHE_PATH}{OUT_FILE}.png")