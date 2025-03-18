import numpy as np
from libs.utils import clutter_remove, WisProcess, unwrap_phase_Ambiguity, fft_find_bpm, set_presence_threshold 
from libs.data_processing.presence_localization import presence_detection_handler, locate_subject, get_angle_info
from libs.data_processing.respiration import extract_respiration, estimate_respiration_pattern, vernier_proccesing, calculate_std, get_change_point, estimate_respiration_pattern_thresh
from libs.data_processing.heartbeat import extract_heartbeat
from libs.data_processing.motion_tracking import motion_detection_micro, motion_detection_macro
from libs.show_results import matplotlib_plot, test_results, matplotlib_plot2
from sys_config import *
from user_config import *

import matplotlib.pyplot as plt

def plot_bin_power_history(data):
    """
    Evaluation purposes
    """
    fig = plt.figure()
    sample_axis = np.linspace(0, len(data), len(data))
    for bin in range(data.shape[1]):
        plt.plot(sample_axis, data[:, bin], label=f"bin {bin+PRESENCE_DETECTION_RANGE[0]}")
    plt.xlabel("Samples")
    plt.ylabel("Power/Magnitude")
    plt.title(f"Avg. power change per bin for {SUBJECT}_{EXPERIMENT_CONDITION}_{RADAR_DEVICE}")    
    plt.legend()
    fig.savefig(f"results/{OUT_FILE}_power_bin.png")

class RadarProcessor(WisProcess):
    def __init__(self, in_q=None, out_q=None, logger=None, calculation_status=CalculationStatus.free.value, **kwargs):
        super().__init__(in_q, out_q, logger)
        self._cal_timer = None
        self.out_range_profile_q = kwargs.get('out_range_profile_q')
        self.calculation_status = calculation_status
        self.range_data_q = kwargs.get('range_data_q')
        self.phase_data_q = kwargs.get('phase_data_q')
        self.vernier_belt_realtime_q = kwargs.get('vernier_belt_realtime_q')
        self.witmotion1_realtime_q = kwargs.get('witmotion1_realtime_q')
        self.witmotion2_realtime_q = kwargs.get('witmotion2_realtime_q')
        self.polar_ecg_realtime_q = kwargs.get('polar_ecg_realtime_q')
        self.polar_hr_realtime_q = kwargs.get('polar_hr_realtime_q')
        self.calibration_flag = kwargs.get("calibration_flag")
        self.results_dict_q = kwargs.get("results_dict_q")
        self.m5stickc_marker_realtime_q = kwargs.get("m5stickc_marker_realtime_q")
        self.hat_realtime_q = kwargs.get("hat_realtime_q")

    def run(self):
        processor = RadarDataProcessor(
            range_data_q = self.range_data_q, 
            phase_data_q = self.phase_data_q, 
            vernier_belt_realtime = self.vernier_belt_realtime_q, 
            witmotion1_realtime = self.witmotion1_realtime_q,
            witmotion2_realtime = self.witmotion2_realtime_q,
            polar_hr_realtime_q = self.polar_hr_realtime_q,
            polar_ecg_realtime_q = self.polar_ecg_realtime_q,
            calibration_flag = self.calibration_flag,
            results_dict_q = self.results_dict_q,
            m5stickc_marker_realtime_q = self.m5stickc_marker_realtime_q,
            hat_realtime_q = self.hat_realtime_q

        )
        while True:
            if USE_RADAR:
                try:
                    dataset = self._in_q.get()
                    self.calculation_status.value = CalculationStatus.busy.value   # ? --> Related to buffer read (Raw data)
                    processor.slide_range_matrix_window(dataset)
                    processor.calculate()
                    self.calculation_status.value = CalculationStatus.free.value  # ? --> Related to buffer read (Raw data)
                except Exception as e: print(e)
            else:
                try:
                    processor.prepare_plot_data()
                except: pass
class RadarDataProcessor:
    RangeMatrixQueueLen = TI_1DFFT_QUEUE_LEN

    def __init__(self, calculation_status=CalculationStatus.free.value, **kwargs):
        self.range_matrix_queue = np.zeros((0, RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)
        self.calculation_status = calculation_status
        self.calibration_flag = kwargs.get("calibration_flag")     
        self.range_data_q = kwargs.get('range_data_q')
        self.phase_data_q = kwargs.get('phase_data_q')
        self.vernier_belt_realtime_q = kwargs.get('vernier_belt_realtime')
        self.witmotion1_realtime_q = kwargs.get('witmotion1_realtime')
        self.witmotion2_realtime_q = kwargs.get('witmotion2_realtime')
        self.polar_ecg_realtime_q = kwargs.get('polar_ecg_realtime_q')
        self.polar_hr_realtime_q = kwargs.get('polar_hr_realtime_q')
        self.hat_realtime_q = kwargs.get('hat_realtime_q')
        self.results_dict_q = kwargs.get("results_dict_q")
        self.m5stickc_marker_realtime_q = kwargs.get("m5stickc_marker_realtime_q")
        self.vernier_belt_new_data = []
        self.polar_ecg_new_data = []
        self.polar_hr_new_data = []
        self.witmotion1_new_data = []
        self.witmotion2_new_data = []
        self.marker_new_data = []
        self.hat_new_data = []
        self.energy_array = np.ones((1, PRESENCE_DETECTION_RANGE[1]-PRESENCE_DETECTION_RANGE[0]), dtype=float)
        self.bin_history = []

        self.count = 0
        
        # Initialize result dictionary
        self.result_dict = {
            # Y/Data vectors
            "magn_data":[],
            "presence_flag":[],
            "energy_array":np.zeros((PLOTTING_WIN, PRESENCE_DETECTION_RANGE[1]-PRESENCE_DETECTION_RANGE[0])),
            "target_bin":[],
            "target_data": [],
            "motion_signal":[],
            "difference_data":[],
            "respiration_signal":[],
            "respiration_bpm_fft":[],
            "respiration_pattern":[],
            "heartbeat_signal":[],
            "heartbeat_bpm_fft":[],
            "witmotion1_new_data":[],
            "witmotion2_new_data":[],
            "vernier_belt_new_data":[],
            "polar_ecg_new_data":[],
            "polar_hr_new_data":[],
            "marker_new_data":[],
            "hat_new_data":[],

            # X vectors
            "x_marker_array" : np.linspace(-PLOTTING_WIN, 0 ,PLOTTING_WIN*M5STICKC_MARKER_FPS),
            "x_vernier_belt_array": np.linspace(-PLOTTING_WIN, 0, PLOTTING_WIN*VERNIER_BELT_FPS),
            "x_polar_ecg_array": np.linspace(-PLOTTING_WIN, 0 , PLOTTING_WIN*POLAR_ECG_FPS),
            "x_polar_hr_array": np.linspace(-PLOTTING_WIN, 0 , PLOTTING_WIN),
            "x_witmotion1_array": np.linspace(-PLOTTING_WIN, 0, WITMOTION1_FPS*PLOTTING_WIN),
            "x_witmotion2_array": np.linspace(-PLOTTING_WIN, 0, WITMOTION2_FPS*PLOTTING_WIN),
            "x_radar_array": np.linspace(-PLOTTING_WIN, 0, FPS*PLOTTING_WIN),
            "x_hat_array": np.linspace(-PLOTTING_WIN, 0, HAT_FPS*PLOTTING_WIN)
        }

    def slide_range_matrix_window(self, range_matrix): 
        # Stack new data:
        # print(range_matrix.shape)
        self.range_matrix_queue = np.vstack((self.range_matrix_queue, range_matrix))
        
        if len(self.range_matrix_queue) < PLOTTING_WIN*FPS:
            print(style.RED + 'Calibrating ... Please Wait !' + style.RESET)
            if ADAPTIVE_PRESENCE_THRESH:
                global PRESENCE_THRESH
                PRESENCE_THRESH = set_presence_threshold(self.range_matrix_queue)
                print(style.UNDERLINE + f"\nEstimating Scene Noise Floor: {round(PRESENCE_THRESH)} ... Please Keep Scene Empty\n"+ style.RESET)
        else:
            if self.calibration_flag.qsize()==0:
                self.calibration_flag.put(False)
                # print('Able to compute ...')

    def calculate(self):
        if USE_RADAR:
            # Waiting for calibration:
            if self.calibration_flag.qsize()==0:
                return
            # print("\n enter calculate ...")
            # print(self.range_matrix_queue.shape)
            if VERSION == "8_ANT":
                _range_matrix_queue = self.range_matrix_queue[-PLOTTING_WIN*FPS:,:MAX_DISTANCE,0].copy()   # Use first antenna
                angle_data = self.range_matrix_queue[-PLOTTING_WIN*FPS:,:MAX_DISTANCE].copy()
            elif VERSION == "1_ANT":
                _range_matrix_queue = self.range_matrix_queue[-PLOTTING_WIN*FPS:,:MAX_DISTANCE].copy() 

            # remove static clutter
            _range_matrix_queue = clutter_remove(_range_matrix_queue)
            # print("\n passed clutter removal ...")

            # Extract magnitude:
            magn_data = np.abs(_range_matrix_queue)
            self.result_dict["magn_data"] = magn_data

            # Get angle info:
            if VERSION == "8_ANT":
                for ant in range(angle_data.shape[0]):
                    angle_data[ant] = clutter_remove(angle_data[ant])
                capon_results, cfar_results = get_angle_info(angle_data)
                self.result_dict["angle_info"] = [capon_results, cfar_results]


            # print("\n passed magn data extraction ...")

            # Presence Detection 
            presence_flag, energy_array = presence_detection_handler(data=magn_data, energy_array=self.result_dict["energy_array"], thresh=PRESENCE_THRESH)   # TODO: current is just initial implementation
            self.result_dict["presence_flag"].append(presence_flag)
            self.result_dict["presence_flag"] = self.result_dict["presence_flag"][-PLOTTING_WIN:]  # Keep the last plotting window

            self.result_dict["energy_array"] = energy_array #########################################

            ####################################################################################
            # plot_bin_power_history(self.energy_array) #### Uncomment for testing 
            ####################################################################################

            if presence_flag:
                # print(style.GREEN + 'Subject is present' + style.RESET)

                ################ Locate Target: #######################
                target_bin, bin_list = locate_subject(data=magn_data)   # TODO: Localization (chest, head, hand, ...)
                self.result_dict["target_bin"].append(target_bin)
                # self.result_dict["target_bin"] = self.result_dict["target_bin"][-PLOTTING_WIN:]
                self.bin_history.append([target_bin, bin_list])

                ############### IQ data : #############################
                self.result_dict["1dfft_IQ_data"] = _range_matrix_queue[:, target_bin]

                ################ Monitoring Subject Motion and Vital Signs: #######################
                # Extract phase data and Unwrap:
                phase_data = np.angle(_range_matrix_queue[:,target_bin[0]], deg=True)
                target_data = unwrap_phase_Ambiguity(phase_data)
                self.result_dict["target_data"] = target_data
                # print("\n passed phase unwrapping ...")

                # Phase difference: --> remove DC
                data_0 = target_data[1:]
                data_1 = target_data[:-1]
                difference_data = np.array(data_0) - np.array(data_1)
                difference_data = difference_data.flatten()
                self.result_dict["difference_data"] = difference_data

                ## --> data is reduced by one sample due to difference operation

                # Motion detection
                # a/ Big Motion: Static, Moving Away, Approaching --> Inter-bin (Macro)
                self.result_dict["Motion_State_bin"] = motion_detection_macro(self.bin_history)   # TODO: Use bin change information

                # b/ small Motion: Static, Moving Away, Approaching --> Inter-bin (Macro)
                self.result_dict["motion_signal"] = motion_detection_micro(magn_data[:,target_bin], self.count)    
                self.count+=1
                # print("passed motion signal")

                # I/ Respiration Monitoring:  --> Need to send all data history
                respiration_signal = extract_respiration(data = difference_data)
                self.result_dict["respiration_signal"] = respiration_signal
            
                # Estimate current respiration rate: ==> FFT-based # TODO: improve
                respiration_bpm_fft = fft_find_bpm(respiration_signal, sampling_period=FPS)
                self.result_dict["respiration_bpm_fft"].append(respiration_bpm_fft)
                self.result_dict["respiration_bpm_fft"] = self.result_dict["respiration_bpm_fft"][-PLOTTING_WIN:]

                # Calculate derivative add by nakajima
                from sklearn import preprocessing
                first_derivative_radar = np.gradient(respiration_signal)
                abs_derivative_radar = np.abs(first_derivative_radar)
                scale_derivative_radar = preprocessing.minmax_scale(abs_derivative_radar)
                self.result_dict["respiration_derivative"] = abs_derivative_radar # add by nakajima

                # Respiration Pattern
                respiration_pattern, mean_list, thresh_list = estimate_respiration_pattern(respiration_signal)
                self.result_dict["respiration_pattern"] = respiration_pattern

                respiration_pattern = estimate_respiration_pattern_thresh(abs_derivative_radar, threshold = 0.05)
                self.result_dict["respiration_pattern"] = respiration_pattern

                # print(len(self.result_dict["respiration_derivative"])) # remove if you don't need
                # print(len(self.result_dict["x_radar_array"][1:]))

                # print("passed respiration processing")
                
                # Heartbeat Signal
                heartbeat_signal = extract_heartbeat(data=difference_data)
                self.result_dict["heartbeat_signal"] = heartbeat_signal
                
                # Estimate Heartrate: ==> FFT-based # TODO: improve
                heartbeat_bpm_fft = fft_find_bpm(heartbeat_signal, sampling_period=FPS)
                self.result_dict["heartbeat_bpm_fft"].append(heartbeat_bpm_fft)
                self.result_dict["heartbeat_bpm_fft"] = self.result_dict["heartbeat_bpm_fft"][-PLOTTING_WIN:]

                # print("passed heartbeat processing")
                # TODO: III/ Eyeblink Monitoring

                # print("passed Vitals estimation")
            else:
                self.result_dict["target_bin"].append([0])
        
            self.prepare_plot_data()

    def prepare_plot_data(self):
        ######################################### Get reference data #########################################
        # Get witmotion reference data:
        if USE_WITMOTION1:
            for i in range(self.witmotion1_realtime_q.qsize()):
                temp = np.asarray(self.witmotion1_realtime_q.get())
                self.witmotion1_new_data.append(temp)

            self.witmotion1_new_data = self.witmotion1_new_data[-PLOTTING_WIN*WITMOTION1_FPS:]
            self.result_dict["witmotion1_new_data"] = np.asarray(self.witmotion1_new_data, dtype=float)
        
        if USE_WITMOTION2:
            for i in range(self.witmotion2_realtime_q.qsize()):
                temp = np.asarray(self.witmotion2_realtime_q.get())
                self.witmotion2_new_data.append(temp)

            self.witmotion2_new_data = self.witmotion2_new_data[-PLOTTING_WIN*WITMOTION2_FPS:]
            self.result_dict["witmotion2_new_data"] = np.asarray(self.witmotion2_new_data, dtype=float)

        # Get Vernier reference data:
        if USE_VERNIER_BELT:
            for i in range(self.vernier_belt_realtime_q.qsize()):
                self.vernier_belt_new_data.append(self.vernier_belt_realtime_q.get())
        
            self.vernier_belt_new_data = self.vernier_belt_new_data[-PLOTTING_WIN*VERNIER_BELT_FPS:]
            self.result_dict["vernier_belt_new_data"] = self.vernier_belt_new_data
            
            # proccesed data
            vernier_processed_data = vernier_proccesing(self.vernier_belt_new_data)
            vernier_processed_data = vernier_processed_data[-PLOTTING_WIN*VERNIER_BELT_FPS:]
            self.result_dict["processed_vernier_new_data"] = vernier_processed_data
            
            # estimate pattern
            vernier_respiration_pattern, mean_list, thresh_list = estimate_respiration_pattern(vernier_processed_data, fs=VERNIER_BELT_FPS, threshold=0.05)
            self.result_dict["vernier_respiration_pattern"] = vernier_respiration_pattern
            
            #for check mean value
            self.result_dict["vernier_mean_list"] = mean_list 
            self.result_dict["x_vernier_mean_array"] = np.linspace(-PLOTTING_WIN, 0, len(mean_list))
 
            self.result_dict["vernier_thresh"] = np.full(len(mean_list), 0.05)
            # self.result_dict["vernier_thresh"] = thresh_list
            self.result_dict["x_vernier_thresh_array"] = np.linspace(-PLOTTING_WIN, 0, len(mean_list))

            v_std = calculate_std(vernier_processed_data)
            self.result_dict["vernier_std"] = v_std
            self.result_dict["x_vernier_std_array"] = np.linspace(-PLOTTING_WIN, 0, len(v_std))

            change_points = get_change_point(vernier_processed_data)
            change_points = [(i - PLOTTING_WIN*VERNIER_BELT_FPS) / VERNIER_BELT_FPS for i in change_points]
            self.result_dict["vernier_change_point"] = change_points

            
        # Get Polar H10 reference data: --> ECG, HR
        if USE_POLAR:
            # Get ECG
            if POLAR_ECG_DATA_FLAG:
                for i in range(self.polar_ecg_realtime_q.qsize()):
                    self.polar_ecg_new_data.append(self.polar_ecg_realtime_q.get())
        
                self.polar_ecg_new_data = self.polar_ecg_new_data[-PLOTTING_WIN*POLAR_ECG_FPS:]
                self.result_dict["polar_ecg_new_data"] = self.polar_ecg_new_data
                
            # Get HR
            if POLAR_HR_DATA_FLAG:
                for i in range(self.polar_hr_realtime_q.qsize()):
                    self.polar_hr_new_data.append(self.polar_hr_realtime_q.get())
        
                self.polar_hr_new_data = self.polar_hr_new_data[-PLOTTING_WIN:]
                self.result_dict["polar_hr_new_data"] = self.polar_hr_new_data   

        if USE_M5STICKC_MARKER:
            # Get Data
            for i in range(self.m5stickc_marker_realtime_q.qsize()):
                self.marker_new_data.append(self.m5stickc_marker_realtime_q.get())

            self.marker_new_data = self.marker_new_data[-PLOTTING_WIN*M5STICKC_MARKER_FPS:]
            self.result_dict["marker_new_data"] = self.marker_new_data # value = 49

        if USE_HAT:
            # Get Data
            for i in range(self.hat_realtime_q.qsize()):
                self.hat_new_data.append(self.hat_realtime_q.get())

            self.hat_new_data = self.hat_new_data[-PLOTTING_WIN*HAT_FPS:]
            self.result_dict["hat_new_data"] = np.asarray(self.hat_new_data, dtype=float)
            # print(self.result_dict["hat_new_data"].shape)

        ##################################### Show Results ##############################################
        if USE_MATPLOTLIB:
            matplotlib_plot(result_dict=self.result_dict)
        else: 
            # Send to GUI
            self.results_dict_q.put(self.result_dict)
        
        

