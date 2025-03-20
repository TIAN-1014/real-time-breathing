from scipy import signal
import numpy as np
from sys_config import *
from user_config import *
from sklearn import preprocessing
from scipy.signal import find_peaks

from libs.utils import combine_overlapping_tuples

import ruptures as rpt

def extract_respiration(data=None):
    b, a = signal.butter(4, RESP_RANGE[1], 'lowpass', fs=FPS) 
    respiration_signal = signal.filtfilt(b, a, data)  
    # print("\n passed respiration filter")

    return respiration_signal


# TODO: Define a more reliable respiration extraction method

# TODO: Define a reliable respiration rate method

def estimate_respiration_pattern(respiration_signal, fs = FPS, threshold = RESP_THRESH):
    hold_window = HOLD_INTERVAL*fs
    hold_list = []
    mean_list = []
    thresh_list = []

    for i in range(hold_window, len(respiration_signal)):
        window_mean = np.mean(abs(respiration_signal[i-hold_window:i]))
        mean_list.append(window_mean)
        thresh_list.append(calculate_threshold(respiration_signal[i-hold_window:i], b=0.5))

        if window_mean <= threshold: # hold_window間の平均値が閾値以下なら
            hold_list.append((i-hold_window,i))   # Hold

    hold_list = combine_overlapping_tuples(hold_list)

    return hold_list, mean_list, thresh_list

def estimate_respiration_pattern_thresh(respiration_signal, fs = FPS, threshold = RESP_THRESH):
    hold_window = HOLD_INTERVAL*fs # window size
    hold_list = [] # apnea duration

    for i in range(hold_window, len(respiration_signal)):
        windowing_data = respiration_signal[i-hold_window:i]
        if all(num <= threshold for num in windowing_data): # windowしたデータの値すべてが閾値以下なら
            hold_list.append((i-hold_window,i))   # Hold

    hold_list = combine_overlapping_tuples(hold_list)

    return hold_list

# TODO: Define a more reliable resp. pattern estimation method

def calculate_threshold(windowing_data, b = 1):
    threshold_value = np.mean(windowing_data) - b * np.std(windowing_data) 
    return threshold_value

def calculate_std(respiration_signal=None):
    data = respiration_signal
    window_size = 5*FPS
    overlap = 1*FPS
    data_std = []

    for i in range(0, len(data) - window_size+1, overlap):
        window_data = data[i:i + window_size]
        std = np.std(window_data)
        data_std.append(std)
    
    return data_std

def get_change_point(respiration_signal=None):
    data = respiration_signal

    algo = rpt.Pelt(model="rbf", min_size=1, jump=10).fit(data)
    result = algo.predict(pen=2)

    change_point = [i for i in result if i<len(data)]
    return change_point

# reference device
def vernier_proccesing(data=None):
   
    # remove dc offset 
    dc_offset = np.mean(data)
    processed_data = data - dc_offset

    # phase difference
    # data_0 = data[1:]
    # data_1 = data[:-1]
    # v_difference_data = np.array(data_0) - np.array(data_1)
    # v_difference_data = v_difference_data.flatten()
    # processed_data = v_difference_data

    # filter
    # b, a = signal.butter(4, 5, btype='low',fs=VERNIER_BELT_FPS)
    # processed_data = signal.filtfilt(b, a, v_difference_data)

    return processed_data

def get_m5timestamp(data = None):
    timestamps = []
    
    return timestamps

def cal_time(index, fs = FPS):
    sampling_interval = 1 / fs
    time = index * sampling_interval
    time = -(PLOTTING_WIN - time) # convert to '-'
    return str(time)

def detect_characteristic_points(respiration_signal, min_distance_threshold=20):
    first_derivative = np.gradient(respiration_signal)
    second_derivative = np.gradient(first_derivative)
    
    height = 0.003
    threshold = 0.0001
    distance = 20
    
    peaks_2rd, _ = find_peaks(second_derivative, distance=distance, height=height, threshold=threshold)
    troughs_2rd, _ = find_peaks(-second_derivative, distance=distance, height=height, threshold=threshold)
    
    zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
    updated_zero_crossings = []
    
    i = 0
    while i < len(zero_crossings) - 1:
        distance_between_crossings = zero_crossings[i+1] - zero_crossings[i]
        
        if distance_between_crossings < min_distance_threshold:
            if (zero_crossings[i] in peaks_2rd or zero_crossings[i+1] in peaks_2rd or 
                zero_crossings[i] in troughs_2rd or zero_crossings[i+1] in troughs_2rd):
                
                if any(abs(peak - zero_crossings[i]) > min_distance_threshold for peak in peaks_2rd) or \
                   any(abs(trough - zero_crossings[i]) > min_distance_threshold for trough in troughs_2rd):
                    updated_zero_crossings.append(zero_crossings[i])
                    updated_zero_crossings.append(zero_crossings[i+1])
                else:
                    midpoint = (zero_crossings[i+1] + zero_crossings[i]) // 2
                    updated_zero_crossings.append(midpoint)
                i += 2
            else:
                midpoint = (zero_crossings[i+1] + zero_crossings[i]) // 2
                updated_zero_crossings.append(midpoint)
                i += 2
        else:
            updated_zero_crossings.append(zero_crossings[i])
            i += 1
    
    if i == len(zero_crossings) - 1:
        updated_zero_crossings.append(zero_crossings[i])
    
    return np.array(updated_zero_crossings)

def estimate_respiration_CPs(respiration_signal,fs=FPS):
    Inhalation_star_points=[]
    Exhalation_star_points=[]
    Inhalation_end_points=[]
    Exhalation_end_points=[]
    holding_star_points=[]
    holding_end_points=[]
    apnea_star_points=[]
    apnea_end_points=[]
    