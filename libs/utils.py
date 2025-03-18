import serial
import multiprocessing
import numpy as np
import time
import winsound
import datetime
from scipy.signal import butter, filtfilt
from scipy.fft import fft

from sys_config import *
from user_config import *

import mysql.connector as sql
import os

try:
    class Window(Enum):
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4
except NameError:
    class Window:
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4

def connectToMainDB():
  conn = sql.connect(host=SERVER,port=PORT,user=USER,passwd=PASSWORD,db=DATABASE)
  curs = conn.cursor()
  return conn, curs

def reset_vitals():
    conn, curs = connectToMainDB()
    curs.execute('''UPDATE vitals SET hr=0, br=0, distance=-1, angle=-1, presence=0, motion=0 WHERE id_record=1''')
    conn.commit()
    conn.close() 

class WisProcess(multiprocessing.Process):
    def __init__(self, in_q=None, out_q=None, logger=None, file_dir_name=None):
        super().__init__()
        self._in_q = in_q
        self._out_q = out_q
        self._logger = logger
        self._file_dir_name = file_dir_name

    def run(self):
        raise NotImplementedError

class WisSerial:
    def __init__(self, port=SERIAL_PORT_NAME, baudrate=921600):
        self.ser = serial.Serial()
        self.ser.baudrate = baudrate
        self.ser.timeout = 2
        self.ser.port = port

    def connect(self):
        try:
            if self.ser.is_open:
                print('will close %s' % (self.ser.port,))
                self.ser.close()
            self.ser.open()
            if self.ser.is_open:
                return
        except:
            pass

        if self.ser.is_open:
            print(' Open UART success!')
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        else:
            print(' Open UART fail')

    def read(self, buf_len):
        rxbuf = self.ser.read(buf_len)
        return rxbuf
    
    def read_buffer_line(self):
        rxbuf_readline = self.ser.readline()
        return rxbuf_readline

    def write(self, content):
        if isinstance(content, str):
            content = content.encode('utf-8')
        self.ser.write(content)

    def close(self):
        try:
            self.ser.close()
        except:
            pass

    def is_open(self):
        try:
            return self.ser.is_open
        except:
            return False
        
def clutter_remove(range_matrix_queue):
    for range_idx in range(range_matrix_queue.shape[1]):
        range_avg = range_matrix_queue[:, range_idx].mean()
        range_matrix_queue[:, range_idx] = range_matrix_queue[:, range_idx] - range_avg
    return range_matrix_queue

def ph_uwp(phase_array):
    for i in range(1,len(phase_array)):
        if phase_array[i]-phase_array[i-1]>180:
            phase_array[i]=phase_array[i]-360
        elif phase_array[i]-phase_array[i-1]<-180:
            phase_array[i]=phase_array[i]+360
    return phase_array

def unwrap_phase_Ambiguity(phase_queue):
    # for range_idx in range(0, points_num):
    phase_arr = phase_queue #[:, range_idx]
    phase_arr_ret = phase_arr.copy()
    phase_diff_correction_cum = 0
    for i in range(len(phase_arr)):
        if not i:
            continue
        else:
            phase_diff = phase_arr[i] - phase_arr[i - 1]
            # print(len(phase_diff))
            if phase_diff > 180:
                mod_factor = 1
            elif phase_diff < -180:
                mod_factor = -1
            else:
                mod_factor = 0
        phase_diff_mod = phase_diff - mod_factor * 2 * 180
        if phase_diff_mod == -180 and phase_diff > 0:
            phase_diff_mod = 180
        phase_diff_correction = phase_diff_mod - phase_diff
        if (phase_diff_correction < 180 and phase_diff_correction > 0) \
                or (phase_diff_correction > -180 and phase_diff_correction < 0):
            phase_diff_correction = 0
        phase_diff_correction_cum += phase_diff_correction
        phase_arr_ret[i] = phase_arr[i] + phase_diff_correction_cum
    # phase_queue[:, range_idx] = phase_arr_ret
        phase_queue = phase_arr_ret
    return phase_queue

def hampel(X):
    length = X.shape[0] - 1
    k = 3
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1  
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma) 

    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

def normalize(signal):
    min_value = np.min(signal)
    max_value = np.max(signal)
    normalized_signal = np.asarray([(x - min_value) / (max_value - min_value) for x in signal])
    return normalized_signal

def fft_find_bpm(data, sampling_period = FPS, start_idx=0):
    # print(f"Sampling frequency used: {sampling_period} Hz")
    n = len(data)
    # FFT result corresponds to actual frequency
    frequency = np.arange(n / 2) * sampling_period / n
    fft_out = np.fft.fft(data)
    fft_am = abs(fft_out)
    fft_out_am = fft_am[start_idx:int(len(fft_am) / 2)]
    fft_max_index = np.argmax(fft_out_am) + start_idx
    fft_max = int(fft_out_am.max())
    max_fq = frequency[fft_max_index] if fft_max_index < len(frequency) else frequency[-1]
    bpm = round(max_fq * 60)
    return bpm

def beep(freq, duration, experiment):
    if EMIT_BEEP:
        winsound.Beep(freq, duration)
    # Add marker:  # TODO: Improve this task
    timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    BEEP_DATA.put([timeStamp, experiment])

def set_presence_threshold(data):
    thresh = 0
    temp_array = []
    for bin in range(PRESENCE_DETECTION_RANGE[0], PRESENCE_DETECTION_RANGE[1]):
        temp_array.append(np.max(np.abs(data[:,bin])))
    thresh = np.max(temp_array)
    thresh = thresh + 100
    return thresh

def filter_data(data, cutoff_frequency, type="bandpass", order = 4, fs = FPS):
    if len(cutoff_frequency) ==2:
        low_cutoff = cutoff_frequency[0]
        high_cutoff = cutoff_frequency[1]
        b, a = butter(order, [low_cutoff, high_cutoff], btype=type, analog=False, fs=fs)
    else:
        cutoff = cutoff_frequency[0]
        b, a = butter(order, cutoff, btype=type, analog=False, fs = fs)
    
    filtered_data = filtfilt(b, a, data)
    print("returning filtered data")
    return filtered_data

def compute_derivative(signal, sampling_rate):
    time_step = 1 / sampling_rate
    derivative = np.gradient(signal, time_step)
    return derivative

def normalize(signal):
    min_value = np.min(signal)
    max_value = np.max(signal)
    normalized_signal = np.asarray([(x - min_value) / (max_value - min_value) for x in signal])
    return normalized_signal

def get_Spectrum(y,Fs):
    """
    Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    mag = abs(Y)
    return frq,mag

def combine_overlapping_tuples(intervals):
    if not intervals:
        return []

    # Sort the intervals by the start index
    intervals.sort(key=lambda x: x[0])
    
    combined_intervals = []
    current_start, current_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start <= current_end + 1:  # Check if intervals overlap or are contiguous
            current_end = max(current_end, end)
        else:
            combined_intervals.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add the last interval
    combined_intervals.append((current_start, current_end))
    
    return combined_intervals

def signal_padding(signal, padding_value=0, position=0, length=None):
    """
    Add padding to the signal.
    
    Input:
        signal: signal to be padded (list or array)
        padding_value: value of the padded segment (default is 0)
        position: start location where to add the dummy padding values (default is 0)
        length: how many samples to add (default is None, which will add padding till the end)
    
    Output:
        padded_signal: signal with added padding
    """
    if length is None:
        length = len(signal) - position
    
    if position < 0 or position > len(signal):
        raise ValueError("Position out of bounds")
    
    if length < 0:
        raise ValueError("Length must be non-negative")
        
    if type(signal) == list:
        padded_signal = signal[:position] + [padding_value] * length + signal[position:]
    else: 
        signal = list(signal)
        padded_signal = signal[:position] + [padding_value] * length + signal[position:]
    return padded_signal

def windowing(input, window_type, axis=0): # --> From OpenRadar
    """Window the input based on given window type.

    Args:
        input: input numpy array to be windowed.

        window_type: enum chosen between Bartlett, Blackman, Hamming, Hanning and Kaiser.

        axis: the axis along which the windowing will be applied.
    
    Returns:

    """
    window_length = input.shape[axis]
    if window_type == Window.BARTLETT:
        window = np.bartlett(window_length)
    elif window_type == Window.BLACKMAN:
        window = np.blackman(window_length)
    elif window_type == Window.HAMMING:
        window = np.hamming(window_length)
    elif window_type == Window.HANNING:
        window = np.hanning(window_length)
    else:
        raise ValueError("The specified window is not supported!!!")

    output = input * window

    return output