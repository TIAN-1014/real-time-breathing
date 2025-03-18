import numpy as np
from scipy import signal
from sys_config import *
from user_config import *

def extract_heartbeat(data=None):
    b, a = signal.butter(4, HEARTBEAT_RANGE, 'bandpass', fs=FPS) 
    heartbeat_signal = signal.filtfilt(b, a, data)  

    return heartbeat_signal

# TODO: Define a reliable/accurate method for heartbeat extraction

# TODO: Define a reliable Heart Rate estimation model
def estimate_heart_rate():
    pass


