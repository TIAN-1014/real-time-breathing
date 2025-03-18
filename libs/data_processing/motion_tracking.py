import numpy as np
from statistics import mode
from scipy import signal
import matplotlib.pyplot as plt

from sys_config import *
from user_config import *

from libs.utils import filter_data, compute_derivative, get_Spectrum

def motion_detection_micro(data, count):
    """
    Target motion based on magnitude/phase at target bin (for small motions --> within a single bin)
    input: magnitude/phase
    output: Static, movement
    """
    # Q: If the target is static, is there micro motion within the target bin?

    output = np.ones(len(data))
    return output[:-1]


# TODO: Define a tracking method

def motion_detection_macro(bin_history):
    """
    Target motion based on bin change history (for big motions --> inter bins)
    input: bin history
    output: Static, Moving Away, Approaching
    """
    # Q: Is the target static or moving? Approaching or moving Away?

    # TODO: How to deal with the case of the target at the edge between 2 bins? Motion and phase handling?

    target_state = ''
    check_list = bin_history[-MACRO_MOVE_CHECK_WINDOW:]

    if check_list[-1][0]>(check_list[0][0]):
        target_state = "Moving Away"
        
    elif check_list[-1][0]<(check_list[0][0]):
        target_state = "Approaching"
    else:
        target_state = "Static"

    # print(f"Target is {target_state}")
    return target_state
