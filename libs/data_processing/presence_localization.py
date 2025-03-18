from sys_config import *
from user_config import *
from libs.data_processing.beamforming import capon2D, ca_cfar_2d, gen_steering_vec, aoa_capon
import numpy as np
from statistics import mode
from collections import Counter

def presence_detection_handler(data=None, energy_array=None, thresh=None):
    """
    Q: Is there a target in the scene?
    """

    # Check energy/magnitude level
    temp = []

    data = data[-PRESENCE_CHECK_TIME_INTERVAL*FPS:]   # don't check all data --> only last time interval

    for bin in range(PRESENCE_DETECTION_RANGE[0], PRESENCE_DETECTION_RANGE[1]):
        max_bin = np.max(data[:,bin])
        temp.append(max_bin)

    # print("checking presence ...")
    energy_array = np.vstack((energy_array, np.asarray(temp)))
    presence_flag = False
    try:
        for idx in range(PRESENCE_DETECTION_RANGE[1]-PRESENCE_DETECTION_RANGE[0]):   # Check for each bin
            decision_array = []
            for i in range(1, PRESENCE_CHECK_TIME_INTERVAL):
                if energy_array[-i,idx]>thresh:
                    decision_array.append(True)
                else:
                    decision_array.append(False)

            # Decision for this bin:
            if np.all(decision_array):

                presence_flag = True
                break
    except Exception as e: print(e)
    
    return presence_flag, energy_array

################################################################
# TODO: Define a more reliable presence detection method
################################################################

def locate_subject(data): # --> Basic Implementation
    """
    Q: if a subject is present in the scene, What is the correct range bin to consider?
    """
    # Method I --> Get max power binfor each frame
    bins = []
    for i in range(1, LOCALIZATION_WIN*FPS):
        bins.append(np.argmax(data[-i,:LOCALIZATION_RANGE[-1]]))
    bin_list = list(set(bins))

    target_bin = mode(bins)

    # Method II --> Bin with max avg. power over a window
    bins = []
    for i in range(LOCALIZATION_RANGE[0], LOCALIZATION_RANGE[1]):
        bins.append(np.mean(data[-LOCALIZATION_WIN*FPS:,i]))

    target_bin = [np.argmax(bins)+LOCALIZATION_RANGE[0]]
    # print(target_bin)
    
    

    return target_bin, bin_list

# def locate_subject(data, bin_history=None): # --> Advanced Implementation
#     """
#     Q: if a subject is present in the scene, What is the correct range bin to consider?
#     """
#     # 1/ Get max power bin for each frame
#     bins = []
#     for i in range(1, LOCALIZATION_WIN*FPS):
#         bins.append(np.argmax(data[-i,:LOCALIZATION_RANGE[-1]]))
#     bin_list = list(set(bins))

#     # 2/ Check distribution of bins within the selection window
#     element_counts = Counter(bins)
#     total_elements = len(bins)
#     distribution = [(element, count / total_elements * 100) for element, count in element_counts.items()]
#     distribution =  sorted(distribution , key=lambda x: x[1], reverse=True)

#     # 3/ Temporary target bin:
#     target_bin = [distribution[0][0]]
#     print(f'\n{distribution}\n')

#     # 4/ Check if same target:
#     # if bin_history:
#     #     if abs(target_bin[0]-bin_history[-1])==0:  # same target, same bin
#     #         pass  # currently selected bin is good
#     #     elif abs(target_bin-bin_history[-1])==1:   # same target, on the edge between bins
#     #         if 
#     #         target_bins = []
        


#     # Check old target is still there: --> older bin in bin_list
#     # In case multiple subjects

            
#     # TODO: Find closest target location: --> Lock-in to target


#     return target_bin, bin_list

#############################################################
# TODO: Locate body parts (Head, Chest, ...)
#############################################################

def get_angle_info_1(data): # --> Pontosence Capon
    process_window = RAM_WINDOW # sec
    azimuth_fov = SEARCHING_AZIMUTH_DOA_RANGE.copy()
    elevation_fov = SEARCHING_ELEVATION_DOA_RANGE.copy()
    capon_result = np.zeros((MAX_DISTANCE, len(azimuth_fov), len(elevation_fov)), dtype=float)
    cfar_result = np.zeros((MAX_DISTANCE, len(azimuth_fov), len(elevation_fov)), dtype=float)

    for rge_idx in range(MAX_DISTANCE):
        # Capon Algorithm:
        power = capon2D(data[-process_window*FPS:,rge_idx,:])
        capon_result[rge_idx] = power
        # weight_vec[rge_idx] = weights 

        # Detecting Objects:
        cfar_result[rge_idx]= ca_cfar_2d(power)
    rge_idx, az_idx, ele_idx = np.unravel_index(capon_result.argmax(), capon_result.shape)
    
    print(rge_idx*RANGE_RESOLUTION, SEARCHING_AZIMUTH_DOA_RANGE[az_idx], SEARCHING_ELEVATION_DOA_RANGE[ele_idx])
    return capon_result, cfar_result

def get_angle_info(data):  # OpenRadar
    window = 2  # sec

    # Capon parameter (ODS)
    phase_rotation = [[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,1,0],
                    [0,0,0,-1]]
    phase_rotation = np.asarray(phase_rotation)

    # Generate Steering Vector
    _, steering_vector_a = gen_steering_vec(70, 2, 4)
    _, steering_vector_e = gen_steering_vec(70, 2, 2)
    # steering_vector_e = steering_vector_a.dot(phase_rotation)   # Account for antenna phase rotation (ODS)

    scan_aoa_capon_a = np.zeros((MAX_DISTANCE, len(steering_vector_a)), dtype=float)
    scan_aoa_capon_e = np.zeros((MAX_DISTANCE, len(steering_vector_e)), dtype=float)

    # Apply Capon Algorithm:
    for i in range(0, MAX_DISTANCE):
        scan_aoa_capon_a[i, :], _ = aoa_capon(data[-window*FPS:, i, [0,3,4,7]].T, steering_vector_a, magnitude=True)
        scan_aoa_capon_e[i, :], _ = aoa_capon(data[-window*FPS:, i, [4,5]].T, steering_vector_e, magnitude=True)

    scan_aoa_capon_a = np.transpose(scan_aoa_capon_a, axes=(1,0))
    scan_aoa_capon_e = np.transpose(scan_aoa_capon_e, axes=(1,0))

    return scan_aoa_capon_a, scan_aoa_capon_e