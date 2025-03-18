import numpy as np
import warnings
from sys_config import *
from user_config import *

SENSOR_L = 0.00483536  # 62GHz
SENSOR_DISTANCE = 0.5 * SENSOR_L  

CALIBRATION = True

def cov_matrix(x):
    """ Calculates the spatial covariance matrix (Rxx) for a given set of input data (x=inputData). 
        Assumes rows denote Vrx axis.

    Args:
        x (ndarray): A 2D-Array with shape (rx, adc_samples) slice of the output of the 1D range fft

    Returns:
        Rxx (ndarray): A 2D-Array with shape (rx, rx)
    """
    
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")

    if x.shape[0] > x.shape[1]:
        warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
        x = x.T

    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, num_adc_samples)

    return Rxx

def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate a steering vector for AOA estimation given the theta range, theta resolution, and number of antennas

    Defines a method for generating steering vector data input --Python optimized Matrix format
    The generated steering vector will span from -angEstRange to angEstRange with increments of ang_est_resolution
    The generated steering vector should be used for all further AOA estimations (bartlett/capon)

    Args:
        ang_est_range (int): The desired span of thetas for the angle spectrum.
        ang_est_resolution (float): The desired resolution in terms of theta
        num_ant (int): The number of Vrx antenna signals captured in the RDC

    Returns:
        num_vec (int): Number of vectors generated (integer divide angEstRange/ang_est_resolution)
        steering_vectors (ndarray): The generated 2D-array steering vector of size (num_vec,num_ant)

    Example:
        >>> #This will generate a numpy array containing the steering vector with 
        >>> #angular span from -90 to 90 in increments of 1 degree for a 4 Vrx platform
        >>> _, steering_vec = gen_steering_vec(90,1,4)

    """
    num_vec = (2 * ang_est_range / ang_est_resolution + 1)
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)

            steering_vectors[kk, jj] = complex(real, imag)

    return [num_vec, steering_vectors]


def forward_backward_avg(Rxx):
    """ Performs forward backward averaging on the given input square matrix

    Args:
        Rxx (ndarray): A 2D-Array square matrix containing the covariance matrix for the given input data

    Returns:
        R_fb (ndarray): The 2D-Array square matrix containing the forward backward averaged covariance matrix
    """
    assert np.size(Rxx, 0) == np.size(Rxx, 1)

    # --> Calculation
    M = np.size(Rxx, 0)  # Find number of antenna elements
    Rxx = np.matrix(Rxx)  # Cast np.ndarray as a np.matrix

    # Create exchange matrix
    J = np.eye(M)  # Generates an identity matrix with row/col size M
    J = np.fliplr(J)  # Flips the identity matrix left right
    J = np.matrix(J)  # Cast np.ndarray as a np.matrix

    R_fb = 0.5 * (Rxx + J * np.conjugate(Rxx) * J)

    return np.array(R_fb)


def aoa_capon(x, steering_vector, magnitude=False):
    """Perform AOA estimation using Capon (MVDR) Beamforming on a rx by chirp slice

    Calculate the aoa spectrum via capon beamforming method using one full frame as input.
    This should be performed for each range bin to achieve AOA estimation for a full frame
    This function will calculate both the angle spectrum and corresponding Capon weights using
    the equations prescribed below.

    .. math::
        P_{ca} (\\theta) = \\frac{1}{a^{H}(\\theta) R_{xx}^{-1} a(\\theta)}
        
        w_{ca} (\\theta) = \\frac{R_{xx}^{-1} a(\\theta)}{a^{H}(\\theta) R_{xx}^{-1} a(\\theta)}

    Args:
        x (ndarray): Output of the 1d range fft with shape (num_ant, numChirps)
        steering_vector (ndarray): A 2D-array of size (numTheta, num_ant) generated from gen_steering_vec
        magnitude (bool): Azimuth theta bins should return complex data (False) or magnitude data (True). Default=False

    Raises:
        ValueError: steering_vector and or x are not the correct shape

    Returns:
        A list containing numVec and steeringVectors
        den (ndarray: A 1D-Array of size (numTheta) containing azimuth angle estimations for the given range
        weights (ndarray): A 1D-Array of size (num_ant) containing the Capon weights for the given input data
    
    Example:
        >>> # In this example, dataIn is the input data organized as numFrames by RDC
        >>> Frame = 0
        >>> dataIn = np.random.rand((num_frames, num_chirps, num_vrx, num_adc_samples))
        >>> for i in range(256):
        >>>     scan_aoa_capon[i,:], _ = dss.aoa_capon(dataIn[Frame,:,:,i].T, steering_vector, magnitude=True)

    """

    if steering_vector.shape[1] != x.shape[0]:
        raise ValueError("'steering_vector' with shape (%d,%d) cannot matrix multiply 'input_data' with shape (%d,%d)" \
        % (steering_vector.shape[0], steering_vector.shape[1], x.shape[0], x.shape[1]))

    Rxx = cov_matrix(x)
    Rxx = forward_backward_avg(Rxx)
    Rxx_inv = np.linalg.inv(Rxx)
    
    # Calculate Covariance Matrix Rxx
    first = Rxx_inv @ steering_vector.T
    den = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), first.T))
    weights = np.matmul(first, den)

    if magnitude:
        return np.abs(den), weights
    else:
        return den, weights

def capon2D(range_matrix):
    if RADAR_DEVICE == "TI-IWR6843AOP":
        # Antenna Geometry0 (Tx1 - Tx2)
        a_v = (np.arange(0, TI_2DAoA_VIRTUAL_ANT_NUM)).T  
        a_v[0] = -1
        a_v[1] = -1
        a_v[2] = 0
        a_v[3] = 0
        a_v[4] = -3
        a_v[5] = -3
        a_v[6] = -2
        a_v[7] = -2

        # Antenna Geometry1 (Tx1 - Tx2)
        e_v = (np.arange(0, TI_2DAoA_VIRTUAL_ANT_NUM)).T  
        e_v[0] = -1
        e_v[1] = 0
        e_v[2] = -1
        e_v[3] = 0
        e_v[4] = -3
        e_v[5] = -2
        e_v[6] = -3
        e_v[7] = -2

        # Phase Rotation (Tx1 - Tx2)
        p_v = (np.arange(0, TI_2DAoA_VIRTUAL_ANT_NUM)).T  
        p_v[0] = -1
        p_v[1] = 1
        p_v[2] = -1
        p_v[3] = 1
        p_v[4] = -1
        p_v[5] = 1
        p_v[6] = -1
        p_v[7] = 1
    
    elif RADAR_DEVICE == "TI-IWR6843ODS":
        # Antenna Geometry0 (Tx1 - Tx2)
        a_v = (np.arange(0, TI_2DAoA_VIRTUAL_ANT_NUM)).T  
        a_v[0] = 0
        a_v[1] = 0
        a_v[2] = -1
        a_v[3] = -1
        a_v[4] = -2
        a_v[5] = -2
        a_v[6] = -3
        a_v[7] = -3

        # Antenna Geometry1 (Tx1 - Tx2)
        e_v = (np.arange(0, TI_2DAoA_VIRTUAL_ANT_NUM)).T  
        e_v[0] = 0
        e_v[1] = -1
        e_v[2] = -1
        e_v[3] = 0
        e_v[4] = 0
        e_v[5] = -1
        e_v[6] = -1
        e_v[7] = 0

        # Phase Rotation (Tx1 - Tx2)
        p_v = (np.arange(0, TI_2DAoA_VIRTUAL_ANT_NUM)).T 
        p_v[0] = 1
        p_v[1] = -1
        p_v[2] = -1
        p_v[3] = 1
        p_v[4] = 1
        p_v[5] = -1
        p_v[6] = -1
        p_v[7] = 1

    range_matrix = range_matrix.T

    # snapshot_num = TI_1DFFT_QUEUE_LEN
    azimuth_fov = SEARCHING_AZIMUTH_DOA_RANGE.copy()
    elevation_fov = SEARCHING_ELEVATION_DOA_RANGE.copy()
    power = np.zeros((len(azimuth_fov), len(elevation_fov)))
    sensor_num, snapshot_num = range_matrix.shape

    #EVA_max diff EVA_min
    differ=0

    try:
        temp_arr = range_matrix.dot(np.conj(range_matrix.T)) / snapshot_num
        w_ave = np.trace(temp_arr) / sensor_num
        # print("trace_average: %s " % w_ave)
        w, v = np.linalg.eig(temp_arr)  # 特征值和特征向量分解，v[:,i]是特征值w[i]对应的特征向量
        w_abs = np.abs(w)  # 特征值分解之后是复数形式，求模值
        EVA = np.conj(w_abs).T
        EVA_sorted = np.array(np.sort(EVA))
        EVA_ave = np.sum(EVA_sorted[0:6])
        # print("PCA_ratio: %s " % PCA_ratio)

        temp_arr = temp_arr + EVA_ave * np.eye(sensor_num)

        inv_temp_arr = np.linalg.inv(temp_arr)

    except Exception as e:
        print(e)
        return power

    for azimuth in range(len(azimuth_fov)):
        for elevation in range(len(elevation_fov)):
            v1 = (-1j) * np.pi * np.sin(azimuth_fov[azimuth] * np.pi / 180)
            v2 = (-1j) * np.pi * np.sin(elevation_fov[elevation] * np.pi / 180)
            v1_t = a_v.dot(v1)
            v2_t = e_v.dot(v2)

            v = np.add(v1_t, v2_t)

            v_e = np.exp(v)

            a_theta = np.array(p_v * v_e)

            a_theta_t = np.array(a_theta.T)
            power[azimuth][elevation] = round(
                1 / np.abs((np.conj(a_theta).dot(inv_temp_arr)).dot(np.array(a_theta_t))), 4)

    rev_data = power
    
    return rev_data

#2D-CA-CFAR
def noise_cal_2d(data, x_idx, y_idx):

    result_sum = data[x_idx - 2][y_idx] + data[x_idx - 3][y_idx] + data[x_idx + 2][y_idx] + data[x_idx + 3][y_idx] + \
                 data[x_idx][y_idx - 2] + data[x_idx][y_idx + 2]

    return result_sum / 6

def ca_cfar_2d(data):

    rev_data = np.array(data)
    azimuth_fov = SEARCHING_AZIMUTH_DOA_RANGE.copy()
    elevation_fov = SEARCHING_ELEVATION_DOA_RANGE.copy()
    decision_pairs = np.zeros((len(azimuth_fov), len(elevation_fov)), dtype=float)
    points = 0

    threshold = 1 # CFAR的阈值

    for azimuth in range(2, len(azimuth_fov)-3, 1):
        for elevation in range(2, len(elevation_fov)-3, 1):
            noise_ave = noise_cal_2d(rev_data, azimuth, elevation)
            if rev_data[azimuth][elevation] > (noise_ave * threshold):
                decision_pairs[azimuth][elevation] = rev_data[azimuth][elevation]
                points += 1

    return decision_pairs