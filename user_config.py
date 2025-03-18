###########################################################################################################
## Experiment Settings
EXPERIMENT_MODE = False     # True --> run according to "RECORDING_SEQUENCE",      False --> run until "q" keyboard press
RESEARCH_TOPIC = 'resp_pattern'             # General research topic (for folders/files organization)
SUBJECT = 'Nakajima'                                      # Name of current subject under test (for folders/files organization)
EXPERIMENT_CONDITION = 'deskN_test'        # Details of the experimental conditions (for folders/files organization)
RECORDING_SEQUENCE = [[30,0]]                          # timing for the different actions during the experiment

EMIT_BEEP = True                                       # Enable beep sound from PC

##############################################################################################################
## Devices:
SAVE_REF_DATA = True                                   # Save data from reference devices

############################
# Radar
############################
USE_RADAR = True
USE_DCA1000 = False     # TODO: Add support for the use of DCA1000
RADAR_DEVICE = "TI-IWR6843ODS"
VERSION = "1_ANT"  # "8_ANT"
RANGE_IDX_NUM = 35  # 6843  1 Ant / 8 ANT
SAVE_FFT_DATA = True                            # Save Radar 1DFFT data

############################
# Polar
############################
USE_POLAR = False   # Use Polar
# POLAR_MAC = 'C7:33:24:85:D9:8F'   # Polar H10 B2BD3A2A
POLAR_MAC = 'E5:4F:14:45:B0:7B'   # Polar H10 AFED7F2E
POLAR_ECG_DATA_FLAG = True
POLAR_ECG_FPS = 130   # Hz
POLAR_HR_DATA_FLAG = True
POLAR_ACC_DATA_FLAG = False
POLAR_ACC_FPS = 200   # Hz

############################
# Vernier-based Devices
############################
# 1/ Respiration Belt
USE_VERNIER_BELT = False   # Use Vernier
VERNIER_BELT_FPS = 20    # Hz
ENABLE_VERNIER_BELT_BLE = False
VERNIER_BELT_MODEL = "GDX-RB 0K202731"
VERNIER_BELT_MAC = "BC:33:AC:AE:BE:35"  # GDX-RB 0K202731

# 2/ Spirometer  # TODO
USE_VERNIER_SPIRO = False
VERNIER_SPIRO_FPS = 20    # Hz
ENABLE_VERNIER_SPIRO_BLE = False
VERNIER_SPIRO_MODEL = "GDX-SPR 115010G6"
VERNIER_SPIRO_MAC = "A4:6D:D4:1C:28:50"  # GDX-SPR 115010G6

############################
# Witmotion (2 instances are supported)
############################
USE_WITMOTION1 = False
WITMOTION1_MAC = "E4:F1:FB:36:E5:2E" 
WITMOTION1_FPS = 100
WITMOTION1_LOCATION = "Head"
WITMOTION1_CHANNEL = "ACC"   # "ACC", "GYR", "ANG"

USE_WITMOTION2 = False
# WITMOTION2_MAC = 'DE:E6:73:F1:BE:25'
WITMOTION2_MAC = 'DB:41:B5:D2:89:02'
WITMOTION2_FPS = 100
WITMOTION2_LOCATION = "Chest"
WITMOTION2_CHANNEL = "ACC"   # "ACC", "GYR", "ANG"

############################
# M5Stick-based sensors
############################
USE_M5STICKC_MARKER = False   # M5Stick Button to mark actions
# M5STICKC_MARKER_MAC = "24:A1:60:53:C1:AA"
M5STICKC_MARKER_MAC = "24:A1:60:53:C4:1E"
M5STICKC_MARKER_FPS = 20


USE_HAT = False  # IMU+EOG data for head
HAT_MAC = '94:B9:7E:8C:45:CE'
HAT_FPS = 75

############################
# Camera and Video Recording
############################
RECORD_VIDEO = False   # Record whole screen
USE_CAMERA = False     # Managed in the same process as the screen recording     
LIVE_FEED = False      # Show camera feed during the recording
CAMERA_INDEX = 0
CAM_FPS = 70
CAM_RES = (640,480)
CAM_SIZE = (480,300)
CAM_POS = (1620, 300)  # TODO: Better handling process

############################################################################################
## Signal Processing Param
MAX_DISTANCE = 35 # distance (bins)

CLUTTER_REMOVAL_ENABLED = True

ADAPTIVE_PRESENCE_THRESH = False  
PRESENCE_THRESH = 300     # if ADAPTIVE_PRESENCE_THRESH is False
PRESENCE_CHECK_TIME_INTERVAL = 5  # last X seconds
PRESENCE_DETECTION_RANGE = [2, 15]    # [30, 225] cm

LOCALIZATION_WIN = 2
LOCALIZATION_RANGE = PRESENCE_DETECTION_RANGE

RAM_WINDOW = 5    # Window length for the angle information processing

MOTION_FILTER_TYPE = 'lowpass'
MOTION_FILTER_FREQUENCY = [0.09]   # Hz
MOTION_FILTER_ORDER = 4
MOTION_PROCESSING_WIN = 5  # TODO

MACRO_MOVE_CHECK_WINDOW = 2   # sec/bins

RESP_RANGE = [.1, .5]   # 6 --> 30 bpm
HOLD_INTERVAL = 5   # sec --> Hold breath interval
RESP_THRESH = .5
RESP_PROCESSING_WIN = 30  # TODO
RR_PROCESSING_WIN = 0 # TODO
RESP_PATTERN_WIN = 0  # TODO

HEARTBEAT_RANGE = [.8, 2]
POLAR_ECG_SCALE = .01 # for better plotting
HEARTBEAT_PROCESSING_WIN = 30  # TODO
HR_PROCESSING_WIN = 0  # TODO

###################################################################################################################
## Plotting config: [True/False, subplot_order]
REALTIME_PLOT = True
USE_MATPLOTLIB = True   # True --> Matplotlib to display results,   False --> GUI
GUI_SCREEN_SIZE = [1600, 900]   # TODO: Dynamic set

MATPLOTLIB_PARAM = {
    'font_family': "Times New Roman",
    'font_size': 15,
    'window_width': 10,
    'plot_height': 4,
}

PLOTTING_WIN = 30 # sec  --> for display
PLOTS  = {
    # Radar-related plots
    "PLOT_RTM": [False, 0],   # Range-Time Map
    "PLOT_RAM": [False, 0],   # Range-Angle Map
    "PLOT_MAGNITUDE": [False, 0],
    "PLOT_PHASE": [True, 2],
    "PLOT_RESP_WAVE": [True, 1],
    "PLOT_RESP_RATE": [False, 0],  # TODO
    "PLOT_HEARTBEAT_WAVE": [False, 5],
    "PLOT_HEART_RATE": [False, 5],  # TODO
    "PLOT_FREQUENCY_SPECTRUM": [False, 3], # TODO: How to add different spectrums?
    "PLOT_IQ": [False, 2],   # 1DFFT-based IQ signal
    "SCATTER_IQ": [False, 2],
    "BIN_SELECTION": [False, 3], # TODO

    # Reference-related plots
    "PLOT_VERNIER_BELT_FORCE": [False, 0],
    "PLOT_VERNIER_BELT_RR": [False, 0],  # TODO
    "PLOT_VERNIER_SPIRO_FORCE": [False, 1],  # TODO
    "PLOT_POLAR_H10_ECG": [False, 4],
    "PLOT_POLAR_H10_HR": [False, 1],  # TODO
    "PLOT_MARKER": [False, 1],   # M5stickC press btn
    "PLOT_WITMOTION_1": [False, 1],
    "PLOT_WITMOTION_2": [False, 1],
    "PLOT_HAT_EOG": [False,1],
    "PLOT_HAT_RAW_EYE": [False,2],
    "PLOT_HAT_ACC": [False,3],
}

# Adding comment

M5STICKC_MARKER_SCALER = 20   # scale the marker values for better plotting
RTM_MAX_MAG = 2000    # TODO: Set dynamically
RAM_MAX_MAG = 130000    # TODO: Set dynamically
DISPLAY_STATS = False  # TODO: display different results (motion, rate, current bin, power, ...)

