from enum import unique, Enum
from multiprocessing import Queue, Value
from time import strftime
import serial.tools.list_ports
import numpy as np

from user_config import *

# Print Colors
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
@unique
class CalculationStatus(Enum):
    free = 0
    busy = 1

# Data Management and control Queues
# 1/ Start Recordings Flags:
START_RADAR_FLAG = Queue()
START_VIDEO_FLAG = Queue()
START_SAVE_FLAG = Queue()
START_POLAR_FLAG = Queue()
START_VERNIER_BELT_FLAG = Queue()
START_VERNIER_SPIRO_FLAG = Queue()
START_WITMOTION1_FLAG = Queue()
START_WITMOTION2_FLAG = Queue()
START_M5STICKC_ACC1_FLAG = Queue()
START_M5STICKC_ACC2_FLAG = Queue()
START_M5STICKC_MARKER_FLAG = Queue()
START_HAT_FLAG = Queue()
START_MUSE_FLAG = Queue()
START_MEDIA_FLAG = Queue()

# 2/ Data Queues:
POLAR_ECG_DATA = Queue()
POLAR_ECG_DATA_REALTIME = Queue()
POLAR_HR_DATA = Queue()
POLAR_HR_DATA_REALTIME = Queue()
POLAR_ACC_DATA = Queue()
POLAR_RRI_DATA = Queue()

VERNIER_BELT_FORCE_DATA = Queue()
VERNIER_BELT_FORCE_DATA_REALTIME = Queue()
VERNIER_BELT_BR_DATA = Queue()

VERNIER_SPIRO_FORCE_DATA = Queue()
VERNIER_SPIRO_FORCE_DATA_REALTIME = Queue()
VERNIER_SPIRO_BR_DATA = Queue()

WITMOTION1_DATA = Queue()
WITMOTION1_DATA_REALTIME = Queue()
WITMOTION2_DATA = Queue()
WITMOTION2_DATA_REALTIME = Queue()

M5STICKC_ACC1_DATA = Queue()
M5STICKC_ACC2_DATA = Queue()
M5STICKC_MARKER_DATA = Queue()
M5STICKC_MARKER_DATA_REALTIME = Queue()
HAT_DATA = Queue()
HAT_DATA_REALTIME = Queue()
MUSE_DATA = Queue()
BEEP_DATA = Queue()

POLAR_CONNECTED = Queue()
VERNIER_BELT_CONNECTED = Queue()
VERNIER_SPIRO_CONNECTED = Queue()
VIDEO_PROCESSED = Queue() 

polar_connection_q = Queue()
vernier_belt_connection_q = Queue()
vernier_spiro_connection_q = Queue()

CALIBRATION_FLAG = Queue()
GUI_QUIT_Q = Queue()

# FIFO Queues
out_fft_array_q = Queue()    # Queue holding 1DFFT
save_fft_array_q = Queue()   # Queue holding 1DFFT (for saving)
out_fft_phase_q = Queue()    # Holds the phase queue of the 1DFFT
out_range_profile_q = Queue()# Queue to hold range profile data for heatmap plot
out_bpm_array_q = Queue()
pt_fft_q = Queue()

range_data_q = Queue()     # Holds the processed amplitude data (for plotting range heatmap)
phase_data_q = Queue()     # Holds the processed unwrapped phase data (for plotting)

state_q = Queue()            # system status
bpm_status = Queue()

CALCULATION_STATUS = CalculationStatus.free.value

calculation_status = Value("i", CALCULATION_STATUS)

STATUS_CONNECTED = 1
STATUS_UNCONNECTED = 0

LOG_LEVEL = 10 #NOTSET=0, DEBUG=10, INFO=20, WARN=30, ERROR=40, and CRITICAL=50
KIVY_LOG = True

# Radar Parameters --> # TODO: Get from profile
NEED_SEND_TI_CONFIG = True
DCA_CONFIG_FILE = 'profiles/ConfigFile.json'
ADC_PARAMS = {'chirps': 16,  # 32
              'rx': 4,
              'tx': 3,
              'samples': 64,
              'IQ': 2,
              'bytes': 2}

if VERSION == "8_ANT":
    RADAR_CONFIG_FILE_PATH = 'profiles/60ghz_25fps_6843_8ant.txt'
    TI_2DAoA_VIRTUAL_ANT_NUM = 8
    TI_1DFFT_QUEUE_LEN = 25
    FRAME_PERIODICITY = 40 # ms
    MAGIC_WORD = [0x708050603040101, 0x708050603040102, 0x708050603040103, 0x708050603040104,
                      0x708050603040105, 0x708050603040106, 0x708050603040107, 0x708050603040108]
    SEARCHING_AZIMUTH_DOA_RANGE = np.arange(-70, 70, 2) 
    SEARCHING_ELEVATION_DOA_RANGE = np.arange(-70, 70, 2)
elif VERSION == "1_ANT":
    RADAR_CONFIG_FILE_PATH = 'profiles/60ghz_100fps_6843_1ant.txt'
    TI_2DAoA_VIRTUAL_ANT_NUM = 1
    TI_1DFFT_QUEUE_LEN = 100
    FRAME_PERIODICITY = 10 # ms
    MAGIC_WORD = 0x708050603040101
    SEARCHING_AZIMUTH_DOA_RANGE = np.arange(-70, 70, 2)   # dummy
    SEARCHING_ELEVATION_DOA_RANGE = np.arange(-70, 70, 2) # dummy
else:
    MAGIC_WORD = 0x708050603040102  # (old/1843-16fPS)

TI_CLI_SERIAL_PORT = ''
SERIAL_PORT_NAME = ''

ports = list(serial.tools.list_ports.comports())
for p in ports:
    # UART Port: Control of the board
    if 'Enhanced COM Port' in p.description:
        TI_CLI_SERIAL_PORT = p.device 
        print(f"Found Control port: {p.device}")

    # Data Port: Receiving radar data
    if 'Standard COM Port' in p.description:
        SERIAL_PORT_NAME = p.device 
        print(f"Found Data port: {p.device}")

RANGE_IDX_NUM = 35  # 6843  1 Ant / 8 ANT

RADAR_DEVICE = "TI-IWR6843ODS"
START_FREQUENCY = 60  # GHz   
CHIRP_LOOP = 16
RANGE_BINS = 64
CHIRP_SLOPE = 32  # MHz/us
CHIRP_END_TIME = 36 # us
ADC_START = 2 # us
CHIRP_IDLE_TIME =   15  # us
ADC_SAMPLE_RATE =  2000   #ksps  ---> Fast time sampling (over 1 chirp)
TX_ANT = 3
RX_ANT = 4

ANTENNAS = TX_ANT*RX_ANT
LIGHT_SPEED =299792458 # m/s
FPS = int(1e3//FRAME_PERIODICITY)  # Hz 
ADC_SAMPLE_PERIOD_USEC = 1e3 / ADC_SAMPLE_RATE * RANGE_BINS
BANDWIDTH = CHIRP_SLOPE*ADC_SAMPLE_PERIOD_USEC*1e-3 # GHz
RANGE_RESOLUTION = round(LIGHT_SPEED/(2*BANDWIDTH*1e9), 2) # m

BEEP_DURATION_START = 1000   # milliseconds
BEEP_START_FREQUENCY = 440   # Hz
BEEP_NEXT_EXPERIMENT = 600   # Hz
BEEP_ACTION_FREQUENCY = 800  # Hz
BEEP_ACTION_DURATION = 1000  # milliseconds
RESULTS_DICT_Q = Queue()

LOCAL_CACHE_PATH = f'data/{RESEARCH_TOPIC}/{strftime("%y%m%d")}/{SUBJECT}/'
OUT_FILE = f'{SUBJECT}_{EXPERIMENT_CONDITION}_{strftime("%y%m%d_%H%M")}'

EXPERIMENT_TIME = int(np.sum([i[0] for i in RECORDING_SEQUENCE]))

# Plot options
orders = []
for sig in PLOTS:
    if PLOTS[sig][0]:
        orders.append(PLOTS[sig][1])
N_SUBPLOTS = np.max(orders)
MATPLOTLIB_FIGURE_SIZE = (MATPLOTLIB_PARAM["window_width"], N_SUBPLOTS*MATPLOTLIB_PARAM["plot_height"])

#########################
## Server Param
#########################
USE_SERVER = False
SERVER = 'localhost'
USER = 'username'
PASSWORD = 'password'
PORT = '3306'
DATABASE = 'vitalsigns'

WAIT_CONNECTION = 10 + PLOTTING_WIN