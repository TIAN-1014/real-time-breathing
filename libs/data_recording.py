import os
import datetime
from time import sleep

from user_config import *
from sys_config import *
from libs.utils import beep
from libs.data_processing.range_processing import range_processing

from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()    
scheduler.start()

import codecs
import socket
import struct
from enum import Enum
import numpy as np

###################################### DCA1000 Data Capture ##################################################
class CMD(Enum):
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'
    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    CONFIG_EEPROM_CMD_CODE = '0400'
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'
    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'
    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'

    def __str__(self):
        return str(self.value)

# MESSAGE = codecs.decode(b'5aa509000000aaee', 'hex')
CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'

# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

# DYNAMIC
BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] *
                  ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes'])
# print(f"UINT16_IN_PACKET: {BYTES_IN_FRAME}")

BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
# print(f"BYTES_IN_FRAME_CLIPPED: {BYTES_IN_FRAME_CLIPPED}")

PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
# print(f"PACKETS_IN_FRAME: {PACKETS_IN_FRAME}")

PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET
# print(f"PACKETS_IN_FRAME_CLIPPED: {PACKETS_IN_FRAME_CLIPPED}")

UINT16_IN_PACKET = BYTES_IN_PACKET // 2
# print(f"UINT16_IN_PACKET: {UINT16_IN_PACKET}")

UINT16_IN_FRAME = BYTES_IN_FRAME // 2
# print(f"UINT16_IN_FRAME: {UINT16_IN_FRAME}")


class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from


    General steps are as follows:
        1. Power cycle DCA1000 and XWR1xxx sensor
        2. Open mmWaveStudio and setup normally until tab SensorConfig or use lua script
        3. Make sure to connect mmWaveStudio to the board via ethernet
        4. Start streaming data
        5. Read in frames using class

    Examples:
        >>> dca = DCA1000()
        >>> adc_data = dca.read(timeout=.1)
        >>> frame = dca.organize(adc_data, 128, 4, 256)

    """

    def __init__(self, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096):
        # Save network data
        # self.static_ip = static_ip
        # self.adc_ip = adc_ip
        # self.data_port = data_port
        # self.config_port = config_port

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

        self.data = []
        self.packet_count = []
        self.byte_count = []

        self.frame_buff = []

        self.curr_buff = None
        self.last_frame = None
        self.next_frame_data=[]

        self.lost_packets = None

        self.previous_packet=0
        self.already_have_beginning_of_frame=False

    def configure(self):
        """Initializes and connects to the FPGA

        Returns:
            None

        """
        # SYSTEM_CONNECT_CMD_CODE
        # 5a a5 09 00 00 00 aa ee
        print(self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE))

        # READ_FPGA_VERSION_CMD_CODE
        # 5a a5 0e 00 00 00 aa ee
        print(self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE))

        # CONFIG_FPGA_GEN_CMD_CODE
        # 5a a5 03 00 06 00 01 02 01 02 03 1e aa ee
        print(self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', 'c005350c0000'))

        # CONFIG_PACKET_DATA_CMD_CODE 
        # 5a a5 0b 00 06 00 c0 05 35 0c 00 00 aa ee
        print(self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c005350c0000'))

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()

    def read(self, timeout=1):
        """ Read in a single packet via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Frame buffer
        ret_frame = np.zeros(UINT16_IN_FRAME, dtype=np.uint16)    # old: uint16

        ############################# OLD #############################################
        # # Wait for start of next frame
        # while True:
        #     packet_num, byte_count, packet_data = self._read_data_packet()
        #     # print(packet_num, len(packet_data))
        #     if byte_count % BYTES_IN_FRAME_CLIPPED == 0:
        #         packets_read = 1
        #         ret_frame[0:UINT16_IN_PACKET] = packet_data
        #         break
        ###############################################################################

        # instead of wait for next frame, get the already saved data from the previous dca.read
        #If we already have the previous frame
        if self.already_have_beginning_of_frame:
            self.already_have_beginning_of_frame=False
            packets_read=1
            ret_frame[0:UINT16_IN_PACKET]=self.frame_first_packet
        else:
            # Wait for start of next frame
            while True:
                packet_num, byte_count, packet_data = self._read_data_packet()
                if byte_count % BYTES_IN_FRAME_CLIPPED == 0:
                    packets_read = 1
                    ret_frame[0:UINT16_IN_PACKET] = packet_data
                    break

        # Read in the rest of the frame            
        while True:
            packet_num, byte_count, packet_data = self._read_data_packet()
            packets_read += 1

            if byte_count % BYTES_IN_FRAME_CLIPPED == 0:
                self.lost_packets = PACKETS_IN_FRAME_CLIPPED - packets_read
                self.frame_first_packet=packet_data
                self.already_have_beginning_of_frame=True
                return ret_frame

            curr_idx = ((packet_num - 1) % PACKETS_IN_FRAME_CLIPPED)
            try:
                ret_frame[curr_idx * UINT16_IN_PACKET:(curr_idx + 1) * UINT16_IN_PACKET] = packet_data
            except:
                pass

            if packets_read > PACKETS_IN_FRAME_CLIPPED:
                packets_read = 0

    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Helper function to send a single commmand to the FPGA

        Args:
            cmd (CMD): Command code to send to the FPGA
            length (str): Length of the body of the command (if any)
            body (str): Body information of the command
            timeout (int): Time in seconds to wait for socket data until timeout

        Returns:
            str: Response message

        """
        # Create timeout exception
        self.config_socket.settimeout(timeout)

        # Create and send message
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        return resp

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        # additional stuff
        # if packet_num-self.previous_packet>1:
        #     print(f"skipped {packet_num-self.previous_packet-1} packets")

        self.previous_packet=packet_num
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)    # old: uint16
        return packet_num, byte_count, packet_data

    def _listen_for_error(self):
        """Helper function to try and read in for an error message from the FPGA

        Returns:
            None

        """
        self.config_socket.settimeout(None)
        msg = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        if msg == b'5aa50a000300aaee':
            print('stopped:', msg)

    def _stop_stream(self):
        """Helper function to send the stop command to the FPGA

        Returns:
            str: Response Message

        """
        return self._send_command(CMD.RECORD_STOP_CMD_CODE)

    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)

        # Separate IQ data
        # Using mmWave Studio
        # ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        # ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]

        # Using CLI:
        ret[0::2] = raw_frame[2::4] + 1j * raw_frame[0::4]
        ret[1::2] = raw_frame[3::4] + 1j * raw_frame[1::4]
        return ret.reshape((num_chirps, num_rx, num_samples))
    
#####################################################################################################################

def control_recording(flag):
    if USE_RADAR:
        START_RADAR_FLAG.put(flag)

    if USE_DCA1000: # TODO
        if flag:
            command = f".\DCA1000EVM_CLI_Control.exe start_record .\{DCA_CONFIG_FILE}"
        else:
            command = f".\DCA1000EVM_CLI_Control.exe stop_record .\{DCA_CONFIG_FILE}"
        os.system(command)

    if USE_VERNIER_BELT:
        START_VERNIER_BELT_FLAG.put(flag)

    if USE_VERNIER_SPIRO:
        START_VERNIER_BELT_FLAG.put(flag)

    if USE_POLAR:
        START_POLAR_FLAG.put(flag)

    if RECORD_VIDEO:   # Start camera recording
        START_VIDEO_FLAG.put(flag)

    if USE_WITMOTION1:
        START_WITMOTION1_FLAG.put(flag)

    if USE_WITMOTION2:
        START_WITMOTION2_FLAG.put(flag)
    
    if USE_M5STICKC_MARKER:
        START_M5STICKC_MARKER_FLAG.put(flag)
    
    if USE_HAT:
        START_HAT_FLAG.put(flag)

    
    beep(BEEP_START_FREQUENCY, BEEP_DURATION_START, 'start' if flag else 'end')


###########################################################################################################################################################################
# Configure DCA1000:
# 1- FPGA:  TODO: check on the FPGA config process
def configFPGA():
    command = f".\DCA1000EVM_CLI_Control.exe fpga .\{DCA_CONFIG_FILE}"
    print(command)
    os.system(command)
    sleep(1)

# 2- Setup recording process:
def configRecording():
    command = f".\DCA1000EVM_CLI_Control.exe record .\{DCA_CONFIG_FILE}"
    os.system(command)
    sleep(1)

def start_timed_recording(**kwargs):
    if USE_DCA1000: # TODO
        # Configure DCA1000
        configFPGA()
        configRecording()
        
        # Initialize DCA1000 Object
        dca = DCA1000()     
        
    # Start the recording
    control_recording(True)
    start_experiment = datetime.datetime.now()

    end_experiment = start_experiment + datetime.timedelta(seconds=EXPERIMENT_TIME)
    
    # create scheduled jobs for experiment
    for order, experiment in enumerate(RECORDING_SEQUENCE):
        if datetime.datetime.now()>end_experiment:
            break
        start_action = datetime.datetime.now()
        beep(BEEP_NEXT_EXPERIMENT, BEEP_DURATION_START, f'experiment {order}') # Alert for start next experiment
        if experiment[1]:
            scheduler.add_job(beep, 'interval', [BEEP_ACTION_FREQUENCY,BEEP_ACTION_DURATION, f'experiment {order}'],seconds=experiment[0]//experiment[1], id='action_moment')  # beeps for actions
        while datetime.datetime.now()<(start_action + datetime.timedelta(seconds=experiment[0])):
            if datetime.datetime.now()>end_experiment:
                if USE_DCA1000: # TODO
                    # Collect data
                    adc_data = dca.read()
                    
                    # Re-organize data
                    frame = dca.organize(adc_data, CHIRP_LOOP*TX_ANT, RX_ANT, 64)

                    # Perform Range FFT:
                    radar_cube = range_processing(frame)

                    # TODO: Send data for processing/saving

                break
        try:
            scheduler.remove_job('action_moment')
        except: pass  
    
    # stop the recording if experiment mode
    control_recording(False)
    try:
        scheduler.remove_job('action_moment')    # delete any scheduled jobs
    except: pass

    print(f"Recording Finished ;) at {datetime.datetime.now()}")   





