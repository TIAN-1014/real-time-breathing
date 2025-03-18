import multiprocessing
import logging
import time
import struct
import traceback
import numpy as np

from libs.utils import WisSerial
from sys_config import *
from user_config import *

# READ_CHIRPS_BUF_SIZE = 61760  # Serial port cache buffer size, the corresponding frame number is 18 frames

class RadarController(multiprocessing.Process):
    def __init__(self, state_q, calculation_status=CalculationStatus.free.value, **kwargs):
        super().__init__()
        self.state_q = state_q
        self.out_fft_array_q = kwargs.get('out_fft_array_q')
        self.save_fft_array_q = kwargs.get('save_fft_array_q')
        self.pt_fft_q = kwargs.get('pt_fft_q')
        self.start_radar_flag = kwargs.get('start_radar_flag')
        self.calculation_status = calculation_status
        self.range_matrix_queue = np.zeros((0, RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)
        self.fft_matrix_queue = np.zeros((0, RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)
        self.able_to_calculate_flag = False
        self.RangeMatrixQueueLen = TI_1DFFT_QUEUE_LEN
        self.able_put_flag = False
        self.recording_flag = False # initialize recording Flag
        self.order = 0
        self.put_fft = np.zeros((RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)

    def run(self):
        logging.basicConfig(level=LOG_LEVEL)  # enable show logging

        ti_cli_ser = WisSerial(SERIAL_PORT_NAME, baudrate=921600)
        self.wait_for_reconnect(ti_cli_ser, True)

    @staticmethod
    def send_ti_config(is_new_config):
        # Send configuration to the development board
        if is_new_config:
            try:
                ti_cli_ser = WisSerial(TI_CLI_SERIAL_PORT, baudrate=115200)
                ti_cli_ser.connect()
                config_path = RADAR_CONFIG_FILE_PATH
                with open(config_path, 'r') as f:
                    print(style.BLUE + '\nSending Configuration to radar ...' + style.RESET)
                    config_line = f.readline()
                    while config_line:
                        if config_line.startswith('mmwDemo:/>'):
                            # print(config_line)
                            config_line = config_line.replace('mmwDemo:/>', '')
                            ti_cli_ser.write(config_line)
                            time.sleep(0.1)
                            feedback = ti_cli_ser.read_buffer_line()
                            time.sleep(0.1)
                        config_line = f.readline()
                    print(style.GREEN + 'Radar started ...\n' + style.RESET)
            except:
                print("send_ti_config error")

    def read_data(self, ti_cli_ser):
        is_error = False
        chirp_temp = b''   # New
        flag_header = False   # New
        while not is_error:
            try:
                temp_buffer = ti_cli_ser.read_buffer_line()  
                if b'\x01\x04\x03\x06\x05\x08\x07' in temp_buffer and b'TIAOP\r\n' in temp_buffer:
                    chirp_temp = temp_buffer[1:]
                    # print('data len: ', len(chirp_temp))
                    if len(chirp_temp) == 163: # ???
                        self.analyticalBuffer(chirp_temp)
                elif b'\x01\x04\x03\x06\x05\x08\x07' in temp_buffer and b'TIAOP\r\n' not in temp_buffer:
                    flag_header = True
                    chirp_temp = temp_buffer[1:]
                elif flag_header:
                    chirp_temp += temp_buffer
                    if b'TIAOP\r\n' in temp_buffer:
                        flag_header = False
                        # print('data len: ', len(chirp_temp))
                        if len(chirp_temp) == 163:
                            self.analyticalBuffer(chirp_temp)          
            except:
                traceback.print_exc()
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def wait_for_reconnect(self, ti_cli_ser, is_first_connect=False):
        while not ti_cli_ser.is_open():
            self.send_ti_config(NEED_SEND_TI_CONFIG)
            ti_cli_ser.connect()
        self.read_data(ti_cli_ser)

    def analyticalBuffer(self, data):
        header_length = 8
        timeLen = 4
        step_size = 4

        magic = struct.unpack('Q', data[:header_length])
        timeStamp = struct.unpack('I', data[header_length:(header_length + timeLen)])

        if magic[0] == MAGIC_WORD[self.order]:
            content_start = header_length + timeLen
            range_matrix_real = np.zeros(RANGE_IDX_NUM, dtype=int)
            range_matrix_imag = np.zeros(RANGE_IDX_NUM, dtype=int)
            output_idx = 0
            for rangeIdx in range(0, RANGE_IDX_NUM * step_size, step_size):
                temp_real = struct.unpack('<h', data[(content_start + rangeIdx):(content_start + rangeIdx + 2)])
                temp_imag = struct.unpack('<h', data[(content_start + rangeIdx + 2):(content_start + rangeIdx + 4)])
                range_matrix_real[output_idx] = temp_real[0]
                range_matrix_imag[output_idx] = temp_imag[0]
                output_idx = output_idx + 1
            endtimestamp = struct.unpack('I', data[(content_start+RANGE_IDX_NUM*step_size):(content_start+RANGE_IDX_NUM*step_size+4)])

            range_matrix_all_ant_real = range_matrix_real.reshape(RANGE_IDX_NUM, 1)
            range_matrix_all_ant_imag = range_matrix_imag.reshape(RANGE_IDX_NUM, 1)
            range_fft = range_matrix_all_ant_real + 1j * range_matrix_all_ant_imag

            self.put_fft[:, self.order] = range_fft.reshape(-1)
            self.order = self.order + 1

            if self.order == 8:
                self.order = 0
                # print('StartTime: {}, endTime: {}'.format(timeStamp, endtimestamp))
                # print('put: ', self.put_fft)

                self.pt_fft_q.put(self.put_fft)
                self.put_fft_data(np.reshape(self.put_fft, (1,self.put_fft.shape[0],self.put_fft.shape[1])))
                

                # Recording:
                if self.start_radar_flag.qsize():
                    self.recording_flag = self.start_radar_flag.get()
                if self.recording_flag:
                    self.save_fft_array_q.put(self.put_fft)
                
                self.put_fft = np.zeros((RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)
            
        else:
            self.order = 0

    def put_fft_data(self, range_fft_aoa_ant_cplx):
        """
        send data to out_fft_array_q after the fft_matrix_queue is full
        Note fft_matrix_queue acts like a queue:
        if run_status is busy and sending data failed,
        the first frame of fft_matrix_queue will be discarded when new frame range_fft_aoa_ant_cplx comes

        Parameters
        ----------
        range_fft_aoa_ant_cplx: 3D np.array, 1*indexes*ants
        """
        self.fft_matrix_queue = np.vstack((self.fft_matrix_queue, range_fft_aoa_ant_cplx))
        # print(len(self.fft_matrix_queue))
        if len(self.fft_matrix_queue) >= self.RangeMatrixQueueLen:
            self.able_put_flag = True
            self.fft_matrix_queue = self.fft_matrix_queue[-self.RangeMatrixQueueLen:]
        else:
            self.able_put_flag = False
            # logging.debug(f"put_fft_data collecting: {len(self.fft_matrix_queue)}")

        if self.out_fft_array_q is not None and self.able_put_flag is True:
            self.out_fft_array_q.put(self.fft_matrix_queue) 

            # reset fft_matrix_queue after sending data
            self.fft_matrix_queue = np.zeros((0, RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)

            # debug
            # logging.info("DProc: put_fft_data success")
            # else:
            #     # debug
            #     logging.error("DProc: put_fft_data failed, run_status busy")

    # def put_bpm_data(self, range_fft_aoa_ant_cplx):
    #     """
    #     for realtime bpm proc
    #     """
    #     self.bpm_matrix_queue = np.vstack((self.bpm_matrix_queue, range_fft_aoa_ant_cplx))
    #     if len(self.bpm_matrix_queue) >= BPM_UNIT_LEN:
    #         self.bpm_ready_flag = True
    #         self.bpm_matrix_queue = self.bpm_matrix_queue[-BPM_UNIT_LEN:]
    #     else:
    #         self.bpm_ready_flag = False
    #         # logging.debug(f"put_bpm_data collecting: {len(self.bpm_matrix_queue)}")

    #     if self.out_bpm_array_q is not None and self.bpm_ready_flag is True:
    #         if self.bpm_status.value == 0:
    #             self.out_bpm_array_q.put(self.bpm_matrix_queue)
    #             # reset bpm_matrix_queue after sending data
    #             self.bpm_matrix_queue = np.zeros((0, RANGE_IDX_NUM, TI_2DAoA_VIRTUAL_ANT_NUM), dtype=complex)
    #             # debug
    #             logging.info("DCProc: put_bpm_data success")
    #         else:
    #             # debug
    #             logging.error("DCProc: put_bpm_data failed, bpm_status busy")