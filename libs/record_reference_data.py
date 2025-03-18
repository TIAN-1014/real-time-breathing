import asyncio
import math
import struct
from datetime import datetime
import datetime
from time import strftime, time, sleep
from multiprocessing import Process
from threading import Thread
import pyautogui
import cv2
import os
import numpy as np
from bleak import BleakClient
from bleak.uuids import uuid16_dict
from sys_config import *
from user_config import *
import sys
import mss
sys.stdout.flush()

################# Vernier Respiration Belt Handler ####################################################################################

class VernierBeltController(Process):
    def __init__(self, vernier_belt_connection_q = None, vernier_belt_ref_q=None, sensors = [1,2], period = 50, **kwargs):
        super().__init__()
        self.sensors = sensors
        self.period = period
        self.vernier_belt_ref_q = vernier_belt_ref_q
        self.vernier_belt_connection_q = vernier_belt_connection_q
        self.data = []
        self.vernier_belt_connection_q.put(0)
        self.start_vernier_belt_q = kwargs.get('start_vernier_belt_q')
        self.vernier_belt_realtime_q = kwargs.get("vernier_belt_realtime_q")
        self.start_recording_flag = False
        print("Vernier belt initialized ...")

    def run(self):
        from libs.gdx import gdx 
        self.gdx = gdx()
        if ENABLE_VERNIER_BELT_BLE:
            self.gdx.open_ble(VERNIER_BELT_MODEL) 
        else:
            self.gdx.open_usb()
        self.gdx.select_sensors(self.sensors)
        print(style.GREEN + f"{VERNIER_BELT_MAC} >>> Vernier Belt Device connected" + style.RESET)
        self.start_recording()      
        
    def start_recording(self):
        self.gdx.start(period=self.period)
        while True:   # Add Experiment timing
            try:
                force, respiration = self.get_data()
                timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if self.start_vernier_belt_q.qsize():
                    self.start_recording_flag = self.start_vernier_belt_q.get()
                if self.start_recording_flag:
                    self.vernier_belt_ref_q.put([timeStamp, force, respiration]) 
                    if REALTIME_PLOT:
                        self.vernier_belt_realtime_q.put(force)
                self.vernier_belt_connection_q.put(1) 
            except:
                print('Vernier Respiration Belt issue !!!')
                self.vernier_belt_connection_q.put(0) 
        
    def get_data(self):
        return self.gdx.read()
        
    def close_device(self):
        self.gdx.stop()
        self.gdx.close()

class VernierSpirometerController(Process):
    """
        sensor number =  1, sensor description =  Flow Rate,             sensor units =  L/s
        sensor number =  2, sensor description =  Volume,                sensor units =  L
        sensor number =  3, sensor description =  Differential Pressure, sensor units =  Pa
        sensor number =  5, sensor description =  Adjusted Volume,       sensor units =  L
        sensor number =  6, sensor description =  Cycle Volume,          sensor units =  L
        sensor number =  9, sensor description =  Respiration Rate,      sensor units =  bpm
    """
    def __init__(self, vernier_spiro_ref_q = None, vernier_spiro_connection_q=None, sensors = [1,2,3], period = 50, **kwargs):
        super().__init__()
        self.sensors = sensors
        self.period = period
        self.vernier_spiro_ref_q = vernier_spiro_ref_q
        self.vernier_spiro_connection_q = vernier_spiro_connection_q
        self.data = []
        self.vernier_spiro_connection_q.put(0)
        self.start_vernier_spiro_q = kwargs.get('start_vernier_spiro_q')
        self.vernier_spiro_realtime_q = kwargs.get("vernier_spiro_realtime_q")
        self.start_recording_flag = False
        print("Vernier initialized ...")

    def run(self):
        from libs.gdx import gdx 
        self.gdx = gdx()
        if ENABLE_VERNIER_SPIRO_BLE:
            self.gdx.open_ble(VERNIER_SPIRO_MODEL) 
        else:
            self.gdx.open_usb()
        self.gdx.select_sensors([1,2])
        print(style.GREEN + f"{VERNIER_SPIRO_MAC} >>> Vernier Spirometer Device connected" + style.RESET)
        self.start_recording()      

    def start_recording(self):
        self.gdx.start(period=self.period)
        while True:   # Add Experiment timing
            # try:
            data = self.get_data()
            print(data)
            timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if self.start_vernier_spiro_q.qsize():
                self.start_recording_flag = self.start_vernier_spiro_q.get()
            # if self.start_recording_flag:
            #     self.vernier_spiro_ref_q.put([timeStamp, force, respiration]) 
            #     if REALTIME_PLOT:
            #         self.vernier_spiro_realtime_q.put(force)
            # self.vernier_spiro_connection_q.put(1) 
            # except:
            #     print('Vernier Spirometer issue !!!')
            #     self.vernier_spiro_connection_q.put(0) 
        
    def get_data(self):
        return self.gdx.read()
        
    def close_device(self):
        self.gdx.stop()
        self.gdx.close()
######################################################################################################################################

################### POLAR H10 Class ##################################################################################################
class PolarController(Process):
    def __init__(self, **kwargs):
        super().__init__()
        # self.name = 'Polar Process'
        self.polar_mac_addr = kwargs.get('polar_mac_addr')
        self.polar_ecg_ref_q = kwargs.get('polar_ecg_ref_q')
        self.polar_hr_ref_q = kwargs.get('polar_hr_ref_q')
        self.polar_rri_ref_q = kwargs.get('polar_rri_ref_q')
        self.polar_acc_ref_q = kwargs.get('polar_acc_ref_q')
        self.polar_connection_q = kwargs.get('polar_connection_q')
        self.polar_ecg_realtime_q = kwargs.get('polar_ecg_realtime_q')
        self.polar_hr_realtime_q = kwargs.get('polar_hr_realtime_q')
        

        self.start_polar_q = kwargs.get('start_polar_q')
        self.start_recording_flag = False

        self.uuid16_dict = {v: k for k, v in uuid16_dict.items()}

        ## This is the device MAC ID, please update with your device ID
        self.ADDRESS = self.polar_mac_addr

        ## UUID for model number ##
        self.MODEL_NBR_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
            self.uuid16_dict.get("Model Number String")
        )

        ## UUID for manufacturer name ##
        self.MANUFACTURER_NAME_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
            self.uuid16_dict.get("Manufacturer Name String")
        )

        ## UUID for battery level ##
        self.BATTERY_LEVEL_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
            self.uuid16_dict.get("Battery Level")
        )

        ## UUID for connection establsihment with device ## PMD ==> Polar Measurement Data
        self.PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"

        ## UUID for Request of stream settings ##
        self.PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"

        ## Requests for start streaming:
        # UUID for Request of start stream ##
        self.PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

        # UUID for Request of ECG Stream ##
        self.ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

        # UUID for Request of PPI Stream ## Not managed by PMD
        self.PPI_WRITE = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(self.uuid16_dict.get("Heart Rate Measurement"))

        # # UUID for Request of ACC Stream ##
        self.ACC_WRITE = bytearray([0x02, 0x02, 0x00, 0x01, 0xC8, 0x00, 0x01, 0x01, 0x10, 0x00, 0x02, 0x01, 0x08, 0x00])

        self.polar_connection_q.put(0)
        print(style.GREEN + "Polar initialized ..."+ style.RESET)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.polar_main())

    async def run_routine(self, client=None, debug=False):       
        print(style.GREEN + f"{self.polar_mac_addr} >>> Polar Device connected" + style.RESET)
        self.polar_connection_q.put(1)

        model_number = await client.read_gatt_char(self.MODEL_NBR_UUID)
        print("Model Number: {0}".format("".join(map(chr, model_number))))

        manufacturer_name = await client.read_gatt_char(self.MANUFACTURER_NAME_UUID)
        print("Manufacturer Name: {0}".format("".join(map(chr, manufacturer_name))))

        battery_level = await client.read_gatt_char(self.BATTERY_LEVEL_UUID)
        print("Battery Level: {0}%".format(int(battery_level[0])))

        ## Writing chracterstic description to control point for request of UUID (defined in Initialization) ##
        await client.is_connected()

        att_read = await client.read_gatt_char(self.PMD_CONTROL)
        
        if POLAR_ECG_DATA_FLAG:
            await client.write_gatt_char(self.PMD_CONTROL, self.ECG_WRITE)  # Request ECG data
        if POLAR_ACC_DATA_FLAG:
            await client.write_gatt_char(self.PMD_CONTROL, self.ACC_WRITE)  # Request Acceleration data
        if POLAR_ECG_DATA_FLAG:
            await client.start_notify(self.PMD_DATA, self.data_conv)  ## ECG stream started
        
        await client.start_notify(self.PPI_WRITE, self.handle_hr_data)  # start HR data stream (which is not broadcasted through PMD)

        while True:
            await asyncio.sleep(0.001)
            if self.start_polar_q.qsize():
                print('changing recording stat')
                self.start_recording_flag = self.start_polar_q.get()     
        
    ## Bit conversion of the Hexadecimal stream: case ECG and Acceleration (PMD streamed data)
    def data_conv(self, sender, data):
        if data[0] == 0x02:  # case acceleration data
            frame_type = data[9]
            resolution = (frame_type + 1) * 8
            step = math.ceil(resolution / 8.0)
            samples = data[10:]
            offset = 0
            timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            while offset < len(samples):
                x = self.convert_array_to_signed_int(samples, offset, step)
                offset += step
                y = self.convert_array_to_signed_int(samples, offset, step)
                offset += step
                z = self.convert_array_to_signed_int(samples, offset, step)
                offset += step
                if self.start_recording_flag:
                    self.polar_acc_ref_q.put([timeStamp, x,y,z])

        if data[0] == 0x00:  # Case ECG
            step = 3   # 3byte data
            samples = data[10:]
            offset = 0
            timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            while offset < len(samples):
                ecg = self.convert_array_to_signed_int(samples, offset, step)
                offset += step
                if self.start_recording_flag:
                    self.polar_ecg_ref_q.put([timeStamp, ecg])
                    self.polar_ecg_realtime_q.put(ecg)

    def handle_hr_data(self, sender, data):
        step = 1  
        samples = data
        offset = 0
        hr = self.convert_array_to_signed_int(samples, 1, step)
        rri = self.convert_array_to_signed_int(samples, 2, 2)
        timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        if self.start_recording_flag:
            self.polar_hr_ref_q.put([timeStamp, hr])
            # self.polar_rri_ref_q.put([timeStamp,rri])
            self.polar_hr_realtime_q.put(hr)

    async def polar_main(self):
        try:
            async with BleakClient(self.ADDRESS) as client:
                tasks = [
                    asyncio.ensure_future(self.run_routine(client=client, debug=True))
                ]
                await asyncio.gather(*tasks)

        except: pass

    def convert_array_to_signed_int(self, data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=True,
        )

    def convert_to_unsigned_long(self, data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=False,
        )
    
######################################################################################################
## WitMotion:
#####################
class WitMotionController(Process):
    def __init__(self, witmotion_mac = None, witmotion_data_q = None, start_witmotion_q = None, witmotion_realtime_q=None, **kwargs):
        super().__init__()
        self.witmotion_data_q = witmotion_data_q
        self.start_witmotion_q = start_witmotion_q
        self.witmotion_realtime_q = witmotion_realtime_q
        self.start_recording_flag = False
        self.witmotion_mac_addr = witmotion_mac
        self.UUID_READ = '0000ffe4-0000-1000-8000-00805f9a34fb'

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.witmotion_main())

    async def run_routine(self, client=None, debug=False):
        print(style.GREEN + f"{self.witmotion_mac_addr} >>> Witmotion Device connected" + style.RESET)
        await client.is_connected()
        await client.start_notify(self.UUID_READ, self.data_conv) 

        while True:
            await asyncio.sleep(1)
            if self.start_witmotion_q.qsize():
                self.start_recording_flag = self.start_witmotion_q.get()

    def data_conv(self, sender, data):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        for i in range(0, len(data), 20):
            if data[i]==0x55:
                try:
                    if data[i+1]==0x61:
                        # print(i)
                        # Acceleration data:
                        a_x = int.from_bytes(bytearray(data[i+2:i+4]), byteorder="little", signed=True)/32768 * 16
                        a_y = int.from_bytes(bytearray(data[i+4:i+6]), byteorder="little", signed=True)/32768 * 16
                        a_z = int.from_bytes(bytearray(data[i+6:i+8]), byteorder="little", signed=True)/32768 * 16
                        # Angular Velocity:
                        w_x = int.from_bytes(bytearray(data[i+8:i+10]), byteorder="little", signed=True)/32768 * 2000
                        w_y = int.from_bytes(bytearray(data[i+10:i+12]), byteorder="little", signed=True)/32768 * 2000
                        w_z = int.from_bytes(bytearray(data[i+12:i+14]), byteorder="little", signed=True)/32768 * 2000
                        # Angle:
                        angl_x = int.from_bytes(bytearray(data[i+14:i+16]), byteorder="little", signed=True)/32768 * 180
                        angl_y = int.from_bytes(bytearray(data[i+16:i+18]), byteorder="little", signed=True)/32768 * 180
                        angl_z = int.from_bytes(bytearray(data[i+18:i+20]), byteorder="little", signed=True)/32768 * 180
                        if self.start_recording_flag:
                            self.witmotion_data_q.put([timestamp, a_x, a_y, a_z, w_x, w_y, w_z, angl_x, angl_y, angl_z])
                            self.witmotion_realtime_q.put([a_x, a_y, a_z, w_x, w_y, w_z, angl_x, angl_y, angl_z])
                except Exception as e: print(style.RED +f'{self.witmotion_mac_addr} >>> {e}, index: {i}' + style.RESET)
    async def witmotion_main(self):
        try:
            async with BleakClient(self.witmotion_mac_addr) as client:
                print(client)
                tasks = [
                    asyncio.ensure_future(self.run_routine(client=client, debug=True))
                ]
                await asyncio.gather(*tasks)
        except: pass

######################################################################################################
## M5Stick-C:
#####################
class M5StickCController(Process):
    def __init__(self, mac_address = None, start_m5stickc_q=None , m5stickc_data_q = None, role=None , **kwargs):
        super().__init__()
        self.mac_address = mac_address
        self.start_recording_flag = False
        self.start_m5stickc_q = start_m5stickc_q
        self.m5stickc_data_q = m5stickc_data_q 
        self.role = role
        self.UUID_READ = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.m5stick_realtime_q = kwargs.get('m5stick_realtime_q')

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.m5stick_main())

    async def run_routine(self, client=None, debug=False):
        print(style.GREEN + f"{self.mac_address} >>> M5Stick {self.role} Device connected" + style.RESET)
        
        await client.is_connected()
        await client.start_notify(self.UUID_READ, self.data_conv) 

        while True:
            await asyncio.sleep(1)
            if self.start_m5stickc_q.qsize():
                self.start_recording_flag = self.start_m5stickc_q.get()

    def data_conv(self, sender, data):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # if USE_M5STICK_ACC1 or USE_M5STICK_ACC2: #### TODO: need correct test 
        #     value = struct.unpack('fff', data)
        # else:
        value = int.from_bytes(bytearray(data), byteorder="little", signed=True)
        
        if self.start_recording_flag:
            # if USE_M5STICK_ACC1 or USE_M5STICK_ACC2: ## TODO: need correct test 
            #     self.m5stickc_data_q.put([timestamp, value[0], value[1], value[2]])
            # else:
            self.m5stickc_data_q.put([timestamp, value])
            self.m5stick_realtime_q.put(int(value))
            
    async def m5stick_main(self):
        try:
            async with BleakClient(self.mac_address) as client:
                print(client)
                tasks = [
                    asyncio.ensure_future(self.run_routine(client=client, debug=True))
                ]
                await asyncio.gather(*tasks)
        except: pass

######################################################################################################
## HAT: (M5Stick-C-based)
#####################
class HATController(Process):
    def __init__(self, mac_address = None, start_m5stickc_q=None , m5stickc_data_q = None, m5stickc_data_realtime_q=None, **kwargs):
        super().__init__()
        self.mac_address = mac_address
        self.start_recording_flag = False
        self.start_m5stickc_q = start_m5stickc_q
        self.m5stickc_data_q = m5stickc_data_q 
        self.m5stickc_data_realtime_q = m5stickc_data_realtime_q
        self.UUID_READ = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        print("inside HATBLE")

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.m5stick_main())

    async def run_routine(self, client=None, debug=False):

        print(style.GREEN + f"{self.mac_address} >>> HAT Device Connected" + style.RESET)
        
        await client.is_connected()
        await client.start_notify(self.UUID_READ, self.data_conv) 

        while True:
            await asyncio.sleep(1)
            if self.start_m5stickc_q.qsize():
                self.start_recording_flag = self.start_m5stickc_q.get()

    def data_conv(self, sender, data):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        value = struct.unpack('fffff', data)
        if self.start_recording_flag: # [AccX, AccY, AccZ, sensor_data, EOG]
            # print([value[0], value[1], value[2], value[3], value[4]])
            self.m5stickc_data_q.put([timestamp, value[0], value[1], value[2], value[3], value[4]])
            self.m5stickc_data_realtime_q.put([value[0], value[1], value[2], value[3], value[4]])
            
    async def m5stick_main(self):
        try:
            async with BleakClient(self.mac_address) as client:
                print(client)
                tasks = [
                    asyncio.ensure_future(self.run_routine(client=client, debug=True))
                ]
                await asyncio.gather(*tasks)
        except: pass
  
######################################################################################################
## Screen Recorder --> Multi-Threading: 
#########################################
# global variables
stop_thread = False             # controls thread execution
img = None                      # stores the image retrieved by the camera

class ScreenRecorder(Process):
    def __init__(self, camera_index, **kwargs):
        super().__init__()
        if not os.path.exists(LOCAL_CACHE_PATH):
            os.makedirs(LOCAL_CACHE_PATH)
        self.output_path = f"{LOCAL_CACHE_PATH}{OUT_FILE}.mkv"
        self.start_video_q = kwargs.get('start_video_q')
        self.record_flag = False
        self.camera_index = camera_index
        
    def run(self):
        count = 0
        if USE_CAMERA:
            global img, stop_thread
            self.vid_capture = cv2.VideoCapture(self.camera_index)
            window_name = "Camera Live Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, CAM_SIZE[0], CAM_SIZE[1])
            cv2.moveWindow(window_name, CAM_POS[0], CAM_POS[1])

        # Screen capture dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        self.vid_cod = cv2.VideoWriter_fourcc(*'X264')
        self.output = cv2.VideoWriter(self.output_path, self.vid_cod, 12, (self.screen_width, self.screen_height))

        if USE_CAMERA:
            # start the capture thread: reads frames from the camera (non-stop) and stores the result in img
            t = Thread(target=self.start_capture_thread, args=(self.vid_capture,), daemon=True) # a deamon thread is killed when the application exits
            t.start()

        # Define the screen region to capture
        monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        count = 0

        # Capture the screen
        with mss.mss() as sct:
            # Update frame
            while True:
                if RECORD_VIDEO:
                    # Check for start recording Flag:
                    if self.start_video_q.qsize():
                        count+=1 # record number of entries
                        self.record_flag = self.start_video_q.get()
                        start_time = datetime.datetime.now() 

                    if self.record_flag:
                        # Capture the screen
                        frame = sct.grab(monitor)
                        frame_np = np.array(frame)
                        frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
                        self.output.write(frame_cv) 

                if USE_CAMERA and LIVE_FEED:
                    # draw FPS text and display image
                    if (img is not None):
                        if self.record_flag:
                            now = datetime.datetime.now()
                            cv2.putText(img, f'{now-start_time}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow(window_name, img)

                # wait 1ms for ESC to be pressed
                key = cv2.waitKey(1)
                if (key == 27) or count==2:
                    stop_thread = True
                    break
                
        try:
            # close the already opened camera
            self.vid_capture.release()
        
            # close the already opened file
            self.output.release()
        
            # close the window and de-allocate any associated memory usage
            cv2.destroyAllWindows()
        except:
            pass

        print("Processing video recording end.")
        self.start_video_q.put(True)

    def start_capture_thread(self, cap):
        global img, stop_thread

        # continuously read fames from the camera
        while True:
            _, img = cap.read()

            if (stop_thread):
                break

# TODO: Add EEG data --> Muse2/Pill Device


 
# ######################################################################################################
# ## Camera Controller --> Multi-Threading: 
# #########################################
# # global variables
# stop_thread = False             # controls thread execution
# img = None                      # stores the image retrieved by the camera
# class CameraController(Process):
#     def __init__(self, camera_index, **kwargs):
#         super().__init__()
#         if not os.path.exists(LOCAL_CACHE_PATH):
#             os.makedirs(LOCAL_CACHE_PATH)
#         self.output_path = f"{LOCAL_CACHE_PATH}{OUT_FILE}.mp4"
#         self.start_video_q = kwargs.get('start_video_q')
#         self.record_flag = False
#         self.camera_index = camera_index

#     def run(self):
#         count = 0

#         self.vid_capture = cv2.VideoCapture(self.camera_index)

#         # retrieve properties of the capture object
#         cap_width = self.vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
#         cap_height = self.vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         cap_fps = self.vid_capture.get(cv2.CAP_PROP_FPS)
#         fps_sleep = int(1000 / cap_fps)
#         print(style.GREEN + f'* Capture width: {cap_width}' + style.RESET)
#         print(style.GREEN + f'* Capture height: {cap_height}' + style.RESET)
#         print(style.GREEN + f'* Capture FPS: {cap_fps} wait time between frames: {fps_sleep}' + style.RESET)

#         # start the capture thread: reads frames from the camera (non-stop) and stores the result in img
#         t = Thread(target=self.start_capture_thread, args=(self.vid_capture,), daemon=True) # a deamon thread is killed when the application exits
#         t.start()

#         # video writer --> save to mp4
#         if RECORD_VIDEO:
#             self.vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
#             self.output = cv2.VideoWriter(self.output_path, self.vid_cod, CAM_FPS, CAM_RES)

#         # initialize time and frame count variables
#         last_time = datetime.datetime.now()
#         frames = 0
#         cur_fps = 0

#         # Update frame
#         while(True):
#             # blocks until the entire frame is read
#             frames += 1

#             # measure runtime: current_time - last_time
#             delta_time = datetime.datetime.now() - last_time
#             elapsed_time = delta_time.total_seconds()

#             # compute fps but avoid division by zero
#             if (elapsed_time != 0):
#                 cur_fps = np.around(frames / elapsed_time, 1)
            
#             if RECORD_VIDEO:
#                 # Check for start recording Flag:
#                 if self.start_video_q.qsize():
#                     count+=1 # record number of entries
#                     self.record_flag = self.start_video_q.get()
#                     start_time = datetime.datetime.now()

#                 if self.record_flag:
#                     now = datetime.datetime.now()
#                     cv2.putText(img, f'{now-start_time}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, cv2.LINE_AA)
#                     self.output.write(img)  # save frame
            
#             if LIVE_FEED:
#                 # draw FPS text and display image
#                 if (img is not None):
#                     # cv2.putText(img, 'FP: ' + str(cur_fps), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, cv2.LINE_AA)
#                     cv2.imshow("Camera Feed", img)

#             # wait 1ms for ESC to be pressed
#             key = cv2.waitKey(1)
#             if (key == 27) or count==2:
#                 stop_thread = True
#                 break

#         # close the already opened camera
#         self.vid_capture.release()
#         # close the already opened file
#         self.output.release()
#         # close the window and de-allocate any associated memory usage
#         cv2.destroyAllWindows()
        
#         print("Processing video recording end.")
#         self.start_video_q.put(True)

#     def start_capture_thread(self, cap):
#         global img, stop_thread

#         # continuously read fames from the camera
#         while True:
#             _, img = cap.read()

#             if (stop_thread):
#                 break
