import numpy as np
import pandas as pd
from sys_config import *
from user_config import *
import os

def save_1dfft(**kwargs):
    data_dict = kwargs.get("data_dict")
    save_fft_array_q = kwargs.get("save_fft_array_q")
    fft_data = []
    for i in range(save_fft_array_q.qsize()):
        fft_data.append(save_fft_array_q.get())
    fft_data = np.asarray(fft_data)
    print(fft_data.shape)
    # with pd.ExcelWriter(f"{LOCAL_CACHE_PATH}{OUT_FILE}.xlsx") as writer:
    for ant in range(fft_data.shape[2]):
        data_dict[f"radar-ANT-{ant}"] = fft_data[:,:,ant]

def save_ref_data(**kwargs):
    print('\n\nStarting excel file creation process ...')
    reference_data = kwargs.get("data_dict")
    polar_ecg_ref_q = kwargs.get('polar_ecg_ref_q')
    polar_hr_ref_q = kwargs.get('polar_hr_ref_q')
    polar_rri_ref_q = kwargs.get('polar_rri_ref_q')
    polar_acc_ref_q = kwargs.get('polar_acc_ref_q')
    vernier_belt_ref_q = kwargs.get("vernier_belt_ref_q")
    # m5stick_acc1_ref_q = kwargs.get("m5stickc_acc1_ref_q")
    # m5stick_acc2_ref_q = kwargs.get("m5stickc_acc2_ref_q")
    m5stickc_marker_ref_q = kwargs.get("m5stickc_marker_ref_q")
    hat_ref_q = kwargs.get("hat_ref_q")
    witmotion1_ref_data_q = kwargs.get("witmotion1_ref_data_q")
    witmotion2_ref_data_q = kwargs.get("witmotion2_ref_data_q")
    
    if USE_POLAR:
        if polar_ecg_ref_q.qsize():
            polar_ecg = []
            for _ in range(polar_ecg_ref_q.qsize()):
                polar_ecg.append(polar_ecg_ref_q.get())
            polar_ecg = np.asarray(polar_ecg)
            print(f'raw polar ecg data shape is : {polar_ecg.shape}')
            reference_data['polar_ecg'] = {}
            try:
                reference_data['polar_ecg']['timestamp'] = polar_ecg[:, 0]
                reference_data['polar_ecg']['ECG'] = np.asarray(polar_ecg[:, 1], dtype=float)
            except: pass

        if polar_hr_ref_q.qsize():
            polar_hr = []
            for _ in range(polar_hr_ref_q.qsize()):
                polar_hr.append(polar_hr_ref_q.get())
            polar_hr = np.asarray(polar_hr)
            print(f'raw polar hr data shape is : {polar_hr.shape}')
            reference_data['polar_hr'] = {}
            try:
                reference_data['polar_hr']['timestamp'] = polar_hr[:, 0]
                reference_data['polar_hr']['HR'] = np.asarray(polar_hr[:, 1], dtype=float)
            except: pass
        
        if polar_acc_ref_q.qsize():
            polar_acc = []
            for _ in range(polar_acc_ref_q.qsize()):
                polar_acc.append(polar_acc_ref_q.get())
            polar_acc = np.asarray(polar_acc)
            print(f'raw polar acc data shape is : {polar_acc.shape}')
            reference_data['polar_acc'] = {}
            try:
                reference_data['polar_acc']['timestamp'] = polar_acc[:, 0]
                reference_data['polar_acc']['X'] = np.asarray(polar_acc[:, 1], dtype=float)
                reference_data['polar_acc']['Y'] = np.asarray(polar_acc[:, 2], dtype=float)
                reference_data['polar_acc']['Z'] = np.asarray(polar_acc[:, 3], dtype=float)
            except: pass

        if polar_rri_ref_q.qsize():
            polar_rri = []
            for _ in range(polar_rri_ref_q.qsize()):
                polar_rri.append(polar_rri_ref_q.get())
            polar_rri = np.asarray(polar_rri)
            print(f'raw polar RRI data shape is : {polar_rri.shape}')
            reference_data['polar_rri'] = {}
            try:
                reference_data['polar_rri']['timestamp'] = polar_rri[:, 0]
                reference_data['polar_rri']['RRI'] = np.asarray(polar_rri[:, 1], dtype=float)
            except: pass

    if USE_VERNIER_BELT:
        vernier_ref_data = []
        for record in range(vernier_belt_ref_q.qsize()):
            vernier_ref_data.append(vernier_belt_ref_q.get())
        vernier_ref_data.sort(key = lambda x: x[0])
        vernier_ref_data = np.asarray(vernier_ref_data)
        print(f'raw vernier_ref data shape is : {vernier_ref_data.shape}')
        try:
            reference_data['vernier'] = {}
            reference_data['vernier']['timestamp'] = vernier_ref_data[:,0]
            reference_data['vernier']['force'] = np.asarray(vernier_ref_data[:,1], dtype=float)
            reference_data['vernier']['respiration'] = np.asarray(vernier_ref_data[:,2], dtype=float)
        except Exception as e: print(e)

    if USE_WITMOTION1:
        witmotion_ref_data_1 = []
        for record in range(witmotion1_ref_data_q.qsize()):
            witmotion_ref_data_1.append(witmotion1_ref_data_q.get())
        # witmotion_ref_data_1.sort(key = lambda x: x[0])
        witmotion_ref_data_1 = np.asarray(witmotion_ref_data_1)
        print(f'raw Witmotion 1 data shape is : {witmotion_ref_data_1.shape}')
        try:
            reference_data[f'witmotion-{WITMOTION1_LOCATION}'] = {}
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['timeStamp'] = witmotion_ref_data_1[:, 0]
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['Acceleration_X'] = np.asarray(witmotion_ref_data_1[:, 1], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['Acceleration_Y'] = np.asarray(witmotion_ref_data_1[:, 2], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['Acceleration_Z'] = np.asarray(witmotion_ref_data_1[:, 3], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['AngularVelocity_X'] = np.asarray(witmotion_ref_data_1[:, 4], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['AngularVelocity_Y'] = np.asarray(witmotion_ref_data_1[:, 5], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['AngularVelocity_Z'] = np.asarray(witmotion_ref_data_1[:, 6], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['Angle_X'] = np.asarray(witmotion_ref_data_1[:, 7], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['Angle_Y'] = np.asarray(witmotion_ref_data_1[:, 8], dtype=float)
            reference_data[f'witmotion-{WITMOTION1_LOCATION}']['Angle_Z'] = np.asarray(witmotion_ref_data_1[:, 9], dtype=float)
        except: print('Issue with Witmotion-1')
    
    if USE_WITMOTION2:
        witmotion_ref_data_2 = []
        for record in range(witmotion2_ref_data_q.qsize()):
            witmotion_ref_data_2.append(witmotion2_ref_data_q.get())
        # witmotion_ref_data_1.sort(key = lambda x: x[0])
        witmotion_ref_data_2 = np.asarray(witmotion_ref_data_2)
        print(f'raw Witmotion 2 data shape is : {witmotion_ref_data_2.shape}')
        try:
            reference_data[f'witmotion-{WITMOTION2_LOCATION}'] = {}
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['timeStamp'] = witmotion_ref_data_2[:, 0]
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['Acceleration_X'] = np.asarray(witmotion_ref_data_2[:, 1], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['Acceleration_Y'] = np.asarray(witmotion_ref_data_2[:, 2], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['Acceleration_Z'] = np.asarray(witmotion_ref_data_2[:, 3], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['AngularVelocity_X'] = np.asarray(witmotion_ref_data_2[:, 4], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['AngularVelocity_Y'] = np.asarray(witmotion_ref_data_2[:, 5], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['AngularVelocity_Z'] = np.asarray(witmotion_ref_data_2[:, 6], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['Angle_X'] = np.asarray(witmotion_ref_data_2[:, 7], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['Angle_Y'] = np.asarray(witmotion_ref_data_2[:, 8], dtype=float)
            reference_data[f'witmotion-{WITMOTION2_LOCATION}']['Angle_Z'] = np.asarray(witmotion_ref_data_2[:, 9], dtype=float)
        except: print('Issue with Witmotion-2')
    
    # if USE_M5STICK_ACC1:
    #     acc1_ref_data = []
    #     for record in range(m5stick_acc1_ref_q.qsize()):
    #         acc1_ref_data.append(m5stick_acc1_ref_q.get())
    #     acc1_ref_data = np.asarray(acc1_ref_data)
    #     print(f'raw M5Stick-C ACC1 data shape is : {acc1_ref_data.shape}')
    #     try:
    #         reference_data['acc1'] = {}
    #         reference_data['acc1']['timestamp'] = acc1_ref_data[:, 0]
    #         reference_data['acc1']['accX'] = np.asarray(acc1_ref_data[:, 1], dtype=float)
    #         reference_data['acc1']['accY'] = np.asarray(acc1_ref_data[:, 2], dtype=float)
    #         reference_data['acc1']['accZ'] = np.asarray(acc1_ref_data[:, 3], dtype=float)
    #     except Exception as e: print(e)

    # if USE_M5STICK_ACC2:
    #     acc2_ref_data = []
    #     for record in range(m5stick_acc2_ref_q.qsize()):
    #         acc2_ref_data.append(m5stick_acc2_ref_q.get())
    #     acc2_ref_data = np.asarray(acc2_ref_data)
    #     print(f'raw M5Stick-C ACC2 data shape is : {acc2_ref_data.shape}')
    #     try:
    #         reference_data['acc2'] = {}
    #         reference_data['acc2']['timestamp'] = acc2_ref_data[:, 0]
    #         reference_data['acc2']['value'] = np.asarray(acc2_ref_data[:, 1], dtype=float)
    #     except Exception as e: print(e)

    if USE_HAT:
        hat_ref_data = []
        for record in range(hat_ref_q.qsize()):
            hat_ref_data.append(hat_ref_q.get())
        hat_ref_data = np.asarray(hat_ref_data)
        print(f'raw M5Stick-C Blink data shape is : {hat_ref_data.shape}')
        try:
            reference_data['hat'] = {}
            reference_data['hat']['timestamp'] = hat_ref_data[:, 0]
            reference_data['hat']['AccX'] = np.asarray(hat_ref_data[:, 1], dtype=float)
            reference_data['hat']['AccY'] = np.asarray(hat_ref_data[:, 2], dtype=float)
            reference_data['hat']['AccZ'] = np.asarray(hat_ref_data[:, 3], dtype=float)
            reference_data['hat']['EOG Raw Data'] = np.asarray(hat_ref_data[:, 4], dtype=float)
            reference_data['hat']['EOG Filtered Data'] = np.asarray(hat_ref_data[:, 5], dtype=float)
        except Exception as e: print(e)

    if USE_M5STICKC_MARKER:
        m5stickc_marker_ref_data = []
        for record in range(m5stickc_marker_ref_q.qsize()):
            m5stickc_marker_ref_data.append(m5stickc_marker_ref_q.get())

        m5stickc_marker_ref_data = np.asarray(m5stickc_marker_ref_data)
        print(f'raw M5Stick-C Marker data shape is : {m5stickc_marker_ref_data.shape}')
        try:
            reference_data['m5stick_marker'] = {}
            reference_data['m5stick_marker']['timestamp'] = m5stickc_marker_ref_data[:, 0]
            reference_data['m5stick_marker']['value'] = np.asarray(m5stickc_marker_ref_data[:, 1], dtype=int)
        except:
            pass

    try:
        beeps_data = []
        for record in range(BEEP_DATA.qsize()):
            beeps_data.append(BEEP_DATA.get())
        beeps_data = np.asarray(beeps_data)
        print(f'Markers data shape is : {beeps_data.shape}')
        reference_data['markers'] = {}
        reference_data['markers']['timestamp'] = beeps_data[:, 0]
        reference_data['markers']['experiment'] = beeps_data[:, 1]
    except:pass

def save_to_excel(data_dict):
    if not os.path.exists(LOCAL_CACHE_PATH):
            os.makedirs(LOCAL_CACHE_PATH)
    with pd.ExcelWriter(f"{LOCAL_CACHE_PATH}{OUT_FILE}.xlsx") as writer:
        for key in data_dict.keys(): ##
            sheet = pd.DataFrame(data_dict[key]) 
            if 'radar' in key:
                sheet.to_excel(writer, sheet_name=key, header = False, index = False)
            else:
                sheet.to_excel(writer, sheet_name=key, header = True, index = False)
        
    print('\n\nExcel file created ;)')