import random
import numpy as np

def returndata():
    data_dir1 = './teacher_data/'
    data_dir2 = './fornn/'
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    label = []
    xdata = []
    ydata = []
    odata = []
    for the_class_name in class_name:
        dir1 = data_dir1 + the_class_name + '_OH_fornn.npy'
        dir2 = data_dir2 + the_class_name + '_simu_fornn.npy'
        dataset = np.load(dir1, allow_pickle=True)
        x_data1 = dataset[0]
        y_data1 = dataset[1]
        o_data1 = dataset[2]
        dataset = np.load(dir2, allow_pickle=True)
        x_data2 = dataset[0]
        y_data2 = dataset[1]
        o_data2 = dataset[2]
        x_data = list(x_data1) + list(x_data2)
        y_data = list(y_data1) + list(y_data2)
        o_data = list(o_data1) + list(o_data2)
        for index in x_data:
            bridge_ele = '?'
            if ((round(float(index[5]), 1) == round(3.0, 1) and round(float(index[6]), 1) == round(3.0, 1))
                                                               or (round(float(index[5]), 1) == round(3.0, 1) and round(float(index[6]), 1) == round(3.0, 1))): # 1.8
                bridge_ele = 'Ru-Ru'
            if ((round(float(index[5]), 1) == round(3.0, 1) and round(float(index[6]), 1) == round(4.0, 1))
                                                               or (round(float(index[5]), 1) == round(4.0, 1) and round(float(index[6]), 1) == round(3.0, 1))): # 2.2
                bridge_ele = 'Ru-Rh'
            if ((round(float(index[5]), 1) == round(3.0, 1) and round(float(index[6]), 1) == round(6.5, 1))
                                                               or (round(float(index[5]), 1) == round(6.5, 1) and round(float(index[6]), 1) == round(3.0, 1))): # 2.6
                bridge_ele = 'Ru-Ir'
            if ((round(float(index[5]), 1) == round(3.0, 1) and round(float(index[6]), 1) == round(8.0, 1))
                                                               or (round(float(index[5]), 1) == round(8.0, 1) and round(float(index[6]), 1) == round(3.0, 1))): # 3.0
                bridge_ele = 'Ru-Pd'
            if ((round(float(index[5]), 1) == round(3.0, 1) and round(float(index[6]), 1) == round(10.5, 1))
                                                               or (round(float(index[5]), 1) == round(10.5, 1) and round(float(index[6]), 1) == round(3.0, 1))): # 3.4
                bridge_ele = 'Ru-Pt'
            if ((round(float(index[5]), 1) == round(4.0, 1) and round(float(index[6]), 1) == round(4.0, 1))
                                                               or (round(float(index[5]), 1) == round(4.0, 1) and round(float(index[6]), 1) == round(4.0, 1))): # 2.6
                bridge_ele = 'Rh-Rh'
            if ((round(float(index[5]), 1) == round(4.0, 1) and round(float(index[6]), 1) == round(6.5, 1))
                                                               or (round(float(index[5]), 1) == round(6.5, 1) and round(float(index[6]), 1) == round(4.0, 1))): # 3.0
                bridge_ele = 'Rh-Ir'
            if ((round(float(index[5]), 1) == round(4.0, 1) and round(float(index[6]), 1) == round(8.0, 1))
                                                               or (round(float(index[5]), 1) == round(8.0, 1) and round(float(index[6]), 1) == round(4.0, 1))): # 3.4
                bridge_ele = 'Rh-Pd'
            if ((round(float(index[5]), 1) == round(4.0, 1) and round(float(index[6]), 1) == round(10.5, 1))
                                                               or (round(float(index[5]), 1) == round(10.5, 1) and round(float(index[6]), 1) == round(4.0, 1))): # 3.8
                bridge_ele = 'Rh-Pt'
            if ((round(float(index[5]), 1) == round(6.5, 1) and round(float(index[6]), 1) == round(6.5, 1))
                                                               or (round(float(index[5]), 1) == round(6.5, 1) and round(float(index[6]), 1) == round(6.5, 1))): # 3.4
                bridge_ele = 'Ir-Ir'
            if ((round(float(index[5]), 1) == round(6.5, 1) and round(float(index[6]), 1) == round(8.0, 1))
                                                               or (round(float(index[5]), 1) == round(8.0, 1) and round(float(index[6]), 1) == round(6.5, 1))): # 3.8
                bridge_ele = 'Ir-Pd'
            if ((round(float(index[5]), 1) == round(6.5, 1) and round(float(index[6]), 1) == round(10.5, 1))
                                                               or (round(float(index[5]), 1) == round(10.5, 1) and round(float(index[6]), 1) == round(6.5, 1))): # 4.2
                bridge_ele = 'Ir-Pt'
            if ((round(float(index[5]), 1) == round(8.0, 1) and round(float(index[6]), 1) == round(8.0, 1))
                                                               or (round(float(index[5]), 1) == round(8.0, 1) and round(float(index[6]), 1) == round(8.0, 1))): # 4.2
                bridge_ele = 'Pd-Pd'
            if ((round(float(index[5]), 1) == round(8.0, 1) and round(float(index[6]), 1) == round(10.5, 1))
                                                               or (round(float(index[5]), 1) == round(10.5, 1) and round(float(index[6]), 1) == round(8.0, 1))): # 4.6
                bridge_ele = 'Pd-Pt'
            if ((round(float(index[5]), 1) == round(10.5, 1) and round(float(index[6]), 1) == round(10.5, 1))
                                                               or (round(float(index[5]), 1) == round(10.5, 1) and round(float(index[6]), 1) == round(10.5, 1))): # 5.0
                bridge_ele = 'Pt-Pt'
            if bridge_ele == '?':
                print('>_<')
            label.append([bridge_ele, the_class_name])
        xdata += x_data
        ydata += y_data
        odata += o_data
    return xdata, ydata, label, odata

returndata()

