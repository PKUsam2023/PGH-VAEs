import random
import numpy as np

def gat_data():
    data_dir = './teacher_data/'
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    xtrain = []
    ytrain = []
    xvalid = []
    yvalid = []
    the_num = 0
    for the_class_name in class_name:
        dir = data_dir + the_class_name + '_OH_fornn.npy'
        #dir = data_dir + the_class_name + '_simu_fornn.npy'
        dataset = np.load(dir, allow_pickle=True)
        x_data = dataset[0]
        y_data = dataset[1]
        the_data_radom = list(zip(x_data, y_data))
        random.shuffle(the_data_radom)
        x = [index[0] for index in the_data_radom]
        y = [index[1] for index in the_data_radom]
        trainset_rate = 0.7
        validset_rate = 0.3
        x_train = x[:int(trainset_rate * len(x))]
        y_train = y[:int(trainset_rate * len(y))]
        x_valid = x[int(trainset_rate * len(x)): int((trainset_rate + validset_rate) * len(x))]
        y_valid = y[int(trainset_rate * len(y)): int((trainset_rate + validset_rate) * len(y))]

        xtrain.append(x_train)
        ytrain.append(y_train)
        xvalid.append(x_valid)
        yvalid.append(y_valid)
        num = len(x_train)
        the_num += num
    print(the_num)
    return xtrain, ytrain, xvalid, yvalid

def gat_data_simu():
    data_dir = './fornn/'
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    xtrain = []
    ytrain = []
    xvalid = []
    yvalid = []
    the_num = 0
    for the_class_name in class_name:
        dir = data_dir + the_class_name + '_simu_fornn.npy'
        dataset = np.load(dir, allow_pickle=True)
        x_data = dataset[0]
        y_data = dataset[1]
        the_data_radom = list(zip(x_data, y_data))
        random.shuffle(the_data_radom)
        x = [index[0] for index in the_data_radom]
        y = [index[1] for index in the_data_radom]
        trainset_rate = 1.0
        validset_rate = 0.0
        x_train = x[:int(trainset_rate * len(x))]
        y_train = y[:int(trainset_rate * len(y))]
        x_valid = x[int(trainset_rate * len(x)): int((trainset_rate + validset_rate) * len(x))]
        y_valid = y[int(trainset_rate * len(y)): int((trainset_rate + validset_rate) * len(y))]

        xtrain.append(x_train)
        ytrain.append(y_train)
        xvalid.append(x_valid)
        yvalid.append(y_valid)
        num = len(x_train)
        the_num += num
    print(the_num)
    return xtrain, ytrain, xvalid, yvalid

def returndata(vaild):
    xtrain, ytrain, xvalid, yvalid = gat_data()
    xtrain_simu, ytrain_simu, xvalid_simu, yvalid_simu = gat_data_simu()
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill',
                  'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1',
                  'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge',
                  'ML_HEA_532_valley2']
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    if vaild == 'all':
        for item in range(len(xtrain)):
            x_train += xtrain[item]
            y_train += ytrain[item]
            x_valid += xvalid[item]
            y_valid += yvalid[item]

        for item in range(len(xtrain_simu)):
            x_train += xtrain_simu[item]
            y_train += ytrain_simu[item]
            x_valid += xvalid_simu[item]
            y_valid += yvalid_simu[item]

    else:
        num_class = class_name.index(vaild)
        for item in range(len(xtrain)):
            x_train += xtrain[item]
            y_train += ytrain[item]

        x_valid = xvalid[num_class]
        y_valid = yvalid[num_class]

    return x_train, y_train, x_valid, y_valid