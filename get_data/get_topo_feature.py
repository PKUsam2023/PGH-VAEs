import numpy as np
import os
import Digraph
import math

def get_estimation(lst):
    the_lst = []
    for index in range(1, 41):
        if lst[index*2] == lst[index*2 - 1]:
            the_lst.append(lst[index*2])
        else:
            the_lst.append(lst[(index-1)*2])
    return the_lst

def dis(coor1, coor2):
    the_dis = math.sqrt(((coor1[0] - coor2[0]) ** 2) + ((coor1[1] - coor2[1]) ** 2) + ((coor1[2] - coor2[2]) ** 2))
    return the_dis

def get_betti(betti_stat):
    the_rad = 4.1
    betti = []
    for step in range(0, int(the_rad * 20)):
        num = 0
        for item in betti_stat:
            if item[0] <= step / 20:
                if item[1] >= step / 20 or item[1] == -1:
                    num += 1
        betti.append(num)
    return betti

def get_betti_feature(path_homology):
    betti0_stat = path_homology['0']
    betti1_stat = path_homology['1']
    betti2_stat = path_homology['2']

    betti0 = get_betti(betti0_stat)
    betti1 = get_betti(betti1_stat)
    betti2 = get_betti(betti2_stat)
    betti0 = get_estimation(betti0)
    betti1 = get_estimation(betti1)
    betti2 = get_estimation(betti2)

    return betti0, betti1, betti2

dir1  = './rawfeature/'
dir2 = './rawfeature_simu/'

data_dir = os.listdir(dir2)
for dir in data_dir:
    name = dir.split('.')[0]
    all_feature = []
    str_feature = []
    fin_oh_energy = []
    npy_dir = dir2 + dir
    loadData = np.load(npy_dir, allow_pickle=True)

    for item in loadData:
        nuber_list = [i+1 for i in range(len(item[1]))]
        path = item[2]
        path_dis = []
        for perpath in path:
            path_dis.append([perpath[0], perpath[1], dis(item[1][perpath[0] - 1], item[1][perpath[1] - 1])])
        V = nuber_list
        weighted_digraph = path_dis
        dim = 3
        dg = Digraph.Digraph(V, weighted_digraph, dim)
        dg.get_persistence()
        path_homology = dg.diagram
        betti0, betti1, betti2 = get_betti_feature(path_homology)

        bett_feature = [max(betti0), np.mean(betti0), np.mean(betti1), np.mean(betti2)]
        bett_feature2 = betti0 + betti1 + betti2
        print(bett_feature)
        ligand = item[7]
        the_feature_zero = [0 for _ in range(53)]
        for i in range(len(ligand)):
            the_feature_zero[i] = ligand[i]
        feature = bett_feature + [ligand[0] + ligand[1]] + the_feature_zero
        #fin_oh_energy.append(item[8])
        all_feature.append(feature)
        str_feature.append(bett_feature2)

    train_data =[]
    train_data.append(all_feature)
    train_data.append(fin_oh_energy)
    train_data.append(str_feature)
    np.save('./student_data/' + dir.split('.')[0] + '_fornn.npy', train_data)
    #np.save('./teacher_data/' + dir.split('.')[0] + '_fornn.npy', train_data)

