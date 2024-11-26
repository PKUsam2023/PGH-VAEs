import numpy as np
import torch
import torch.nn as nn
import random

class Network(nn.Module):
    def __init__(self, feat_num):
        super(Network, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(54, 35),
            nn.ReLU(),
            nn.Linear(35, 20),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 4)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(10, 35),
            nn.ReLU(),
            nn.Linear(35, 54)
        )

        self.fc = nn.Sequential(
            nn.Linear(20, 1)
        )

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
        return mu + sigma * eps, KLD

    def encode(self, x, encoder):
        h = encoder(x)
        mu, logvar = h[:, :10], h[:, 10:]
        return mu, logvar

    def decode(self, z, decoder):
        return decoder(z)

    def forward(self, x):
        x_str = x[:, :4]
        x_ele = x[:, 4:]
        mu1, logvar1 = self.encode(x_str, self.encoder1)
        z1, KLD1 = self.reparameterize(mu1, logvar1)
        recon_x1 = self.decode(z1, self.decoder1)

        mu2, logvar2 = self.encode(x_ele, self.encoder2)
        z2, KLD2 = self.reparameterize(mu2, logvar2)
        recon_x2 = self.decode(z2, self.decoder2)

        # 不相关的全连接层输出
        ele_sampled_z = torch.cat((z1, z2), dim=1)
        predict = self.fc(ele_sampled_z)

        KLD = KLD1 + KLD2
        simu_x = torch.cat((recon_x1, recon_x2), dim=1)

        return predict, recon_x1, recon_x2


def get_100_result():
    dataset = np.load('./result/step_one.npy', allow_pickle=True)
    predict_list = dataset[0]
    z_str_list = dataset[1]
    z_ele_list = dataset[2]
    str_list = dataset[6]
    ele_list = dataset[7]
    label = dataset[3]
    predict_list = [np.array(i) for i in predict_list]
    z_str_list = [np.array(i) for i in z_str_list]
    z_ele_list = [np.array(i) for i in z_ele_list]
    label = [np.array(i) for i in label]

    str_type = []
    ele_type = []
    str_feature = []
    ele_feature = []
    for index in range(len(predict_list)):
        if predict_list[index] > 0:
            str_type.append(z_str_list[index])
            ele_type.append(z_ele_list[index])
            str_feature.append(str_list[index])
            ele_feature.append(ele_list[index])

    return str_type, ele_type, str_feature, ele_feature

def get_stu(recon_x1, recon_x2):
    data = [["./ML_HEA_100/", 75, 'bridge_OH', 48, 32.0],  # 46
            ["./ML_HEA_111/", 75, 'bridge_OH', 49, 32.725],  # 47
            ["./ML_HEA_110/", 95, 'bridge_OH', 38, 25.225],  # 38
            ["./ML_HEA_211/", 107, 'edge_OH', 41, 27.45],  # 39
            ["./ML_HEA_211/", 107, 'hill_OH', 51, 33.8],  # 49
            ["./ML_HEA_211/", 107, 'valley_OH', 53, 35.275],  # 51
            ["./ML_HEA_211/", 107, 'summit_OH', 46, 30.625],  # 44
            ["./ML_HEA_532/", 104, 'higher_edge_OH', 43, 28.55],  # 41
            ["./ML_HEA_532/", 104, 'inner_higher_edge_OH', 44, 29.075],  # 42
            ["./ML_HEA_532/", 104, 'hill1_OH', 51, 33.925],  # 49
            ["./ML_HEA_532/", 104, 'valley2_OH', 48, 32.15],  # 46
            ["./ML_HEA_532/", 104, 'hill2_OH', 48, 31.85],  # 46
            ["./ML_HEA_532/", 104, 'outer_higher_edge_OH', 41, 27.1]]  # 39
    a = recon_x1[1]
    dis_list = []
    for i in range(len(data)):
        dis_list.append(abs(data[i][4]-a))
    dis_list = np.array(dis_list)
    closest_row_index = np.argmin(dis_list)
    str = data[closest_row_index][0] + data[closest_row_index][2]
    #print(recon_x1)

    ele_data = [[3.0, 'Ru'],
                [4.0, 'Rh'],
                [6.5, 'Ir'],
                [8.0, 'Pd'],
                [10.5, 'Pt']]
    ele_list = []
    for ele in range(len(recon_x2) - 1):
        dis_list = []
        for index in range(len(ele_data)):
            dis_list.append(abs(ele_data[index][0]-recon_x2[ele + 1]))
        closest_row_index = np.argmin(dis_list)
        ele_list.append(ele_data[closest_row_index][1])

    return str, ele_list

def get_structure(feature_list1, feature_list2):
    model = torch.load('./793_epoch.pth')
    model.eval()
    feature_list1 = torch.Tensor(feature_list1)
    feature_list2 = torch.Tensor(feature_list2)
    recon_x1 = model.decode(feature_list1, decoder=model.decoder1)
    recon_x2 = model.decode(feature_list2, decoder=model.decoder2)
    for_energy = torch.cat((feature_list1, feature_list2),dim = 0)
    energy = model.fc(for_energy)
    recon_x1 = recon_x1.detach().numpy()
    recon_x2 = recon_x2.detach().numpy()
    energy = energy.detach().numpy()[0]

    return recon_x1, recon_x2, energy

if __name__ == "__main__":
    feature_list1, feature_list2, str_feature, ele_feature = get_100_result()
    all_num = 0
    all_data = []
    for index in range(len(feature_list1)):
        recon_x1, recon_x2, energy = get_structure(feature_list1[index], feature_list2[index])
        str, ele_list = get_stu(recon_x1, recon_x2)

        for i in range(20):
            feature_list2_new = [x * 0.8 for x in feature_list2[index]]
            recon_x1_new, recon_x2_new, energy_new = get_structure(feature_list1[index], feature_list2_new)
            if energy_new > 0:
                all_num += 1
                str, ele_list = get_stu(recon_x1_new, recon_x2_new) # 2, 9+10, other

                all_data += [str, ele_list]

    np.save('./result/other_str.npy', all_data)