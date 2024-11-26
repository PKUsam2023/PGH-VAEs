from getdata import returndata
from torch.nn import init
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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
        return mu, logvar, h

    def decode(self, z, decoder):
        return decoder(z)

    def forward(self, x):
        x_str = x[:, :4]
        x_ele = x[:, 4:]
        mu1, logvar1, h1 = self.encode(x_str, self.encoder1)
        z1, KLD1 = self.reparameterize(mu1, logvar1)
        recon_x1 = self.decode(z1, self.decoder1)

        mu2, logvar2, h2 = self.encode(x_ele, self.encoder2)
        z2, KLD2 = self.reparameterize(mu2, logvar2)
        recon_x2 = self.decode(z2, self.decoder2)

        # 不相关的全连接层输出
        ele_sampled_z = torch.cat((z1, z2), dim=1)
        predict = self.fc(ele_sampled_z)

        KLD = KLD1 + KLD2
        simu_x = torch.cat((recon_x1, recon_x2), dim=1)

        return predict, z1, z2, h1, h2, x_str, x_ele

def run_nn():
    batch_size = 16
    # 判断GPU是否可用
    use_gpu = torch.cuda.is_available()
    xdata, ydata, label, odata = returndata()
    ydata = np.array([[y] for y in ydata])
    xdata = torch.Tensor(xdata)
    ydata = torch.Tensor(ydata)
    print(len(ydata))
    train_set = TensorDataset(xdata, ydata)

    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              shuffle=False)

    model = torch.load('./793_epoch.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    predict_list = []
    z_str_list = []
    z_ele_list = []
    h1_list = []
    h2_list = []
    x_str_list = []
    x_ele_list = []
    for x, y in train_loader:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)
        predict, z1, z2, h1, h2, x_str, x_ele = model(x)
        for item in predict.detach().numpy():
            predict_list.append(list(item))

        for item in z1.detach().numpy():
            z_str_list.append(list(item))

        for item in z2.detach().numpy():
            z_ele_list.append(list(item))

        for item in h1.detach().numpy():
            h1_list.append(list(item))

        for item in h2.detach().numpy():
            h2_list.append(list(item))

        for item in x_str.detach().numpy():
            x_str_list.append(list(item))

        for item in x_ele.detach().numpy():
            x_ele_list.append(list(item))

    all_data = []
    all_data.append(predict_list)
    all_data.append(z_str_list)
    all_data.append(z_ele_list)
    all_data.append(label)
    all_data.append(h1_list)
    all_data.append(h2_list)
    all_data.append(x_str_list)
    all_data.append(x_ele_list)
    all_data.append(odata)
    np.save('./result/step_one.npy', all_data)

if __name__ == "__main__":
    run_nn()