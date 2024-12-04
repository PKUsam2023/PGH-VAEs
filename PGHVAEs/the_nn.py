from getdata import returndata
from sklearn import metrics
from torch.nn import init
import torch
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import argparse
import sys

def test_model(predict, y_test):
    predict = predict.detach().numpy()
    y_test = y_test.detach().numpy()
    MAE = metrics.mean_absolute_error(y_test, predict)
    MSE = metrics.mean_squared_error(y_test, predict)
    RMSE = MSE ** 0.5

    return MAE, RMSE

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
            #nn.Dropout(p=0.1),
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

        KLD = 10 * KLD1 + KLD2
        simu_x = torch.cat((recon_x1, recon_x2), dim=1)

        return predict, KLD, simu_x


def run_nn(the_dir, batch_size, learning_rate, num_epochs, weight1, weight2):
    batch_size = batch_size
    learning_rate = learning_rate
    num_epochs = num_epochs
    weight1 = weight1
    weight2 = weight2
    the_dir = the_dir
    if not os.path.isdir(the_dir):
        os.mkdir(the_dir)
    file = open(the_dir + 'log.t', 'w')

    use_gpu = torch.cuda.is_available()
    x_train, y_train, x_valid, y_valid = returndata('all')
    y_train = np.array([[y] for y in y_train])
    y_valid = np.array([[y] for y in y_valid])
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_valid = torch.Tensor(x_valid)
    y_valid = torch.Tensor(y_valid)

    train_set = TensorDataset(x_train, y_train)
    valid_set = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_set,
                              batch_size = batch_size,
                              shuffle=True)

    model = Network(57)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()
    criterion_conggou = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    draw_train_MAE = []
    draw_train_RMSE = []
    draw_train_MSE = []
    draw_test_MAE = []
    draw_test_RMSE = []
    draw_test_MSE = []
    draw_train_KLD = []
    draw_test_KLD = []
    draw_train_loss2 = []
    draw_test_loss2 = []

    marked_test_MAE = 1
    marked_num = 0

    t = range(num_epochs)
    for epoch in range(num_epochs):
        file.write(f'epoch {epoch + 1}' + '\n')
        running_loss = 0.0
        running_LST = 0.0
        running_loss2 = 0.0
        model.train()
        train_MAE = []
        train_RMSE = []
        train_KLD = []
        train_loss2 = []
        n_train = 0
        for x, y in train_loader:
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            x = x.to(device)
            y = y.to(device)
            model = model.to(device)
            out, LST, x_simu = model(x)
            loss_2 = criterion_conggou(x_simu[:, :4], x[:, :4])
            loss_3 = criterion_conggou(x_simu[:, 4:], x[:, 4:])
            loss = criterion(out, y) + weight1 * LST + 10 * weight2 * loss_2 + weight2 * loss_3
            running_loss += loss.item()
            running_LST += LST.item()
            running_loss2 += loss_2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            MAE, RMSE = test_model(out, y)
            train_MAE.append(MAE)
            train_RMSE.append(RMSE)
            train_KLD.append(LST)
            n_train += 1
        loss1 = running_loss/n_train
        fin_train_MAE = np.sum(train_MAE) / n_train
        fin_train_RMSE = np.sum(train_RMSE)/ n_train
        fin_train_KLD = running_LST / n_train
        fin_train_loss2 = running_loss2 / n_train
        draw_train_MAE.append(fin_train_MAE)
        draw_train_RMSE.append(fin_train_RMSE)
        draw_train_MSE.append(loss1)
        draw_train_KLD.append(fin_train_KLD)
        draw_train_loss2.append(fin_train_loss2)
        file.write("train_MAE: {} train_RMSE: {} loss: {} train_KLD: {} train_loss2: {}".format(fin_train_MAE, fin_train_RMSE, loss1, fin_train_KLD, fin_train_loss2) + '\n')
        print(
            "train_MAE: {} train_RMSE: {} loss: {} train_KLD: {} train_loss2: {}".format(fin_train_MAE, fin_train_RMSE,
                                                                                         loss1, fin_train_KLD,
                                                                                         fin_train_loss2) + '\n')
        test_MAE = []
        test_RMSE = []
        test_KLD = []
        test_loss2 = []
        n_test = 0

        model.eval()
        eval_loss = 0.0
        eval_LST = 0.0
        eval_loss2 = 0.0
        for x, y in valid_loader:
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            x = x.to(device)
            y = y.to(device)
            out, LST, x_simu = model(x)
            loss_2 = criterion_conggou(x_simu, x)
            loss = criterion(out, y) + weight1 * LST + weight2 * loss_2
            eval_loss += loss.item()
            eval_LST += LST.item()
            eval_loss2 += loss_2.item()
            MAE, RMSE = test_model(out, y)
            test_MAE.append(MAE)
            test_RMSE.append(RMSE)
            test_KLD.append(LST)
            n_test += 1
        loss1 = eval_loss / n_test
        fin_test_MAE = np.sum(test_MAE) / n_test
        fin_test_RMSE = np.sum(test_RMSE) / n_test
        fin_test_KLD = eval_LST / n_test
        fin_test_loss2 = eval_loss2 / n_test

        # save model
        if marked_test_MAE > fin_test_MAE:
            marked_test_MAE = fin_test_MAE
            marked_num = epoch
            torch.save(model, "./" + the_dir + str(marked_num + 1) + "_epoch.pth")

        draw_test_MAE.append(fin_test_MAE)
        draw_test_RMSE.append(fin_test_RMSE)
        draw_test_MSE.append(loss1)
        draw_test_KLD.append(fin_test_KLD)
        draw_test_loss2.append(fin_test_loss2)
        file.write("test_MAE: {} test_RMSE: {}  loss: {} test_KLD: {} test_loss2: {}".format(fin_test_MAE, fin_test_RMSE, loss1, fin_test_KLD, fin_test_loss2) + '\n')
        print("test_MAE: {} test_RMSE: {}  loss: {} test_KLD: {} test_loss2: {}".format(fin_test_MAE, fin_test_RMSE,
                                                                                        loss1, fin_test_KLD,
                                                                                        fin_test_loss2) + '\n')

    file.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

    ax1.plot(draw_train_MAE, label='train loss')
    ax1.legend()
    ax1.set_title("train_MAE")

    ax2.plot(draw_test_MAE, label='valid loss')
    ax2.legend()
    ax2.set_title("test_MAE")

    fig.savefig("./" + the_dir + "MAE.png", bbox_inches='tight')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

    ax1.plot(draw_train_RMSE, label='train loss')
    ax1.legend()
    ax1.set_title("Loss Curve")

    ax2.plot(draw_test_RMSE, label='valid loss')
    ax2.legend()
    ax2.set_title("Loss Curve")

    fig.savefig("./" + the_dir + "RMSE.png", bbox_inches='tight')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

    ax1.plot(draw_train_MSE, label='train loss')
    ax1.legend()
    ax1.set_title("Loss Curve")

    ax2.plot(draw_test_MSE, label='valid loss')
    ax2.legend()
    ax2.set_title("Loss Curve")

    fig.savefig("./" + the_dir + "MSE.png", bbox_inches='tight')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

    ax1.plot(draw_train_KLD, label='train loss')
    ax1.legend()
    ax1.set_title("Loss Curve")

    ax2.plot(draw_test_KLD, label='valid loss')
    ax2.legend()
    ax2.set_title("Loss Curve")

    fig.savefig("./" + the_dir + "KLD.png", bbox_inches='tight')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

    ax1.plot(draw_train_loss2, label='train loss')
    ax1.legend()
    ax1.set_title("Loss Curve")

    ax2.plot(draw_test_loss2, label='valid loss')
    ax2.legend()
    ax2.set_title("Loss Curve")

    fig.savefig("./" + the_dir + "loss2.png", bbox_inches='tight')

def vae_main(args):
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    weight1 = args.weight1
    weight2 = args.weight2

    the_dir = './result/' + str(batch_size) + '_' + str(learning_rate) + '_' + str(num_epochs) + '_' + str(weight1) + '_' + str(weight2) + './'
    run_nn(the_dir, batch_size, learning_rate, num_epochs, weight1, weight2)

def parse_args(args):
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.0007, type=float)
    parser.add_argument('--num_epochs', default= 1000, type=int)
    parser.add_argument('--weight1', default=1e-05, type=float)
    parser.add_argument('--weight2', default=1e-06, type=float)
    args = parser.parse_args()
    return args

def do_main():
    args = parse_args(sys.argv[1:])
    vae_main(args)

if __name__ == "__main__":
    do_main()