import os
import random
import numpy as np
from torch import nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchinfo import summary
from tqdm import tqdm
from model import CAW_MASA_STST
from dataset import EEGDataset
from sklearn.model_selection import KFold
import torch
from utils.util import train, test

channelNum = 124
num_class = 6
chan_spe = 50
tlen = 32
epochs = 70
data = 'EEG72'

batch_size = 64
k = 10
Fs = 62.5
seed_value = 3407

k_fold = KFold(n_splits=k, shuffle=False)
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子
history = np.zeros((10, 10))

if __name__ == '__main__':  # 10个人分别进行10折交叉验证
    dataPath1 = f'H:\\EEG\\EEGDATA\\{data}'
    dataPath2 = f'H:\\EEG\\EEGDATA\\{data}-CWT'
    print(
        '\r参数设置: dataset={}, num_class={}，epochs={}，batch_size={}，k_fold={}，manual_seed={}'
        .format(data, num_class, epochs, batch_size, k, seed_value))

    for i in range(0, k):
        dataset = EEGDataset(file_path1=dataPath1 + f'\\S{i + 1}.mat', file_path2=dataPath2 + f'\\S{i + 1}_CWT.npy',
                             num_class=num_class)

        for fold, (train_i, test_i) in enumerate(k_fold.split(dataset)):
            train_sampler = SubsetRandomSampler(train_i)
            test_sampler = SubsetRandomSampler(test_i)
            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=0,
                                      drop_last=True)
            test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size, num_workers=0,
                                     drop_last=True)

            n_train = len(train_loader) * batch_size
            n_test = len(test_loader) * batch_size

            # 创建模型
            model = CAW_MASA_STST.CAW_MASA_STST(classNum=num_class, channelNum=channelNum, chan_spe=chan_spe,
                                                tlen=tlen).cuda()

            # 设置网络参数
            criterion = nn.CrossEntropyLoss()  # 交叉熵损失
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            if i == 0 and fold == 0:
                summary(model, input_size=[(10, 124, 32), (10, 124, 50, 32)])

            if fold == 0:
                print('\r第{}位受试者:  train_num={}, test_num={}'.format(int(i + 1), n_train, n_test))

            avg_test_acc = 0
            for epoch in range(epochs):
                losses = []
                accuracy = []

                train_loop = tqdm(train_loader, total=len(train_loader))
                for (x, x_spe, y) in train_loop:
                    x = x.cuda()
                    x_spe = x_spe.cuda()
                    y = y.cuda()
                    loss, y_ = train(model=model, optimizer=optimizer, criterion=criterion, x=x, x_spe=x_spe, y=y)
                    corrects = (torch.argmax(y_, dim=1).data == y.data)
                    acc = corrects.cpu().int().sum().numpy() / batch_size

                    # 获取当前学习率
                    current_lr = optimizer.param_groups[0]['lr']

                    train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}] - Train')
                    train_loop.set_postfix(loss=loss.item(), acc=acc, lr=current_lr)

                test_loop = tqdm(test_loader, total=len(test_loader))
                for (xx, xx_spe, yy) in test_loop:
                    val_loss, val_acc = test(model=model, criterion=criterion, x=xx, x_spe=xx_spe, y=yy)
                    val_acc = val_acc / batch_size
                    losses.append(val_loss)
                    accuracy.append(val_acc)

                    # 获取当前学习率
                    current_lr = optimizer.param_groups[0]['lr']

                    test_loop.set_description(f'               Test ')
                    test_loop.set_postfix(loss=val_loss.item(), acc=val_acc, lr=current_lr)

                avg_test_acc = np.sum(accuracy) / len(accuracy)
                # print(accuracy)
                print('\repoch 平均准确率：{:.6f}'.format(avg_test_acc))

            history[i][fold] = avg_test_acc
            print('\r受试者{}，第{}折测试准确率：{}'.format(i + 1, fold + 1, history[i][fold]))
            print('\r---------------------------------------------------------')

        print(history[i])
        print('\r受试者{}训练完成，平均准确率：{}'.format(i + 1, np.mean(history[i], axis=0)))
        print('\r*************************************************************')

    print(history)
    print('\r训练完成，{}类平均准确率：{}'.format(num_class, np.mean(history)))
