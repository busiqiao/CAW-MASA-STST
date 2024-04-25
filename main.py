import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from torch import nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchinfo import summary
from tqdm import tqdm

from dataset import EEGDataset
from model import CAW_MASA_STST
from utils.util import train, test

channelNum = 20
num_class = 72
chan_spe = 25
tlen = 32
epochs = 70
base = 'acc'
data = 'EEG72'

batch_size = 64
k = 10
Fs = 62.5
seed_value = 3407

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # hash seed
torch.manual_seed(seed_value)  # CPU seed
torch.cuda.manual_seed(seed_value)  # GPU seed
history = np.zeros((10, 10))
kf = KFold(n_splits=k, shuffle=True, random_state=seed_value)

if __name__ == '__main__':
    dataPath1 = f'/data/{data}'
    dataPath2 = f'/data/{data}-CWT20'
    output_path = f'./outputs/{num_class}/test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(
        '\rParameters: dataset={}, num_class={}，epochs={}，batch_size={}，k_fold={}，manual_seed={}'
        .format(data, num_class, epochs, batch_size, k, seed_value))

    for i in range(k):
        dataset = EEGDataset(file_path1=dataPath1 + f'/S{i + 1}.mat', file_path2=dataPath2 + f'/sub{i + 1}_cwt.npy',
                             num_class=num_class)

        for fold, (train_i, test_i) in enumerate(kf.split(dataset)):
            train_i, val_i = train_test_split(train_i, test_size=1/9, random_state=42)

            train_sampler = SubsetRandomSampler(train_i)
            val_sampler = SubsetRandomSampler(val_i)
            test_sampler = SubsetRandomSampler(test_i)
            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=3, prefetch_factor=2,
                                      drop_last=True)
            val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size, num_workers=1, prefetch_factor=1,
                                    drop_last=True)
            test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size, num_workers=1,prefetch_factor=1,
                                     drop_last=True)

            n_train = len(train_loader) * batch_size
            n_val = len(val_loader) * batch_size
            n_test = len(test_loader) * batch_size

            # create model
            model = CAW_MASA_STST.CAW_MASA_STST(classNum=num_class, channelNum=channelNum, chan_spe=chan_spe,
                                                tlen=tlen).cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # print model summary
            if i == 0 and fold == 0:
                summary(model, input_size=[(10, 124, 32), (10, 20, 25, 32)])

            if fold == 0:
                print('\rSub{}:  train_num={}, test_num={}'.format(int(i + 1), n_train, n_test))

            best_val_loss = float('inf')
            best_val_acc = 0
            best_model = None
            for epoch in range(epochs):
                # train
                train_loop = tqdm(train_loader, total=len(train_loader))
                for (x, x_spe, y) in train_loop:
                    loss, acc = train(model=model, optimizer=optimizer, criterion=criterion, x=x, x_spe=x_spe, y=y)

                    current_lr = optimizer.param_groups[0]['lr']
                    train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}] - Train')
                    train_loop.set_postfix(loss=loss.item(), acc=acc, lr=current_lr)

                # validation
                val_loop = tqdm(val_loader, total=len(val_loader))
                val_losses = []
                val_accuracy = []
                for (x_val, x_spe_val, y_val) in val_loop:
                    val_loss, val_acc = test(model=model, criterion=criterion, x=x_val, x_spe=x_spe_val, y=y_val)
                    val_losses.append(val_loss.item())
                    val_accuracy.append(val_acc)

                    val_loop.set_description(f'               Validation')
                    val_loop.set_postfix(val_loss=val_loss.item(), val_acc=val_acc)

                # save best_model
                if base == 'loss':
                    # loss
                    avg_val_loss = np.mean(val_losses)
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model, f'{output_path}/best_model_{i}_{fold}.pth')
                elif base == 'acc':
                    # acc
                    avg_val_acc = np.mean(val_accuracy)
                    if avg_val_acc > best_val_acc:
                        best_val_acc = avg_val_acc
                        torch.save(model, f'{output_path}/best_model_{i}_{fold}.pth')
                else:
                    raise ValueError('base must be "loss" or "acc"')

            # test
            losses = []
            accuracy = []
            torch.load(f'{output_path}/best_model_{i}_{fold}.pth')  # load best_model
            test_loop = tqdm(test_loader, total=len(test_loader))
            for (xx, xx_spe, yy) in test_loop:
                test_loss, test_acc = test(model=model, criterion=criterion, x=xx, x_spe=xx_spe, y=yy)
                losses.append(test_loss)
                accuracy.append(test_acc)

                test_loop.set_description(f'                 Test ')
                test_loop.set_postfix(loss=test_loss.item(), acc=test_acc)

            avg_test_acc = np.mean(accuracy)
            history[i][fold] = avg_test_acc
            print('\rSub{} kfold-{} test acc: {}'.format(i + 1, fold + 1, history[i][fold]))
            print('\r---------------------------------------------------------')

        print(history[i])
        print('\rSub{} train down，average acc: {}'.format(i + 1, np.mean(history[i], axis=0)))
        print('\r*************************************************************')

    print(history)
    print('\rTrain Down，{}class average acc: {}'.format(num_class, np.mean(history)))
