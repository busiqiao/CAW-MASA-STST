import numpy as np
import scipy.io
import torch
from torch.utils.data.dataset import Dataset


def channelSelect(X, channelNum):
    mstd = X.mean(0).std(1)
    stdIdx = np.argsort(mstd)
    selIdx = stdIdx[-channelNum:]
    sIdx = [False] * X.shape[1]
    for i in range(X.shape[1]):
        if i in selIdx:
            sIdx[i] = True
    X = X[:, sIdx, :]
    return X


class EEGDataset(Dataset):

    def __init__(self, file_path1, file_path2, num_class=6):
        mat = scipy.io.loadmat(file_path1)
        data1 = np.asarray(mat['X_3D'])
        data1 = np.transpose(data1, (2, 0, 1))
        self.data1 = torch.from_numpy(data1[:, :, :]).float()

        date2 = np.load(file_path2)
        self.data2 = torch.from_numpy(date2[:, :, :, :]).float()

        if num_class == 6:
            self.label = torch.from_numpy(np.asarray(mat['categoryLabels']).squeeze() - 1).long()
        elif num_class == 72:
            self.label = torch.from_numpy(np.asarray(mat['exemplarLabels']).squeeze() - 1).long()

    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.label[index]
