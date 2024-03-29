import numpy as np
import scipy.io
import torch
from torch.utils.data.dataset import Dataset


class EEGDataset(Dataset):

    def __init__(self, file_path1, file_path2, num_class=6):
        mat = scipy.io.loadmat(file_path1)
        data1 = np.asarray(mat['X_3D'])
        data1 = np.transpose(data1, (2, 0, 1))
        self.data1 = data1[:, :, :]

        date2 = np.load(file_path2)
        self.data2 = date2[:, :, :, :]

        if num_class == 6:
            self.label = np.asarray(mat['categoryLabels']).squeeze() - 1
        elif num_class == 72:
            self.label = np.asarray(mat['exemplarLabels']).squeeze() - 1

    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        feature1 = torch.tensor(self.data1[index, :, :], dtype=torch.float)
        feature2 = torch.tensor(self.data2[index, :, :, :], dtype=torch.float)
        label = torch.tensor(self.label[index])
        return feature1, feature2, label
