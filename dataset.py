import numpy as np
import scipy.io
import torch
from torch.utils.data.dataset import Dataset


class EEGDataset72(Dataset):

    def __init__(self, file_path, num_class=6):
        mat = scipy.io.loadmat(file_path)
        data = np.asarray(mat['X_3D'])
        data = np.transpose(data, (2, 0, 1))
        self.data = data[:, np.newaxis, :, :]

        if num_class == 6:
            self.label = np.asarray(mat['categoryLabels']).squeeze() - 1
        elif num_class == 72:
            self.label = np.asarray(mat['exemplarLabels']).squeeze() - 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        feature = torch.tensor(self.data[index, :, :, :], dtype=torch.float)
        label = torch.tensor(self.label[index])
        return feature, label
