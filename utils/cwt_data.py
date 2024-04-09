import numpy as np
import scipy.io as sio
import pywt
import concurrent.futures
import os


def channel_select(X, channel_num):
    mstd = X.mean(axis=0).std(axis=1)
    std_idx = np.argsort(mstd)[-channel_num:]
    s_idx = np.zeros(X.shape[1], dtype=bool)
    s_idx[std_idx] = True
    return X[:, s_idx, :]


def process_subject(sub_id, eeg_path):
    base_path = "/home/my/Documents/cxy/object_visual/code/clean_data/"
    sampling_rate = 62.5
    gamma_scales = np.arange(0.25, 0.5, step=0.05)
    beta_scales = np.arange(0.5, 1, step=0.1)
    alpha_scales = np.arange(1, 2, step=0.2)
    theta_scales = np.arange(2, 4, step=0.4)
    dela_scales = np.arange(4, 8, step=0.8)
    scales = np.concatenate((gamma_scales, beta_scales, alpha_scales, theta_scales, dela_scales))
    chan2 = len(scales)

    print("处理子：", sub_id)
    data = sio.loadmat(eeg_path)
    eeg_data = data['X_3D'].transpose(2, 0, 1)

    # X = channel_select(eeg_data, 20)  # 减少数据
    X = eeg_data

    X_cwt = np.zeros((X.shape[0], X.shape[1], chan2, X.shape[2]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_cwt[i, j], _ = pywt.cwt(X[i, j], scales, 'mexh', 1.0 / sampling_rate)

    filename = os.path.splitext(os.path.basename(eeg_path))[0]  # 获取原始文件名（不包括扩展名）
    np.save(base_path + "cwt/" + filename + "_cwt.npy", X_cwt)
    print("完成子：", sub_id)


def main():
    base_path = "/home/my/Documents/cxy/object_visual/code/clean_data/"
    eeg_paths = [base_path + "S%d.mat" % sub_id for sub_id in range(1, 11)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_subject, range(1, 11), eeg_paths)


if __name__ == "__main__":
    main()
