import numpy as np
from scipy.signal import cwt, morlet
import scipy.io
import matplotlib.pyplot as plt


def transform():
    savePath = 'H:\\EEG\\EEGDATA\\EEG72-CWT\\'

    data_shape = (5188, 124, 32)

    # 设置连续小波变换的参数
    widths = np.arange(1, 51)  # 尺度参数范围

    # 创建空数组来存储结果
    transformed_data = np.zeros((data_shape[0], data_shape[1], len(widths), data_shape[2]), dtype=np.complex64)

    for f in range(1, 11):
        # 读取数据
        file_path = f'H:\\EEG\\EEGDATA\\EEG72\\S{f}.mat'
        mat = scipy.io.loadmat(file_path)
        raw_data = np.asarray(mat['X_3D'])
        raw_data = np.transpose(raw_data, (2, 0, 1))

        # 对每个通道进行连续小波变换
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                signal = raw_data[i, j, :]
                # 进行连续小波变换
                coefficients = cwt(signal, morlet, widths)  # 使用Morlet小波作为基函数
                transformed_data[i, j, :, :] = coefficients

        # 保存结果
        np.save(savePath + f'S{f}_CWT.npy', transformed_data)
        print(f'S{f} done')


def test():
    data_shape = (5188, 124, 32)

    # 读取数据
    path = 'continuous_wavelet.npy'
    transformed_data = np.load(path)

    # 可视化结果（这部分可选）
    # 选择一个通道的连续小波变换结果进行可视化
    channel_index = 0  # 选择第一个通道进行可视化
    expt_index = 0  # 选择第一个实验进行可视化

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(transformed_data[expt_index, channel_index, :, :]), extent=[0, data_shape[2], 1, 20],
               cmap='jet', aspect='auto')
    plt.xlabel('时间点')
    plt.ylabel('频率（Hz）')
    plt.colorbar(label='变换系数的绝对值')
    plt.title('连续小波变换结果')
    plt.show()


if __name__ == '__main__':
    transform()
    # test()
