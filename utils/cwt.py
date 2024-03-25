import numpy as np
import scipy.signal
import scipy.io

def morse_wavelet(M, beta, gamma):
    # 定义Morse wavelet
    t = np.linspace(-1, 1, M)
    wavelet = (beta ** 0.5) * np.exp(-beta * (t ** 2)) * np.exp(1j * 2 * np.pi * gamma * t)
    return wavelet

def wavelet_func(t, width):
    return morse_wavelet(20, 3.5, 3.5)  # 使用自定义的Morse wavelet

def transform():
    for f in range(1, 2):
        # 读取数据
        file_path = f'H:\\EEG\\EEGDATA\\S{f}.mat'
        mat = scipy.io.loadmat(file_path)
        raw_data = np.asarray(mat['X_3D'])
        raw_data = np.transpose(raw_data, (2, 0, 1))

        # 定义频率范围
        scales = np.arange(1, 21)

        # 初始化一个用于存储结果的数组
        cwt_data = np.empty((5188, 124, 20, 32))

        # 对每次测试的数据进行小波变换
        for i in range(5188):
            for j in range(124):
                cwt_result = scipy.signal.cwt(raw_data[i, j, :], wavelet_func, scales)
                cwt_data[i, j, :, :] = cwt_result
                # print(f'第{i}个测试的第{j}个通道的小波变换完成')

        # 保存结果
        np.save(f'H:\\EEG\\EEGDATA\\cwt\\S{f}_cwt_data.npy', cwt_data)
        print(f'S{f} done.')
def test():
    # 读取数据
    file_path = 'H:\\EEG\\EEGDATA\\cwt\\S1_cwt_data.npy'
    cwt_data = np.load(file_path)
    print(cwt_data.shape)


if __name__ == '__main__':
    # transform()
    test()
