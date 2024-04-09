###################### CWT  ###########################################
import numpy as np
import scipy.io as sio
import pywt


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


basePath = "H:\\EEG\\EEGDATA\\EEG72\\"

sampling_rate = 62.5
gamma_scales = np.arange(0.25, 0.5, step=0.05)  # [62.5        52.08333333 44.64285714 39.0625     34.72222222]
beta_scales = np.arange(0.5, 1, step=0.1)  # [31.25       26.04166667 22.32142857 19.53125    17.36111111]
alpha_scales = np.arange(1, 2, step=0.2)  # [15.625      13.02083333 11.16071429  9.765625    8.68055556]
theta_scales = np.arange(2, 4, step=0.4)  # [7.8125     6.51041667 5.58035714 4.8828125  4.34027778]
dela_scales = np.arange(4, 8, step=0.8)  # [3.90625    3.25520833 2.79017857 2.44140625 2.17013889]
scales = np.concatenate((gamma_scales, beta_scales, alpha_scales, theta_scales, dela_scales))

chan2 = len(scales)

for subID in range(1, 10 + 1):
    print("sub:", subID)
    eegPath = basePath + "S%d.mat" % subID
    data = sio.loadmat(eegPath)
    eegData = data['X_3D']  # [124,32,5186]
    eegData = eegData.transpose(2, 0, 1)

    # X = channelSelect(eegData, 20)  # reduce data
    X = eegData

    X_cwt = np.zeros((X.shape[0], X.shape[1], chan2, X.shape[2]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_cwt[i][j], _ = pywt.cwt(X[i][j], scales, 'mexh', 1.0 / sampling_rate)
    np.save(basePath + "cwt\\sub%d_cwt.npy" % subID, X_cwt)
    print("Done!")
