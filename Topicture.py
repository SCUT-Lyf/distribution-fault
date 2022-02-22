import numpy as np
import matplotlib.pyplot as plt
import os
from PyEMD import EEMD,EMD,Visualisation
from scipy.signal import hilbert
from vmdpy import VMD

def decompose_lw(signal, t, method = 'eemd', K = 5, draw = 1):
    name = ['emd', 'eemd', 'vmd']
    idx = name.index(method)

    if idx == 0:
        emd = EMD()
        IMFs = emd.emd(signal)

    if idx == 2:
        alpha = 2000
        tau = 0
        DC = 0
        init = 1
        tol = 1e-7
        IMFs, _, _ = VMD(signal,alpha,tau,K,DC,init,tol)

    if idx == 1:
        eemd = EEMD()
        emd = eemd.EMD()
        IMFs = eemd.eemd(signal,t)

    if draw == 1:
        plt.figure()
        for i in enumerate(IMFs):
            plt.subplot(len(IMFs),1,i+1)
            plt.plot(t,IMFs[i])
            if i == 0:
                plt.rcParams['font.sans-serif'] = 'Times New Roman'
                plt.title('Decomposition Signal', fontsize = 14)
            elif i == len(IMFs) - 1:
                plt.rcParams['font.sans-serif'] = 'Times New Roman'
                plt.xlabel('Time/s')

    return IMFs

def hhtlw(IMFs, t, f_range = [0,500], t_range = [0,1], ft_size = [128,128], draw = 1):
    fmin, fmax = f_range[0], f_range[1]
    tmin, tmax = t_range[0], t_range[1]
    fdim, tdim = ft_size[0], ft_size[1]
    dt = (tmax - tmin)/(tdim - 1)
    df = (fmax - fmin)/(fdim - 1)
    vis = Visualisation()
    c_matrix = np.zeros((fdim, tdim))
    for imf in IMFs:
        imf = np.array([imf])
        freqs = abs(vis._calc_inst_freq(imf, t, order=False, alpha=None))
        amp = hilbert(abs(imf))
        freqs = np.squeeze(freqs)
        amp = np.squeeze(amp)

        temp_matrix = np.zeros((fdim, tdim))
        n_matrix = np.zeros((fdim, tdim))

        for i, j, k in zip(t, freqs, amp):
            if tmin <= i <= tmax and fmin <= j <= fmax:
                temp_matrix[round((j - fmin) / df)][round((i - tmin) / dt)] += k
                n_matrix[round((j - fmin) / df)][round((i - tmin) / dt)] += 1
        n_matrix=n_matrix.reshape(-1)
        idx=np.where(n_matrix==0)[0]
        n_matrix[idx]=1
        n_matrix=n_matrix.reshape(fdim,tdim)
        temp_matrix=temp_matrix/n_matrix
        c_matrix+=temp_matrix

    t = np.linspace(tmin, tmax, tdim)
    f = np.linspace(fmin, fmax, fdim)


