import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def make_datasets(dir_path_list):
    datasets = np.array([])
    labels = np.array([])
    for i, dir_path in enumerate(dir_path_list):
        for file in os.listdir(dir_path):
            data = pd.read_csv(os.path.join(dir_path, file), header=None, index_col=0, delim_whitespace=True,
                               skiprows=9802).to_numpy()
            # data = StandardScaler().fit_transform(data)
            data = MinMaxScaler().fit_transform(data)
            labels = np.append(labels, i).reshape(-1, 1)
            datasets = np.append(datasets, data).reshape(-1, *data.shape)
    return datasets, labels


if __name__ == '__main__':
    path_A = r"E:\大四下学期\毕业设计\数据\数据\A相接地"
    path_B = r"E:\大四下学期\毕业设计\数据\数据\B相接地"
    path_C = r"E:\大四下学期\毕业设计\数据\数据\C相接地"
    path_N = r"E:\大四下学期\毕业设计\数据\数据\无故障"
    datasets, labels = make_datasets([path_N, path_A, path_B, path_C])
    np.savez(r"../datasets/datasets1_minmax", datasets=datasets, labels=labels)
