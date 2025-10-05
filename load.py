import torch
import pandas as pd
import numpy as np
from scipy import signal
import os, scipy
import random
from config import data_aug_config

def Load_Dataset_fnirs(data_path, start=100, end=300):
    """
    Load Dataset B

    Args:
        data_path (str): dataset path.
        start (int): start sampling point, default=100.
        end (int): end sampling point, default=300.

    Returns:
        feature : fNIRS signal data.
        label : fNIRS labels.
    """
    feature = []
    label = []
    for sub in range(1, 30):
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_oxy.xls'
        oxy = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_deoxy.xls'
        deoxy = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_desc.xls'
        desc = pd.read_excel(name, header=None)

        HbO = []
        HbR = []
        for i in range(1, 61):
            name = 'Sheet' + str(i)
            HbO.append(oxy[name].values)
            HbR.append(deoxy[name].values)

        # (60, 350, 36) --> (60, 36, 350)
        HbO = np.array(HbO).transpose((0, 2, 1))
        HbR = np.array(HbR).transpose((0, 2, 1))
        desc = np.array(desc)

        HbO_LMI = []
        HbO_RMI = []
        HbR_LMI = []
        HbR_RMI= []
        for i in range(60):
            if desc[i, 0] == 1:
                HbO_LMI.append(HbO[i, :, start:end])
                HbR_LMI.append(HbR[i, :, start:end])
            elif desc[i, 0] == 2:
                HbO_RMI.append(HbO[i, :, start:end])
                HbR_RMI.append(HbR[i, :, start:end])

        # (30, 36, 200) --> (30, 1, 36, 200)
        HbO_LMI = np.array(HbO_LMI).reshape((30, 1, 36, end-start))
        HbO_RMI = np.array(HbO_RMI).reshape((30, 1, 36, end-start))
        HbR_LMI = np.array(HbR_LMI).reshape((30, 1, 36, end-start))
        HbR_RMI = np.array(HbR_RMI).reshape((30, 1, 36, end-start))

        # (30, 2, 36, 200)
        HbO_LMI = np.concatenate((HbO_LMI, HbR_LMI), axis=1)
        HbO_RMI = np.concatenate((HbO_RMI, HbR_RMI), axis=1)

        for i in range(30):
            feature.append(HbO_LMI[i, :, :, :])
            feature.append(HbO_RMI[i, :, :, :])
            label.append(0)
            label.append(1)

        print(str(sub) + '  OK')

    feature = np.array(feature).reshape((29, -1, 2, 36, end-start))
    label = np.array(label).reshape(29, -1)

    print('fnirs feature ', feature.shape)
    print('fnirs label ', label.shape)

    return feature, label

def Load_Dataset_EEG(data_path):
    X, Y = [], []
    for i in range(0, 29):
        mat_path = os.path.join(data_path, str(i + 1) + '.mat')
        data = scipy.io.loadmat(mat_path)
        # x:(4200, 30, 60) y:(2, 60) 
        # x时间点是 -10~25, 每秒采样120个点
        start_time = 0
        end_time = 10
        x = data['x'][(start_time + 10) * 120: (end_time + 10) * 120, :, :]
        y = data['y'][1] ## y 直接取 1 就表示它的类别 idx
        # 根据 y 将 x 切分成左右        
        left_x, left_y, right_x, right_y = [], [], [], []
        for j in range(60):
            if y[j] == 0:
                left_x.append(x[:, :, j])
                left_y.append(y[j])
            elif y[j] == 1:
                right_x.append(x[:, :, j])
                right_y.append(y[j])
        # 将 x 写入列表中
        x, y = [], []
        for j in range(30):
            x.append(left_x[j])
            x.append(right_x[j])
            y.append(left_y[j])
            y.append(right_y[j])

        x = np.stack(x, axis = 2).transpose(2, 1, 0)[:,None,:,:]
        y = np.asarray(y)
        # print(x.shape, y.shape, y)
        X.append(x), Y.append(y)
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    print("eeg shape: ", X.shape, Y.shape)
    return X, Y

def Load_Dataset_WG(data_path, start, end):
    """
    Load Dataset WG
    start: 起始点，end: 结束点 单位是秒
    """
    eeg_feature, fnirs_feature, labels = [], [], []
    for i in range(1, 27):
        file_name = os.path.join(data_path, f"vp%03d.npz" % i)
        # 取出数据
        data = np.load(file_name)
        eeg, hbo, hbr, label = data['eeg'], data['hbo'], data['hbr'], data['label']
        # 扩展维度
        hbo = np.expand_dims(hbo, axis=1)
        hbr = np.expand_dims(hbr, axis=1)
        eeg = np.expand_dims(eeg, axis=1)
        
        fnirs = np.concatenate([hbo, hbr], axis=1)
        
        eeg_feature.append(eeg[:, :, :, (start + 5) * 120 : (end + 5) * 120])
        fnirs_feature.append(fnirs[:, :, :, (start + 5) * 10 : (end + 5) * 10])
        labels.append(label[1])
    eeg_feature = np.stack(eeg_feature, axis=0)
    fnirs_feature = np.stack(fnirs_feature, axis=0)
    labels = np.stack(labels, axis = 0)
    print("Load_Dataset_WG all shapes: ", eeg_feature.shape, fnirs_feature.shape, labels.shape)

    return eeg_feature, fnirs_feature, labels, labels
    
class Dataset(torch.utils.data.Dataset):
    """
    Load data for training

    Args:
        feature: input data.
        label: class for input data.
        transform: Z-score normalization is used to accelerate convergence (default:True).
    """
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        print(self.feature.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        x = self.feature[item].clone()
        y = self.label[item]
        
        # z-score normalization
        if self.transform:
            mean, std = x.mean(), x.std()
            x = (x - mean) / std

        return x, y

class EnhanceDataset(torch.utils.data.Dataset):
    """
    Load data for training

    Args:
        feature: input data.
        label: class for input data.
        transform: Z-score normalization is used to accelerate convergence (default:True).
    """
    def __init__(self, eeg, fnirs, label, transform=True):
        self.eeg_feature = eeg
        self.fnirs_feature = fnirs
        self.label = label
        self.transform = transform
        self.eeg_feature = torch.tensor(self.eeg_feature, dtype=torch.float)
        self.fnirs_feature = torch.tensor(self.fnirs_feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        print("EnhanceDataset: ", self.eeg_feature.shape, self.fnirs_feature.shape, self.label.shape)

    def __len__(self):
        return len(self.label)
    
    def time_warp(self, x, max_warp=0.1):
        """改进的时间扭曲增强，真正实现时间维度拉伸"""
        length = x.shape[-1]
        warp_size = int(length * max_warp)
        
        # 随机选择扭曲区域
        start = random.randint(0, length - 2*warp_size)
        end = start + 2*warp_size
        
        # 创建原始时间轴
        original = np.arange(length)
        
        # 创建扭曲后的时间轴
        if random.random() > 0.5:
            # 拉伸模式 - 将中间区域放大
            warped = np.concatenate([
                np.linspace(0, start, start),  # 前段不变
                np.linspace(start, end, int(warp_size*3)),  # 中间区域拉伸
                np.linspace(end, length-1, length-end)  # 后段不变
            ])
        else:
            # 压缩模式 - 将中间区域缩小
            warped = np.concatenate([
                np.linspace(0, start, start),  # 前段不变
                np.linspace(start, end, int(warp_size*0.5)),  # 中间区域压缩
                np.linspace(end, length-1, length-end)  # 后段不变
            ])
        
        # 确保长度一致
        if len(warped) != length:
            warped = np.interp(np.linspace(0, 1, length), 
                            np.linspace(0, 1, len(warped)), 
                            warped)
        
        # 应用时间扭曲
        warped_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                warped_x[i,j,:] = np.interp(warped, original, x[i,j,:])
        
        return warped_x

    def transform_data(self, x, data_type):
        if self.transform and data_aug_config.use_data_aug:
            if random.random() > 0.2:
                # 通用增强（适用于EEG和fNIRS）
                if random.random() > 0.35 and data_aug_config.use_generic:
                    # 时域随机平移
                    shift = random.randint(1, 3)
                    x = np.roll(x, shift=shift, axis=-1)
                    
                if random.random() > 0.35 and data_aug_config.use_generic:
                    # 添加高斯噪声
                    noise = np.random.normal(0, 0.05, size=x.shape)
                    x = x + noise
                
                if (random.random() > 0.35) and (data_type == "eeg") and data_aug_config.use_eeg:
                    # print("********x shape: ", x.shape)
                    # 时间扭曲增强
                    x = self.time_warp(x)

                # fNIRS特异性增强        
                if data_type == "fnirs" and data_aug_config.use_fnirs:  # fNIRS数据 (2, 36, 30)
                    if random.random() > 0.35:
                        # 幅度缩放
                        scale = random.uniform(0.9, 1.1)
                        x = x * scale
            
        mean, std = x.mean(), x.std()
        x = (x - mean) / std
        return x
    def __getitem__(self, item):
        # print("getitem: ", item)
        eeg_x, fnirs_x, y = self.eeg_feature[item], self.fnirs_feature[item], self.label[item]
        # z-score normalization
        eeg_x = self.transform_data(eeg_x, "eeg")
        fnirs_x = self.transform_data(fnirs_x, "fnirs")
        # print(type(eeg_x), type(fnirs_x), y)
        return torch.Tensor(eeg_x).to(torch.float32), torch.Tensor(fnirs_x).to(torch.float32), y

if __name__ == "__main__":
    # fnirs_data_path = "/home/ubuntu/datasets/predata_MI/"
    # eeg_data_path = "/home/ubuntu/datasets/eeg_results/"
    # # fnirs_feature, fnirs_label = Load_Dataset_fnirs(fnirs_data_path, 100, 200)
    # # eeg_feature, eeg_label = Load_Dataset_EEG(eeg_data_path)
    # eeg_feature, fnirs_feature, eeg_label = np.random.rand(384, 1, 30, 360), np.random.rand(384, 2, 36, 30), np.random.rand(384)
    # dataset = EnhanceDataset(eeg_feature, fnirs_feature, eeg_label, True)
    # dataset.time_warp(eeg_feature[0])

    # print(fnirs_feature.shape, fnirs_label.shape)
    # print(eeg_feature.shape, eeg_label.shape)
    # print((eeg_label - fnirs_label).sum())
    
    # fnirs shape: (29, 60, 2, 36, 100) (29, 60)
    # eeg shape: (29, 60, 1, 30, 1200) (29, 60)
    
    wg_data_path = "/home/ubuntu/datasets/WG_dataset/epoch/"
    Load_Dataset_WG(wg_data_path, 0, 10)
    # Load_Dataset_WG all shapes:  (26, 60, 1, 28, 1200) (26, 60, 2, 36, 100) (26, 60)