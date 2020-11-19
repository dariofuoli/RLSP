import os
from parameters import params
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import random
from skimage.io import imread, imsave
import numpy as np


# Dataset
class Data(Dataset):
    def __init__(self):

        root = params["dataset root"]
        self.videos = 4
        self.seq_len = params["validation sequence length"]

        self.file_list_input_lr = []
        for i in range(0, self.videos):
            path_input = os.path.abspath(root + "val/LR/" + str(i).zfill(3) + "/*.npy")
            self.file_list_input_lr.append(sorted(glob.glob(path_input)))

        self.file_list_input_hr = []
        for i in range(0, self.videos):
            path_input = os.path.abspath(root + "val/HR/" + str(i).zfill(3) + "/*.npy")
            self.file_list_input_hr.append(sorted(glob.glob(path_input)))

    def __len__(self):
        return self.videos

    def __getitem__(self, idx):

        input_list_lr = self.file_list_input_lr[idx]
        input_list_hr = self.file_list_input_hr[idx]

        vid_len = len(input_list_lr)

        seq_input_lr = []
        for i in range(self.seq_len):

            input = np.load(input_list_lr[i])
            seq_input_lr.append(input)

        seq_input_hr = []
        for i in range(self.seq_len):

            input = np.load(input_list_hr[i])
            seq_input_hr.append(input)

        x = np.moveaxis(np.array(seq_input_lr, dtype=np.float32), -1, -3)
        y = np.moveaxis(np.array(seq_input_hr, dtype=np.float32), -1, -3)

        return torch.from_numpy(x), torch.from_numpy(y)


# Dataloader
class Loader:
    def __init__(self):

        self.dataset = Data()
        self.epoch_size = self.dataset.videos
        self.batch_size = 1
        self.shuffle = False
        self.num_workers = 1

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=self.num_workers,
                                     pin_memory=True)

