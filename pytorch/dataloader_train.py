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
        self.videos = 200
        self.seq_len = params["sequence length"]
        self.cropsize_h = params["crop size h"]
        self.cropsize_w = params["crop size w"]
        self.max_cropsize_h = 180
        self.max_cropsize_w = 320

        assert self.cropsize_h <= self.max_cropsize_h, "Cropsize h too large!"
        assert self.cropsize_w <= self.max_cropsize_w, "Cropsize w too large!"

        self.file_list_input_lr = []
        for i in range(0, self.videos):
            path_input = os.path.abspath(root + "train/LR/" + str(i).zfill(3) + "/*.npy")
            self.file_list_input_lr.append(sorted(glob.glob(path_input)))

        self.file_list_input_hr = []
        for i in range(0, self.videos):
            path_input = os.path.abspath(root + "train/HR/" + str(i).zfill(3) + "/*.npy")
            self.file_list_input_hr.append(sorted(glob.glob(path_input)))

    def __len__(self):
        return 10**8  # get "infinite" dataloader and omit initialization after each training epoch

    def __getitem__(self, idx):

        index = idx % self.videos

        input_list_lr = self.file_list_input_lr[index]
        input_list_hr = self.file_list_input_hr[index]

        vid_len = len(input_list_lr)
        seq_start = random.randint(0, vid_len - self.seq_len)

        rand_h = random.randint(0, self.max_cropsize_h - self.cropsize_h)
        rand_w = random.randint(0, self.max_cropsize_w - self.cropsize_w)

        seq_input_lr = []
        for i in range(self.seq_len):

            input = np.load(input_list_lr[seq_start + i])
            # crop
            input = input[rand_h:rand_h+self.cropsize_h, rand_w:rand_w+self.cropsize_w, :]
            seq_input_lr.append(input)

        seq_input_hr = []
        for i in range(self.seq_len):

            input = np.load(input_list_hr[seq_start + i])
            # crop
            input = input[4*rand_h:4*(rand_h+self.cropsize_h), 4*rand_w:4*(rand_w+self.cropsize_w), :]
            seq_input_hr.append(input)

        x = np.moveaxis(np.array(seq_input_lr, dtype=np.float32), -1, -3)
        y = np.moveaxis(np.array(seq_input_hr, dtype=np.float32), -1, -3)

        return torch.from_numpy(x), torch.from_numpy(y)


# Dataloader
class Loader:
    def __init__(self):

        self.dataset = Data()
        self.epoch_size = self.dataset.videos
        self.batch_size = params["bs"]
        self.shuffle = True
        self.num_workers = params["number of workers"]

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=self.num_workers,
                                     pin_memory=True)
