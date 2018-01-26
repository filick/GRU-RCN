import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import math
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader as Video


__all__ = ['default_loader', 'UCF101Folder']


def _same_transform(img, trans, state):
    random.setstate(state)
    return trans(img)

def default_loader(path, frame_selector, transform=None):
    video = Video(path)
    frames = frame_selector.select(video.nframes, video.fps)
    if transform:
        state = random.getstate()
        data = [_same_transform((video.get_frame(f / video.fps)), transform, state) for f in frames]
        random.seed(None)
    else:
        data = [video.get_frame(f / video.fps) for f in frames]
    if isinstance(data[0], np.ndarray):
        return np.stack(data)
    elif isinstance(data[0], torch.Tensor):
        return torch.stack(data)
    else:
        raise RuntimeError()


class UCF101Folder(Dataset):

    def __init__(self, data_root, split_root, mode, frame_selector, split=0,
                 transform=None, target_transform=None, loader=default_loader):
        classes = [line.split(' ')[1].strip() for line in open(os.path.join(split_root, 'classInd.txt'), 'r')]
        class_to_idx = {classes[i] : i for i in range(len(classes))}

        imgs = []
        part = ['01', '02', '03'][split - 1]
        for line in open(os.path.join(split_root, mode + 'list' + part + '.txt'), 'r'):
            l = line.strip().split(' ')[0].split('/')
            path = os.path.join(data_root, *l)
            imgs.append((path, class_to_idx[l[0]]))

        self.data_root = data_root
        self.split_root = split_root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.frame_selector = frame_selector
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        data = self.loader(path, self.frame_selector, self.transform)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.imgs)