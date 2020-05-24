import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset


def transform(input_pose):
    return input_pose


class PoseNormalize(object):
    def __init__(self, neck):
        """
        using the location of neck to normalize
        :param neck: neck index
        :return: minus neck location
        """
        self.neck_index = neck

    @staticmethod
    def numpy_normalize(x):
        length = np.abs(np.max(x))
        return x / length

    def __call__(self, sample):
        pose = sample['key_point']
        num, _ = pose.shape
        neck = pose[self.neck_index]
        for _i in range(num):
            pose[_i, :] -= neck
        pose[:, 0] = self.numpy_normalize(pose[:, 0])
        pose[:, 1] = self.numpy_normalize(pose[:, 1])

        return {'key_point': pose.reshape(-1), 'label': sample['label']}


class ToTensor(object):
    def __call__(self, sample):
        pose = sample['key_point']
        label = sample['label']

        pose = torch.tensor(pose)
        label = torch.tensor(label)

        return {'key_point': pose, 'label': label}


class KeyPointDataset(Dataset):
    def __init__(self, json_path, key_joints, transform=None):
        super(KeyPointDataset, self).__init__()
        self.json_path = json_path
        self.key_joints = key_joints
        self.transform = transform

        f = open(self.json_path, 'r')
        self.data = json.load(f)
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        person_data = self.data[idx]
        if self.transform is None:
            pose = np.array(person_data['key_point'])[self.key_joints, :].reshape(-1)
            label = np.array(person_data['label'])
            return {'key_point': pose, 'label': label}
        else:
            pose = np.array(person_data['key_point'])[self.key_joints, :]
            label = np.array(person_data['label'])
            sample = {'key_point': pose, 'label': label}
            sample = self.transform(sample)
            return sample
