import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset


def transform(input_pose):
    return input_pose


class KeyPointDataset(Dataset):
    def __init__(self, json_path, key_joints):
        super(KeyPointDataset, self).__init__()
        self.json_path = json_path
        self.key_joints = key_joints

        f = open(self.json_path, 'r')
        self.data = json.load(f)
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        person_data = self.data[idx]
        pose = np.array(person_data['key_point'])[self.key_joints, :].reshape(-1)
        label = np.array(person_data['label'])

        return {'key_point': pose, 'label': label}
