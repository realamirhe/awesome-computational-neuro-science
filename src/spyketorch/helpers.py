import os
import random
from collections import namedtuple
from os.path import join as pathjoin
from shutil import copyfile
from scipy.spatial import distance

import numpy as np
import torch


# *****************
#      Helpers
# *****************

def reset_seed(cuda=False):
    SEED = 2020
    random.seed(SEED)
    np.random.seed(SEED)
    if cuda:
        torch.cuda.manual_seed_all(SEED)
    else:
        torch.manual_seed(SEED)


def argument_parser(arguments_mapping, typename='Args'):
    parser = namedtuple(typename.title(), arguments_mapping.keys())
    return parser._make(arguments_mapping.values())


def save_drive(src='./data', dest='./drive', filename='snn-model.pth'):
    if not os.path.exists(dest):
        print(f'Please make {dest} directory first,',
              'There is no sudo access to make it.')
        return

    drive_model_path = pathjoin(dest, filename)
    local_model_path = pathjoin(src, filename)

    local_epoch = torch.load(local_model_path)['epoch']
    drive_epoch = torch.load(drive_model_path)['epoch']
    if local_epoch <= drive_epoch:
        print('It is not safe to save a lower or same model for',
              f'{local_epoch} and {drive_epoch}')
        return

    if os.path.isfile(local_model_path):
        copyfile(local_model_path, drive_model_path)
        print('File successfully saved in Drive!')
    else:
        print('File does not exist')


def load_drive(src='./data', dest='./drive', filename='snn-model.pth'):
    if not os.path.exists(dest):
        print(f'Please make {dest} directory first,',
              'There is no sudo access to make it.')
        return

    local_model_path = pathjoin(src, filename)
    drive_model_path = pathjoin(dest, filename)

    dangours = (
        os.path.isfile(local_model_path) and
        os.path.isfile(drive_model_path) and
        torch.load(local_model_path)['epoch'] -
        torch.load(drive_model_path)['epoch'] > 0
    )

    if not dangours:
        copyfile(drive_model_path, local_model_path)
        print(f'File {filename} successfully loaded!')
    else:
        print(f'File {filename} does not exist, ',
              f'or can not safely copied to your {dest}')


def measure():
    """
     T: True labeled
     F: False labeled
     U: Unknown
    """
    return np.array(
        (0, 0, 0),
        dtype=[('T', '<f8'), ('F', '<f8'), ('U', '<f8')]
    )


def train_test_split(dataset, split=0.75):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(split*len(indices))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    print("Size of the training set:", len(train_indices))
    print("Size of the  testing set:", len(test_indices))
    return (train_indices, test_indices)


# TODO: use_balanced can be cheaper in computation
def get_decision_map(path, max_count=20, use_balanced=True):
    normalized_count = []
    decision_map = []
    for category in os.listdir(path):
        if category == 'cached':
            continue
        normalized_count.append(
            len(os.listdir(pathjoin(path, category))))

    normalized_count = np.array(normalized_count)
    if use_balanced:
        count = max_count // normalized_count.shape[0] + 1
        for i in range(len(normalized_count)):
            decision_map.extend([i] * count)
    else:
        normalized_count = np.ceil(
            normalized_count / np.sum(normalized_count) * max_count)
        for i, count in enumerate(normalized_count):
            decision_map.extend([i] * int(count))

    return decision_map[:max_count]


def rdm(matrix):
    """ Computes the representational dissimilarity matrix for one matrix. """
    return distance.squareform(distance.pdist(matrix, 'correlation'))
