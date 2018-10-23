"""
child class implementation that loads the NLI data
"""

from libs.data_loaders.BaseLoader import BaseLoader
import numpy as np
import torch
from torch.utils.data import DataLoader
from config.constants import HyperParamKey, PathKey, LoaderParamKey
from config.basic_conf import DEVICE
from torch.utils.data import Dataset
from collections import Counter
import os
import io
import logging
import string

logger = logging.getLogger('__main__')
LABEL2INDEX = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


class SNLILoader(BaseLoader):
    def __init__(self, cparams, hparams, tqdm, train_size=None):
        super().__init__(cparams, hparams, tqdm)
        self.train_size = train_size
        self.pretrained_vec = None

    def load(self):
        self._load_raw_data()

    def _load_raw_data(self):
        """ loads raw data """
        # raw data
        train_set = construct_dataset(self.cparams[PathKey.SNLI_TRAIN_PATH], self.train_size)
        val_set = construct_dataset(self.cparams[PathKey.SNLI_VAL_PATH], self.train_size)
        self.pretrained_vec = load_pretrained_vectors(self.cparams[PathKey.PRETRAINED_PATH]
                                                      , self.hparams[HyperParamKey.VOC_SIZE])

        self.data['train'] = train_set
        self.data['val'] = val_set


class SNLIDatum:
    """
    Class that represents a train/validation datum
    """
    def __init__(self, raw_tuple, label):
        self.raw_tuple = raw_tuple
        self.tokens_tuple = None
        self.token_idx_tuple = None
        self.label = label

    def set_token_idx(self, token_idx_tuple):
        self.token_idx_tuple = token_idx_tuple

    def set_tokens(self, tokens_tuple):
        self.tokens_tuple = tokens_tuple


class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        :param data_list: list of IMDBDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx_tuple, label = self.data_list[key].token_idx_tuple, self.data_list[key].label
        return [token_idx_tuple, len(token_idx_tuple), label]


def construct_dataset(file_path, size=None):
    """
    Loads a file into a dataset list
    :param file_path: path to file
    :param size: number of rows to read
    :return: list of NLIDatum
    """
    data_list = []
    for i, l in enumerate(open(file_path, "r")):
        if i != 0:
            raw_parts = l.split('\t')
            parts = preprocess(raw_parts)
            raw_tuple = (parts[0], parts[1])
            label = LABEL2INDEX[parts[2]]
            data_list.append(SNLIDatum(raw_tuple, label))
        if size is not None and i >= size:
            break
    return data_list


def preprocess(raw_parts):
    """ preprocesses raw data by lower casing and removing \n"""
    parts = []
    for raw_part in raw_parts:
        parts.append(raw_part.lower().replace('\n', ''))
    return parts


def load_pretrained_vectors(fname, rows=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        if rows is not None and i >= rows - 1:
            break
    return data
