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
import logging
import string

logger = logging.getLogger('__main__')


class NLILoader(BaseLoader):
    def __init__(self, cparams, hparams, tqdm):
        super().__init__(cparams, hparams, tqdm)
        pass

    def load(self):
        return {}
        pass
