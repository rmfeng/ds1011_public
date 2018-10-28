"""
Used for trace
"""
import pandas as pd
import numpy as np
import itertools
import torch
from config.constants import HyperParamKey
from config import basic_conf as conf
conf.DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from libs import ModelManager as mm
import logging
conf.init_logger(logging.INFO, logfile=None)

mgr = mm.ModelManager()
mgr.load_data(mm.loaderRegister.SNLI)
mgr.new_model(mm.modelRegister.NLICNN)
mgr.train()
