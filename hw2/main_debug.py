"""
Used for trace
"""
import pandas as pd
import numpy as np
from libs import ModelManager as mm
from config import basic_conf as conf
import logging
conf.init_logger(logging.INFO, logfile=None)


mgr = mm.ModelManager()
mgr.load_data(mm.loaderRegister.SNLI, train_size=8)
mgr.new_model(mm.modelRegister.NLIRNN)
mgr.train()
