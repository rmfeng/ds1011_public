"""
Used for trace
"""
import pandas as pd
import numpy as np
import itertools
from libs import ModelManager as mm
from config.constants import HyperParamKey
from config import basic_conf as conf
import logging
conf.init_logger(logging.INFO, logfile=None)

mgr = mm.ModelManager()
mgr.load_data(mm.loaderRegister.SNLI)

# grid search params
lr_list = [0.01, 0.001]
cnn_hidden_sizes = [50, 100, 200]
kernal_sizes = [3, 5, 7]


def hparam_to_label(tp):
    label = 'cnn'
    label += ' lrinv' + str(int(1 / tp[0]))
    label += ' hs' + str(int(tp[1]))
    label += ' kern' + str(int(tp[2]))
    return label


for hp_tuple in list(itertools.product(lr_list, cnn_hidden_sizes, kernal_sizes)):
    lr, hidden_size, kernal_size = tuple(hp_tuple)
    hparam_overrides = {HyperParamKey.LR: lr,
                        HyperParamKey.CNN_HIDDEN_SIZE: hidden_size,
                        HyperParamKey.CNN_KERNAL_SIZE: kernal_size
                        }

    mgr.hparams.update(hparam_overrides)
    mgr.new_model(mm.modelRegister.NLICNN, label=hparam_to_label(hp_tuple))
    mgr.train()
    mgr.graph_training_curves()
    mgr.dump_model()

mgr.get_results().to_csv('model_saves/cnn_results.csv')
