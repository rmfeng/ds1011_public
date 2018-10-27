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
decay_rates = [0.95, 0.8]
rnn_hidden_sizes = [50, 100, 200]
dropouts = [0.25, 0.5, 0.75]


def hparam_to_label(tp):
    label = 'mod'
    label += '_lrinv' + str(int(1 / tp[0]))
    label += '_decay' + str(int(tp[1] * 100))
    label += '_rhs' + str(int(tp[2]))
    label += '_drop' + str(int(tp[3] * 100))
    return label


for hp_tuple in list(itertools.product(lr_list, decay_rates, rnn_hidden_sizes, dropouts)):
    lr, decay, rnn_hidden, dropout = tuple(hp_tuple)
    hparam_overrides = {HyperParamKey.LR: lr,
                        HyperParamKey.SCHEDULER_GAMMA: decay,
                        HyperParamKey.RNN_HIDDEN_SIZE: rnn_hidden,
                        HyperParamKey.DROPOUT_FC: dropout,
                        HyperParamKey.DROPOUT_RNN: dropout}

    mgr.hparams.update(hparam_overrides)
    mgr.new_model(mm.modelRegister.NLIRNN, label=hparam_to_label(hp_tuple))
    mgr.train()
    mgr.dump_model()

mgr.get_results().to_csv('model_saves/results.csv')
