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


# grid search params
voc_sizes = [100000, 400000]
rnn_hidden_sizes = [100, 400, 1600]
dropouts = [0.2, 0.5]


def hparam_to_label(tp):
    label = 'rnn'
    label += ' voc' + str(int(tp[0]))
    label += ' rhs' + str(int(tp[1]))
    label += ' drop' + str(int(tp[2] * 100))
    return label


for hp_tuple in list(itertools.product(voc_sizes, rnn_hidden_sizes, dropouts)):
    voc_size, rnn_hidden, dropout = tuple(hp_tuple)
    hparam_overrides = {HyperParamKey.VOC_SIZE: voc_size,
                        HyperParamKey.RNN_HIDDEN_SIZE: rnn_hidden,
                        HyperParamKey.DROPOUT_FC: dropout,
                        HyperParamKey.DROPOUT_RNN: dropout}

    mgr.hparams.update(hparam_overrides)
    mgr.load_data(mm.loaderRegister.SNLI)
    mgr.new_model(mm.modelRegister.NLIRNN, label=hparam_to_label(hp_tuple))
    mgr.train()
    mgr.graph_training_curves()
    mgr.dump_model()

mgr.get_results().to_csv('model_saves/rnn_rerun_results.csv')
