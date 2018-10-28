"""
Used for trace
"""
import pandas as pd
import numpy as np
import itertools
from libs import ModelManager as mm
from config.constants import HyperParamKey, LoadingKey
from config import basic_conf as conf
import logging
conf.init_logger(logging.INFO, logfile=None)
logger = logging.getLogger('__main__')

mgr = mm.ModelManager()
best_model_label = 'modlrinv100decay95rhs100drop25'

# check we are loading the right model
mgr.load_data(mm.loaderRegister.SNLI)
hparams = {
    HyperParamKey.LR: 0.01,
    HyperParamKey.SCHEDULER_GAMMA: 0.95,
    HyperParamKey.BATCH_SIZE: 256,
    HyperParamKey.RNN_HIDDEN_SIZE: 100,
    HyperParamKey.DROPOUT_FC: 0.25,
    HyperParamKey.DROPOUT_RNN: 0.25,
    HyperParamKey.NO_IMPROV_LOOK_BACK: 25
}
mgr.hparams.update(hparams)
mgr.new_model(mm.modelRegister.NLIRNN, label=best_model_label)
mgr.load_model(which_model=LoadingKey.LOAD_BEST)
mgr.propogate_params()
acc = mgr.model.eval_model(mgr.dataloader.loaders['val'])[0]
logger.info("Validation Error for Loaded Model = %s" % acc)


# tuning each of the MNLI sets an additional 5 epochs as smaller batch size at 1/2 the learning rate and higher dropout
for genre in conf.GENRE_LIST:
    hparams = {
        HyperParamKey.LR: 0.005,
        HyperParamKey.SCHEDULER_GAMMA: 0.95,
        HyperParamKey.BATCH_SIZE: 32,
        HyperParamKey.RNN_HIDDEN_SIZE: 100,
        HyperParamKey.DROPOUT_FC: 0.5,
        HyperParamKey.DROPOUT_RNN: 0.5,
        HyperParamKey.NO_IMPROV_LOOK_BACK: 25
    }
    mgr.hparams.update(hparams)
    mgr.new_model(mm.modelRegister.NLIRNN, label=genre)
    mgr.load_model(path_to_model_ovrd='model_saves/%s/model_best.tar' % best_model_label)
    mgr.propogate_params()
    mgr.load_data(mm.loaderRegister.MNLI, genre=genre)
    mgr.model.output_dict['initial_acc'] = mgr.model.eval_model(mgr.dataloader.loaders['val'])
    mgr.model.add_epochs(5, reset_curve=True)
    mgr.train()
    mgr.graph_training_curves()

mgr.get_results().to_csv('model_saves/mnli_tuning.csv')
