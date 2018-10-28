"""
hyperparameter settings, defaults
"""
from config.constants import HyperParamKey
import torch


DEFAULT_HPARAMS = {
    HyperParamKey.NUM_EPOCH: 10,
    HyperParamKey.LR: 0.01,
    HyperParamKey.VOC_SIZE: 100000,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 10,
    HyperParamKey.DROPOUT_RNN: 0.5,
    HyperParamKey.DROPOUT_FC: 0.5,
    HyperParamKey.BATCH_SIZE: 256,
    HyperParamKey.FC_HIDDEN_SIZE: 100,
    HyperParamKey.RNN_HIDDEN_SIZE: 50,
    HyperParamKey.CNN_HIDDEN_SIZE: 100,
    HyperParamKey.RNN_NUM_LAYERS: 1,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.EARLY_STOP_LOOK_BACK: 10,
    HyperParamKey.DECAY_LR_NO_IMPROV: 0.25,
    HyperParamKey.EARLY_STOP_REQ_PROG: 0.01,
    HyperParamKey.OPTIMIZER_ENCODER: torch.optim.Adam,
    HyperParamKey.OPTIMIZER_DECODER: torch.optim.Adam,  # not needed, but kept for future use
    HyperParamKey.SCHEDULER: torch.optim.lr_scheduler.ExponentialLR,
    HyperParamKey.SCHEDULER_GAMMA: 0.95,
    HyperParamKey.CRITERION: torch.nn.CrossEntropyLoss
}
