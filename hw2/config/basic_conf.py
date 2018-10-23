"""
non hyperparameter settings
"""
import logging
import logging.config
import torch
from config.constants import PathKey, LogConfig, ControlKey

DEFAULT_CONTROLS = {
    ControlKey.SAVE_BEST_MODEL: True,
    ControlKey.SAVE_EACH_EPOCH: True,
    PathKey.SNLI_TRAIN_PATH: 'data/nli/snli_train.tsv',
    PathKey.SNLI_VAL_PATH: 'data/nli/snli_val.tsv',
    PathKey.PRETRAINED_PATH: 'data/nli/wiki-news-300d-1M.vec',
    PathKey.MODEL_SAVES: 'model_saves/'
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_LEVEL_DEFAULT = getattr(logging, LogConfig['handlers']['default']['level'])


def init_logger(loglevel=LOG_LEVEL_DEFAULT, logfile='mt.log'):
    logging.getLogger('__main__').setLevel(loglevel)
    if logfile is None:
        LogConfig['loggers']['']['handlers'] = ['console']
        LogConfig['handlers']['default']['filename'] = 'mt.log'
    else:
        LogConfig['loggers']['']['handlers'] = ['console', 'default']
        LogConfig['handlers']['default']['filename'] = logfile
    logging.config.dictConfig(LogConfig)

