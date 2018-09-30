import logging
import sys
import os
import torch

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
LOG_STREAM = sys.stdout
LOG_LEVEL_DEFAULT = logging.DEBUG


def init_logger():
    logging.basicConfig(stream=LOG_STREAM, level=LOG_LEVEL_DEFAULT, format=LOG_FORMAT)


# DEFAULT HYPERPARAMETERS
LR = 0.01                   # learnign rate
NEPOCH = 1                  # number epoch to train
BATCH_SIZE = 32             # number of data in each minibatch
NGRAM_SIZE = 2              # (1, 2, 3, 4)
VOC_SIZE = 10000            # takes top n word from the vocab
EMBEDDING_DIM = 100         # dimension size for the ngram embeddings
NGRAM_MODE = 'naive'        # {'naive', 'spacy'}
VAL_SIZE = 5000             # total data size 25k for both train/val and test
VAL_FREQ = 4                # check for early stop every n batches

# OTHER PARAMETERS
PAD_IDX = 0
UNK_IDX = 1
TRAIN_AND_VAL_SIZE = 25000  # total data size 25k for both train/val and test
TEST_SIZE = 25000           # total data size 25k for both train/val and test
DIR_DATA = r'./data/aclImdb/'
DIR_PICKLE = r'./data/pickles/'
DIR_TRAIN = os.path.join(DIR_DATA, "train")
DIR_TEST = os.path.join(DIR_DATA, "test")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_HPARAMS = {
    'LR': LR,
    'NEPOCH': NEPOCH,
    'BATCH_SIZE': BATCH_SIZE,
    'NGRAM_SIZE': NGRAM_SIZE,
    'VOC_SIZE': VOC_SIZE,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'NGRAM_MODE': NGRAM_MODE,
    'VAL_SIZE': VAL_SIZE,
    'OPTIMIZER': torch.optim.Adam,
    'VAL_FREQ': VAL_FREQ
}

DATA_HPARAMS = ['NGRAM_SIZE', 'VOC_SIZE', 'NGRAM_MODE']
