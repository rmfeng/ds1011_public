import logging
import sys
import os
import torch

# LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
LOG_FORMAT = '%(levelname)-8s %(message)s'
LOG_STREAM = sys.stdout
LOG_LEVEL_DEFAULT = logging.INFO


def init_logger():
    logging.basicConfig(stream=LOG_STREAM, level=LOG_LEVEL_DEFAULT, format=LOG_FORMAT)


# DEFAULT HYPERPARAMETERS
LR = 0.01                       # learning rate
LR_DECAY_RATE = 0.95            # decay factor for the learning rate per epoch
NEPOCH = 10                     # number epoch to train
BATCH_SIZE = 32                 # number of data in each minibatch
NGRAM_SIZE = 4                  # (1, 2, 3, 4)
VOC_SIZE = 50000                # takes top n word from the vocab
EMBEDDING_DIM = 50              # dimension size for the ngram embeddings
NGRAM_MODE = 'naive'            # {'naive', 'spacy'}
VAL_SIZE = 5000                 # total data size 25k for both train/val and test
VAL_FREQ = 4                    # check for early stop every n batches
REMOVE_STOP_WORDS = False       # whether to remove stop words in the text
REMOVE_PUNC = True              # whether to remove punctuation in the text
EARLY_STOP = True               # whether or not the model considers early stopping
EARLY_STOP_LOOKBACK = 8        # number of batches to look back when consider to early stop
EARLY_STOP_MIN_IMPROVE = 0.01   # minimum improvement required in early stop

# OTHER PARAMETERS
PAD_IDX = 0
UNK_IDX = 1
TRAIN_AND_VAL_SIZE = 25000  # total data size 25k for both train/val and test
TEST_SIZE = 25000           # total data size 25k for both train/val and test
DIR_DATA = r'./data/aclImdb/'
DIR_PICKLE = r'./data/pickles/'
DIR_RES = r'./results/'
DIR_TRAIN = os.path.join(DIR_DATA, "train")
DIR_TEST = os.path.join(DIR_DATA, "test")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'

DEFAULT_HPARAMS = {
    'LR': LR,
    'LR_DECAY_RATE': LR_DECAY_RATE,
    'NEPOCH': NEPOCH,
    'BATCH_SIZE': BATCH_SIZE,
    'NGRAM_SIZE': NGRAM_SIZE,
    'VOC_SIZE': VOC_SIZE,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'NGRAM_MODE': NGRAM_MODE,
    'VAL_SIZE': VAL_SIZE,
    'OPTIMIZER': torch.optim.Adam,
    'VAL_FREQ': VAL_FREQ,
    'REMOVE_STOP_WORDS': REMOVE_STOP_WORDS,
    'REMOVE_PUNC': REMOVE_PUNC,
    'EARLY_STOP': EARLY_STOP,
    'EARLY_STOP_LOOKBACK': EARLY_STOP_LOOKBACK,
    'EARLY_STOP_MIN_IMPROVE': EARLY_STOP_MIN_IMPROVE,
}

DATA_HPARAMS = ['NGRAM_SIZE', 'NGRAM_MODE', 'REMOVE_STOP_WORDS', 'REMOVE_PUNC']
INDEXER_HPARAMS = ['NGRAM_SIZE', 'NGRAM_MODE', 'REMOVE_STOP_WORDS', 'REMOVE_PUNC', 'VOC_SIZE', 'VAL_SIZE']
