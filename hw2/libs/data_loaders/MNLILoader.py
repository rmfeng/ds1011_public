"""
child class implementation that loads the MNLI data

Note: The DataLoader sorts the batches by decending length on sentence 1

The data is keyed bt the genre
"""

from libs.data_loaders.SNLILoader import SNLILoader
from libs.data_loaders.SNLILoader import snli_collate_func, SNLIDatum, SNLIDataset, load_pretrained_vectors
import numpy as np
import torch
from torch.utils.data import DataLoader
from config.constants import HyperParamKey, PathKey, LoaderParamKey
from config.basic_conf import DEVICE
from torch.utils.data import Dataset
import io
import logging

logger = logging.getLogger('__main__')
LABEL2INDEX = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
PAD_TOKEN, PAD_IDX = '<pad>', 0
UNK_TOKEN, UNK_IDX = '<unk>', 1


class MNLILoader(SNLILoader):
    def __init__(self, cparams, hparams, tqdm, train_size=None, genre=None):
        super().__init__(cparams, hparams, tqdm, train_size)
        self.genre = genre

    def _load_raw_data(self):
        """ overloads the load raw data method"""
        logger.info("loading raw training data set ...")
        train_set = construct_dataset(self.cparams[PathKey.MNLI_TRAIN_PATH], self.genre, self.train_size)

        logger.info("loading raw training data set ...")
        val_set = construct_dataset(self.cparams[PathKey.MNLI_VAL_PATH], self.genre)

        logger.info("loading pre-trained word vectors, building vocab ...")
        token2id, id2token, loaded_embeddings = load_pretrained_vectors(self.cparams[PathKey.PRETRAINED_PATH]
                                                                        , self.hparams[HyperParamKey.VOC_SIZE])
        self.loaded_embeddings = loaded_embeddings
        self.id2token = id2token
        self.token2id = token2id
        self.act_vocab_size = len(id2token)

        logger.info("converting training set to index ...")
        self.data['train'] = self._convert_data_to_idx(train_set)

        logger.info("converting val set to index ...")
        self.data['val'] = self._convert_data_to_idx(val_set)

    '''
    def _convert_data_to_idx(self, data):
        """ iterates through each data item in train and val, sets its tokens and token indices, MNLI variant """
        for key in data.keys():
            for rec in data[key]:
                tokens1 = rec.raw_tuple[0].split(" ")
                tokens2 = rec.raw_tuple[1].split(" ")
                token_idx1 = [self.token2id[x] if x in self.token2id.keys() else UNK_IDX for x in tokens1]
                token_idx2 = [self.token2id[x] if x in self.token2id.keys() else UNK_IDX for x in tokens2]
                rec.set_tokens((tokens1, tokens2))
                rec.set_token_idx((token_idx1, token_idx2))
        return data

    def _data_to_pipe(self):
        """ pipes the data through to pytorch's DataLoader object """
        logger.info("piping data into pytorch DataLoaders ...")
        # training
        dict_train = self.data['train']
        self.loaders['train'] = {}
        for key in dict_train.keys():
            ds = SNLIDataset(dict_train[key])
            dl = DataLoader(dataset=ds,
                            batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                            collate_fn=snli_collate_func,
                            shuffle=True)
            self.loaders['train'][key] = dl

        # validation
        dict_val = self.data['val']
        self.loaders['val'] = {}
        for key in dict_val.keys():
            ds = SNLIDataset(dict_val[key])
            dl = DataLoader(dataset=ds,
                            batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                            collate_fn=snli_collate_func,
                            shuffle=False)
            self.loaders['val'][key] = dl
    '''


def construct_dataset(file_path, target_genre, size=None):
    """
    Loads a file into a dataset list
    :param file_path: path to file
    :param target_genre: the specified genre
    :param size: number of rows to read
    :return: dict of lists of SNLI Datums, keyed by genre
    """
    data_list = []
    for i, l in enumerate(open(file_path, "r")):
        if i != 0:
            raw_parts = l.split('\t')
            parts = preprocess(raw_parts)
            raw_tuple = (parts[0], parts[1])

            # hard coded skips for that 1 bad data point
            if len(raw_tuple[0]) < 5000:
                label = LABEL2INDEX[parts[2]]
                genre = parts[3]

                if genre == target_genre:
                    data_list.append(SNLIDatum(raw_tuple, label))
        if size is not None and i >= size:
            break
    return data_list


def preprocess(raw_parts):
    """ preprocesses raw data by lower casing and removing \n"""
    parts = []
    for raw_part in raw_parts:
        parts.append(raw_part.lower().replace('\n', ''))
    return parts
