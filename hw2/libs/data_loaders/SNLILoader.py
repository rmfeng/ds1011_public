"""
child class implementation that loads the NLI data

Note: The DataLoader sorts the batches by decending length on sentence 1
"""

from libs.data_loaders.BaseLoader import BaseLoader
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


class SNLILoader(BaseLoader):
    def __init__(self, cparams, hparams, tqdm, train_size=None):
        super().__init__(cparams, hparams, tqdm)
        self.train_size = train_size
        self.num_classes = len(LABEL2INDEX.keys())
        self.loaded_embeddings = None
        self.vocab_words = None
        self.id2token = None
        self.token2id = None
        self.act_vocab_size = None

    def load(self):
        self._load_raw_data()
        self._data_to_pipe()
        return {
            LoaderParamKey.ACT_VOCAB_SIZE: self.act_vocab_size,
            LoaderParamKey.PRETRAINED_VECS: self.loaded_embeddings,
            LoaderParamKey.EMBEDDING_DIM: self.loaded_embeddings.shape[1],
            LoaderParamKey.NUM_CLASSES: self.num_classes
        }

    def _load_raw_data(self):
        """ loads raw data """
        # raw data
        logger.info("loading raw training data set ...")
        train_set = construct_dataset(self.cparams[PathKey.SNLI_TRAIN_PATH], self.train_size)

        logger.info("loading raw training data set ...")
        val_set = construct_dataset(self.cparams[PathKey.SNLI_VAL_PATH])

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

    def _convert_data_to_idx(self, data):
        """ iterates through each data item in train and val, sets its tokens and token indices """
        for rec in data:
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
        train_set = SNLIDataset(self.data['train'])
        val_set = SNLIDataset(self.data['val'])
        self.loaders['train'] = DataLoader(dataset=train_set,
                                           batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                                           collate_fn=snli_collate_func,
                                           shuffle=True)

        self.loaders['val'] = DataLoader(dataset=val_set,
                                         batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                                         collate_fn=snli_collate_func,
                                         shuffle=False)


def snli_collate_func(batch):
    """
    collate founction that goes to the DataLoader from pytorch
    :param batch:
    :return: returns tensors of the data in order: []
    """
    sent1_list = []
    sent2_list = []
    length1_list = []
    length2_list = []
    label_list = []

    for datum in batch:
        length1_list.append(datum[1][0])
        length2_list.append(datum[1][1])
        label_list.append(datum[2])

    max_length1 = np.max(length1_list)
    max_length2 = np.max(length2_list)

    # padding
    for datum in batch:
        padded_vec1 = np.pad(np.array(datum[0][0]),
                             pad_width=(0, max_length1 - datum[1][0]),
                             mode="constant", constant_values=0)
        sent1_list.append(padded_vec1)

        padded_vec2 = np.pad(np.array(datum[0][1]),
                             pad_width=(0, max_length2 - datum[1][1]),
                             mode="constant", constant_values=0)
        sent2_list.append(padded_vec2)

    n1 = np.array(sent1_list).astype(int)
    n2 = np.array(sent2_list).astype(int)

    t_sent1 = torch.from_numpy(n1).to(DEVICE)
    t_sent2 = torch.from_numpy(n2).to(DEVICE)
    t_len1 = torch.LongTensor(length1_list).to(DEVICE)
    t_len2 = torch.LongTensor(length2_list).to(DEVICE)
    t_label = torch.LongTensor(label_list).to(DEVICE)

    # sorting by descending len1
    sorted_t_len1, idx_sort = torch.sort(t_len1, dim=0, descending=True)

    return [
        torch.index_select(t_sent1, 0, idx_sort),
        torch.index_select(t_sent2, 0, idx_sort),
        sorted_t_len1,
        torch.index_select(t_len2, 0, idx_sort),
        torch.index_select(t_label, 0, idx_sort)
    ]


class SNLIDatum:
    """
    Class that represents a train/validation datum
    """
    def __init__(self, raw_tuple, label):
        self.raw_tuple = raw_tuple
        self.tokens_tuple = None
        self.token_idx_tuple = None
        self.label = label

    def set_token_idx(self, token_idx_tuple):
        self.token_idx_tuple = token_idx_tuple

    def set_tokens(self, tokens_tuple):
        self.tokens_tuple = tokens_tuple


class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        :param data_list: list of IMDBDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx_tuple, label = self.data_list[key].token_idx_tuple, self.data_list[key].label
        len_tuple = len(token_idx_tuple[0]), len(token_idx_tuple[1])
        return [token_idx_tuple, len_tuple, label]


def construct_dataset(file_path, size=None):
    """
    Loads a file into a dataset list
    :param file_path: path to file
    :param size: number of rows to read
    :return: list of NLIDatum
    """
    data_list = []
    for i, l in enumerate(open(file_path, "r")):
        if i != 0:
            raw_parts = l.split('\t')
            parts = preprocess(raw_parts)
            raw_tuple = (parts[0], parts[1])
            label = LABEL2INDEX[parts[2]]
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


def load_pretrained_vectors(fname, rows):
    """ builds the vocabulary from the pretrained vectors """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    loaded_embeddings = np.zeros((rows + 2, d))
    token2id = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    id2token = [PAD_TOKEN, UNK_TOKEN]
    loaded_embeddings[0, :] = np.zeros(d)  # PAD
    sd_list = []

    for i, line in (enumerate(fin)):
        if i >= rows:
            break
        s = line.split()
        loaded_embeddings[i + 2, :] = np.asarray(s[1:])
        token2id[s[0]] = i + 2
        id2token.append(s[0])
        sd_list.append(loaded_embeddings[i + 2, :].std())

    avg_sd = np.mean(sd_list)
    loaded_embeddings[1, :] = np.random.normal(0, avg_sd, d)  # UNK (init at sd = group avg)
    return token2id, id2token, loaded_embeddings

