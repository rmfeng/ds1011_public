"""
# reference https://github.com/nyu-mll/DS-GA-1011-Fall2017
"""
import os
from tqdm import tqdm_notebook as tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import config_defaults as cd


class IMDBDatum:
    """
    Class that represents a train/validation/test datum
    - self.raw_text
    - self.label: 0 neg, 1 pos
    - self.file_name: dir for this datum
    - self.tokens: list of tokens
    - self.token_idx: index of each token in the text
    """

    def __init__(self, raw_text, label, file_name):
        self.raw_text = raw_text
        self.label = label
        self.file_name = file_name
        self.ngram = None
        self.token_idx = None
        self.tokens = None

    def set_ngram(self, ngram_ctr):
        self.ngram = ngram_ctr

    def set_token_idx(self, token_idx):
        self.token_idx = token_idx

    def set_tokens(self, tokens):
        self.tokens = tokens


class IMDBDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of IMDBDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx, label = self.data_list[key].token_idx, self.data_list[key].label
        return [token_idx, len(token_idx), label]


def preprocess_text(text):
    """
    Function that cleans the string
    """
    text = text.lower().replace("<br />", "")
    return text


def read_file_as_datum(file_name, label):
    """
    Function that reads a file
    """
    with open(file_name, "r") as f:
        content = f.read()
        content = preprocess_text(content)
    return IMDBDatum(raw_text=content, label=label, file_name=file_name)


def construct_dataset(dataset_dir, dataset_size, offset=0):
    """
    Function that loads a dataset
    @param offset: skip first offset items in this dir
    @param dataset_dir:
    @param dataset_size:
    """
    pos_dir = os.path.join(dataset_dir, "pos")
    neg_dir = os.path.join(dataset_dir, "neg")
    single_label_size = int(dataset_size / 2)
    output = []
    all_pos = os.listdir(pos_dir)
    all_neg = os.listdir(neg_dir)
    for i in tqdm(range(offset, offset + single_label_size)):
        output.append(read_file_as_datum(os.path.join(pos_dir, all_pos[i]), 1))
        output.append(read_file_as_datum(os.path.join(neg_dir, all_neg[i]), 0))
    return output


def imdb_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    max_length = np.max(length_list)
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]),
                            pad_width=(0, max_length - datum[1]),
                            mode="constant", constant_values=0)
        data_list.append(padded_vec)

    return [torch.from_numpy(np.array(data_list)).to(cd.DEVICE),
            torch.LongTensor(length_list).to(cd.DEVICE),
            torch.LongTensor(label_list).to(cd.DEVICE)]

