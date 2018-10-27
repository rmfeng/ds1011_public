"""
RNN Model for encoding the sentence pairs in snli
"""
import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    NLI classification model - RNN
    """
    def __init__(self):
        super(RNN, self).__init__()
        pass

    def forward(self, sent1, sent2, len1, len2):
        # the loader should have already sorted the inputs in descending len1
        _, idx1_sort = torch.sort(len1, dim=0, descending=True)
        _, idx2_sort = torch.sort(len2, dim=0, descending=True)
