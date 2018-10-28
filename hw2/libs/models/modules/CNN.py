"""
CNN Model for encoding the sentence pairs in snli
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from libs.data_loaders.SNLILoader import PAD_IDX
from config.basic_conf import DEVICE

logger = logging.getLogger('__main__')


class CNN(nn.Module):
    """
    NLI classification model - CNN
    """

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hidden_size,
                 fc_hidden_size,
                 dropout_fc,
                 num_classes,
                 pretrained_vecs
                 ):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.embed.weight = nn.Parameter(pretrained_vecs)  # using pretrained
        self.embed.weight.requires_grad = False  # freeze params

        self.conv1 = nn.Conv1d(emb_dim, hidden_size, kernel_size=3, padding=1).to(DEVICE)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1).to(DEVICE)

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_hidden_size, num_classes)
        ).to(DEVICE)

    def forward(self,  sent1, sent2, len1, len2):
        # sentence 1
        batch1_size, seq1_len = sent1.size()
        embed1 = self.embed(sent1)
        hidden1 = self.conv1(embed1.transpose(1, 2)).transpose(1, 2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch1_size, seq1_len, hidden1.size(-1))
        hidden1 = self.conv2(hidden1.transpose(1, 2)).transpose(1, 2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch1_size, seq1_len, hidden1.size(-1))
        hidden1 = torch.sum(hidden1, dim=1)

        # sentence 2
        batch2_size, seq2_len = sent2.size()
        embed2 = self.embed(sent2)
        hidden2 = self.conv1(embed2.transpose(1, 2)).transpose(1, 2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch2_size, seq2_len, hidden2.size(-1))
        hidden2 = self.conv2(hidden2.transpose(1, 2)).transpose(1, 2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch2_size, seq2_len, hidden2.size(-1))
        hidden2 = torch.sum(hidden2, dim=1)

        concated_hidden = torch.cat((hidden1, hidden2), 1)

        # fpass through fc layers
        fc_out = self.fc(concated_hidden)

        # returns logits
        return fc_out
