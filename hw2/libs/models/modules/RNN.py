"""
RNN Model for encoding the sentence pairs in snli
"""
import torch
import torch.nn as nn
import logging
from libs.data_loaders.SNLILoader import PAD_IDX
from config.basic_conf import DEVICE

logger = logging.getLogger('__main__')


class RNN(nn.Module):
    """
    NLI classification model - RNN
    """
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 rnn_hidden_size,
                 rnn_num_layers,
                 dropout_rnn,
                 dropout_fc,
                 fc_hidden_size,
                 num_classes,
                 pretrained_vecs):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes
        self.dropout_fc = dropout_fc
        if self.rnn_num_layers > 1:
            self.dropout_rnn = dropout_rnn
        else:
            self.dropout_rnn = 0.0

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.embed.weight = nn.Parameter(pretrained_vecs)  # using pretrained
        self.embed.weight.requires_grad = False  # freeze params
        # self.embed = nn.Embedding.from_pretrained(pretrained_vecs, freeze=True)

        self.rnn = nn.GRU(emb_dim,
                          rnn_hidden_size,
                          rnn_num_layers,
                          batch_first=True,
                          dropout=self.dropout_rnn,
                          bidirectional=True).to(DEVICE)

        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * rnn_hidden_size * rnn_num_layers, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_hidden_size, num_classes)
        ).to(DEVICE)

        self.concated_hidden = None

    def _init_rnn_hidden(self, batch_size):
        return torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size).to(DEVICE)

    def forward(self, sent1, sent2, len1, len2):
        # the loader should have already sorted the inputs in descending len1
        _, idx2_sort = torch.sort(len2, dim=0, descending=True)
        _, idx1_sort = torch.sort(idx2_sort, dim=0)

        batch_size, sent1_len = sent1.size()
        _, sent2_len = sent2.size()

        # feeding the first sent through the GRU
        embed1 = self.embed(sent1)
        embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, len1, batch_first=True)
        self.rnn.flatten_parameters()
        hidden_vec1 = self._init_rnn_hidden(batch_size)
        rnn_out1, hidden_vec1 = self.rnn(embed1, hidden_vec1)  # RNN handles hidden state init

        # feeding the 2nd sent through the GRU
        # first sorting
        sorted_sent2 = torch.index_select(sent2, 0, idx2_sort)
        sorted_len2 = torch.index_select(len2, 0, idx2_sort)

        # fpass
        embed2 = self.embed(sorted_sent2)
        embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, sorted_len2, batch_first=True)
        self.rnn.flatten_parameters()

        hidden_vec2 = self._init_rnn_hidden(batch_size)
        rnn_out2, hidden_vec2 = self.rnn(embed2, hidden_vec2)  # RNN handles hidden state init

        # unsort
        hidden_vec2 = torch.index_select(hidden_vec2, 1, idx1_sort)

        # cat
        h1_tsr_tuple = tuple(hidden_vec1[x] for x in range(hidden_vec1.shape[0]))
        h2_tsr_tuple = tuple(hidden_vec2[x] for x in range(hidden_vec2.shape[0]))
        cat_h1_tsr = torch.cat(h1_tsr_tuple, 1)
        cat_h2_tsr = torch.cat(h2_tsr_tuple, 1)

        self.concated_hidden = torch.cat((cat_h1_tsr, cat_h2_tsr), 1)

        # fpass through fc layers
        fc_out = self.fc(self.concated_hidden)

        # returns logits
        return fc_out




