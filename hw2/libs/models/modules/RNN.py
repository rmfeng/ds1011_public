"""
RNN Model for encoding the sentence pairs in snli
"""
import torch
import torch.nn as nn
from libs.data_loaders.SNLILoader import PAD_IDX
import logging

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
                 dropout,
                 fc_hidden_size,
                 num_classes,
                 pretrained_vecs):
        super(RNN, self).__init__()
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        # self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        # self.embed.weight = nn.Parameter(pretrained_vecs)  # using pretrained
        # self.embed.weight.requires_grad = False  # freeze params
        self.embed = nn.Embedding.from_pretrained(pretrained_vecs, freeze=True)

        self.rnn = nn.GRU(emb_dim,
                          rnn_hidden_size,
                          rnn_num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_classes)
        )

        self.hidden_vec1 = None
        self.hidden_vec2 = None

    def _init_rnn_hidden(self, batch_size):
        return torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size)

    def forward(self, sent1, sent2, len1, len2):
        # the loader should have already sorted the inputs in descending len1
        _, idx2_sort = torch.sort(len2, dim=0, descending=True)
        _, idx1_sort = torch.sort(idx2_sort, dim=0)

        batch_size, sent1_len = sent1.size()
        _, sent2_len = sent2.size()

        # init hidden states for both GRUs
        # self.hidden_vec1 = self._init_rnn_hidden(batch_size)
        # self.hidden_vec2 = self._init_rnn_hidden(batch_size)
        self.hidden_vec1 = None
        self.hidden_vec2 = None

        # feeding the first sent through the GRU
        embed1 = self.embed(sent1)
        embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, len1.cpu().numpy(), batch_first=True)
        self.rnn.flatten_parameters()

        logger.info("embed1 type: %s" % type(embed1))
        logger.info("hidden1 type: %s" % type(self.hidden_vec1))

        rnn_out1, self.hidden_vec1 = self.rnn(embed1, self.hidden_vec1)
        rnn_out1, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out1, batch_first=True)

        # feeding the 2nd sent through the GRU
        # first sorting
        sorted_sent2 = torch.index_select(sent2, 0, idx2_sort)
        sorted_len2 = torch.index_select(len2, 0, idx2_sort)

        # fpass
        embed2 = self.embed(sorted_sent2)
        embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, sorted_len2.cpu().numpy(), batch_first=True)
        rnn_out2, self.hidden_vec2 = self.rnn(embed2, self.hidden_vec2)
        rnn_out2, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out2, batch_first=True)

        # unsort
        rnn_out2 = torch.index_select(rnn_out2, 0, idx1_sort)

        # cat
        concat = torch.cat((rnn_out1, rnn_out2), 1)

        # fpass through fc layers
        fc_out = self.fc(concat)

        # returns logits
        return fc_out




