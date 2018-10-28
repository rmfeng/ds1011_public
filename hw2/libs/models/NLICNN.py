"""
NLI Model using RNNs
"""
from libs.models.NLIRNN import NLIRNN
from config.constants import HyperParamKey, LoaderParamKey
from config.basic_conf import DEVICE
import torch
from libs.models.modules.CNN import CNN
import logging

logger = logging.getLogger('__main__')


class NLICNN(NLIRNN):
    """
    NLI with RNN encoder
    """
    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super().__init__(hparams, lparams, cparams, label, nolog)
        np_pretrained = lparams[LoaderParamKey.PRETRAINED_VECS]
        t_pretrained = torch.from_numpy(np_pretrained).type(torch.FloatTensor).to(DEVICE)

        self.model = CNN(vocab_size=lparams[LoaderParamKey.ACT_VOCAB_SIZE],
                         emb_dim=lparams[LoaderParamKey.EMBEDDING_DIM],
                         hidden_size=hparams[HyperParamKey.CNN_HIDDEN_SIZE],
                         fc_hidden_size=hparams[HyperParamKey.FC_HIDDEN_SIZE],
                         dropout_fc=hparams[HyperParamKey.DROPOUT_FC],
                         num_classes=lparams[LoaderParamKey.NUM_CLASSES],
                         pretrained_vecs=t_pretrained)
