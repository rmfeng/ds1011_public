"""
manager for running the model with various hyperparameters

handles loading serialized data if available
"""
import logging
import os
import torch.nn as nn
import torch.nn.functional as tfunc
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import pandas as pd
import config_defaults as cd
import data_processor as dp
import BagOfWords as BoW
import ngrams
import pickle as pkl
import time
from tqdm import tnrange
# from tqdm import tqdm_notebook as tqdm
cd.init_logger()
logger = logging.getLogger('__main__')


class ModelManager:
    def __init__(self, hparams=None, save_pickles=True, load_pickles=True, res_name=None):
        self.hparams = cd.DEFAULT_HPARAMS.copy()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        self.save_pickles = save_pickles
        self.load_pickles = load_pickles

        # init other vars
        self.res_name = res_name
        self.data = {}
        self.loaders = {}
        self.model = None
        self.validation_acc_history = []
        self.res_df = None
        self.cur_res = _init_cur_res()
        self.cur_res.update(self.hparams)
        self.optimizer = None

        assert self.hparams['VAL_SIZE'] < cd.TRAIN_AND_VAL_SIZE

        logger.info("initialized model with hyperparametrs:")
        for key in self.hparams:
            logger.info("%s: %s" % (key, self.hparams[key]))
        logger.info("allow pickle loads: %s, allow pickle saves: %s" % (self.load_pickles, self.save_pickles))

    def train(self, epoch_override=None, reload_data=True):
        if reload_data:
            self.load_data()
            self.data_to_pipe()
        self.model_init()  # make sure we force the model to re-init
        self.training_loop(epoch_override)
        self.append_results()

    def load_data(self):
        """
        does the hard work of loading and processing the data
        considers if we can load the data by comparing current hparams with hparams on the pickle files
        """
        pickle_path_train_val = cd.DIR_PICKLE + 'trainval_' + hparam_to_str(self.hparams, cd.DATA_HPARAMS)
        pickle_path_test = cd.DIR_PICKLE + 'test_' + hparam_to_str(self.hparams, cd.DATA_HPARAMS)
        pickle_indexer = cd.DIR_PICKLE + 'idx_' + hparam_to_str(self.hparams, cd.INDEXER_HPARAMS)

        """ --- DATA LOADING AND RECALC --- """
        if self.load_pickles and os.path.isfile(pickle_path_train_val) \
                and os.path.isfile(pickle_path_test):
            # do load
            logger.info("found pickle files in %s, loading them instead of rebuilding ... " % cd.DIR_PICKLE)
            train_and_val_data = pkl.load(open(pickle_path_train_val, "rb"))
            test_data = pkl.load(open(pickle_path_test, "rb"))
        else:
            logger.info("did not find pickle files in %s, rebuilding ..." % cd.DIR_PICKLE)

            logger.info("loading datasets ...")
            train_and_val_set = dp.construct_dataset(cd.DIR_TRAIN, cd.TRAIN_AND_VAL_SIZE)
            test_set = dp.construct_dataset(cd.DIR_TRAIN, cd.TEST_SIZE)

            logger.info("extracting ngram from training and val set of size %s..." % cd.TRAIN_AND_VAL_SIZE)

            train_and_val_data = ngrams.extract_ngrams(train_and_val_set,
                                                       self.hparams['NGRAM_SIZE'],
                                                       remove_stopwords=self.hparams['REMOVE_STOP_WORDS'],
                                                       remove_punc=self.hparams['REMOVE_PUNC'],
                                                       mode=self.hparams['NGRAM_MODE'])

            logger.info("extracting ngram from test set of size %s ..." % cd.TEST_SIZE)
            test_data = ngrams.extract_ngrams(test_set,
                                              self.hparams['NGRAM_SIZE'],
                                              remove_stopwords=self.hparams['REMOVE_STOP_WORDS'],
                                              remove_punc=self.hparams['REMOVE_PUNC'],
                                              mode=self.hparams['NGRAM_MODE'])

            # only saves if we did not load the data
            if self.save_pickles:
                logger.info("saving pickled data to folder %s ..." % cd.DIR_PICKLE)
                pkl.dump(train_and_val_data, open(pickle_path_train_val, "wb"))
                pkl.dump(test_data, open(pickle_path_test, "wb"))

        """ --- INDEXER LOADING AND RECALC --- """
        if self.load_pickles and os.path.isfile(pickle_indexer):
            # load the indexer
            logger.info("found pickle files for indexer in %s, loading them ... " % cd.DIR_PICKLE)
            train_ngram_indexer = pkl.load(open(pickle_indexer, "rb"))
        else:
            # need to do process the ngrams data=
            train_ngram_indexer, _ = ngrams.create_ngram_indexer(train_and_val_data,
                                                                 topk=self.hparams['VOC_SIZE'],
                                                                 val_size=self.hparams['VAL_SIZE'])
            if self.save_pickles:
                logger.info("saving pickled indexer to folder %s ..." % cd.DIR_PICKLE)
                pkl.dump(train_ngram_indexer, open(pickle_indexer, "wb"))

        train_and_val_data = ngrams.process_dataset_ngrams(train_and_val_data, train_ngram_indexer)
        test_data = ngrams.process_dataset_ngrams(test_data, train_ngram_indexer)

        self.data['train'] = train_and_val_data[:len(train_and_val_data) - self.hparams['VAL_SIZE']]
        self.data['val'] = train_and_val_data[-self.hparams['VAL_SIZE']:]
        self.data['test'] = test_data
        self.data['vocab'] = train_ngram_indexer

    def _calc_unknown_ratio(self, set_name):
        num_unk, num_tot = 0, 0

        for cur_data in self.data[set_name]:
            for cur_idx in cur_data.token_idx:
                if cur_idx != ngrams.PAD_IDX:
                    num_tot += 1
                if cur_idx == ngrams.UNK_IDX:
                    num_unk += 1

        return num_unk / num_tot

    def data_to_pipe(self):
        """
        coverts the data objects to the torch.*.DataLoader pipes
        """
        imdb_train = dp.IMDBDataset(self.data['train'])
        imdb_validation = dp.IMDBDataset(self.data['val'])
        imdb_test = dp.IMDBDataset(self.data['test'])

        self.loaders['train'] = DataLoader(dataset=imdb_train,
                                           batch_size=self.hparams['BATCH_SIZE'],
                                           collate_fn=dp.imdb_collate_func,
                                           shuffle=True)

        self.loaders['val'] = DataLoader(dataset=imdb_validation,
                                         batch_size=self.hparams['BATCH_SIZE'],
                                         collate_fn=dp.imdb_collate_func,
                                         shuffle=False)

        self.loaders['test'] = DataLoader(dataset=imdb_test,
                                          batch_size=self.hparams['BATCH_SIZE'],
                                          collate_fn=dp.imdb_collate_func,
                                          shuffle=False)

    def model_init(self):
        """ initializes the model """
        self.model = BoW.BagOfWords(len(self.data['vocab']), self.hparams['EMBEDDING_DIM']).to(cd.DEVICE)
        self.validation_acc_history = []
        self.cur_res['initial_val_acc'] = self.test_model(loader=self.loaders['val'])

        # need to reinit the cur_res with new hparams
        self.cur_res = _init_cur_res()
        self.cur_res.update(self.hparams)

    def training_loop(self, epoch_override=None):
        """
        main training loop
        """
        start_time = time.time()
        # has the ability to initialize the model too
        if self.model is None:
            self.model_init()

        # the epoch is overridable incase we want to partial train
        if epoch_override is None:
            train_epochs = self.hparams['NEPOCH']
        else:
            train_epochs = epoch_override

        op_constr = self.hparams['OPTIMIZER']
        optimizer = op_constr(self.model.parameters(), lr=self.hparams['LR'])
        self.optimizer = optimizer
        criterion = nn.CrossEntropyLoss()

        # lr adjustments
        scheduler = ExponentialLR(optimizer, gamma=self.hparams['LR_DECAY_RATE'])

        # saving unk ratios
        self.cur_res['pct_unk_train'] = self._calc_unknown_ratio(set_name='train')
        self.cur_res['pct_unk_val'] = self._calc_unknown_ratio(set_name='val')

        stop_training = False
        total_iterated = 0

        for epoch in range(train_epochs):
            scheduler.step()
            for i, (data, lengths, labels) in enumerate(self.loaders['train']):
                total_iterated += self.hparams['BATCH_SIZE']
                self.model.train()  # good practice to set the model to training mode (dropout)

                data_batch, length_batch, label_batch = data, lengths, labels
                optimizer.zero_grad()
                outputs = self.model(data_batch, length_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()
                # validate every 4 batches
                if (i + 1) % (self.hparams['BATCH_SIZE'] * self.hparams['VAL_FREQ']) == 0:
                    val_acc = test_model(self.loaders['val'], self.model)
                    logger.info('Epoch: [%s/%s], Step: [%s/%s], Val Acc: %s, LR: %.4f' %
                                (epoch + 1,
                                 self.hparams['NEPOCH'],
                                 i + 1,
                                 len(self.loaders['train']),
                                 val_acc,
                                 optimizer.param_groups[0]['lr']))

                    self.validation_acc_history.append(val_acc)
                    # check if we need to earily stop the model
                    if self.hparams['EARLY_STOP']:
                        stop_training = early_stop(self.validation_acc_history,
                                                   self.hparams['EARLY_STOP_LOOKBACK'],
                                                   self.hparams['EARLY_STOP_MIN_IMPROVE'])
                if stop_training:
                    logger.info("--- earily stop triggered ---")
                    break
            if stop_training:
                break
            if epoch in (1, 2, 3):  # saves first 3 epochs only if we haven't broken
                dict_str = 'epoch' + str(epoch) + '_val_acc'
                val_acc = self.test_model(loader=self.loaders['val'])
                self.cur_res[dict_str] = val_acc

        # stopped training so save the final val acc
        if len(self.validation_acc_history) > 0:
            self.cur_res['final_val_acc'] = self.validation_acc_history[-1]

        # timing training and other training results
        self.cur_res['training_time'] = str(round(time.time() - start_time, 2))
        self.cur_res['total_data_iterated'] = total_iterated
        self.cur_res['early_stopped'] = str(stop_training)

    def test_model(self, loader=None):
        if loader is None:
            loader = self.loaders['test']
            logger.critical("!!!! THIS IS THE FINAL TEST RESULT, DO NOT FURTHER OPTIMIZE !!!!")
        if self.model is None:
            logger.error("ERROR: cannot test model, was not initialized")
        else:
            return test_model(loader, self.model)
        return -1

    def append_results(self):
        """ tack results onto a pandas dataframe """
        if self.res_name:
            res_path = cd.DIR_RES + self.res_name
        else:
            res_path = cd.DIR_RES + 'res_df.p'

        if self.res_df is None and os.path.isfile(res_path):
            # history found
            logger.info("found historical file, loading the dataframe at %s" % res_path)
            df = pkl.load(open(res_path, "rb"))
            df_cur = pd.DataFrame(self.cur_res, index=[df.shape[0] + 1])
            self.res_df = pd.concat([df, df_cur])
        elif self.res_df is None:
            # new res_df and not history found
            logger.info("generating new pandas dataframe to store results")
            self.res_df = pd.DataFrame(self.cur_res, index=[1])
        else:
            # res_df exists and hasn't been cleared
            logger.info("appending new results to existing dataframe")
            df = pd.DataFrame(self.cur_res, index=[len(self.res_df) + 1])
            self.res_df = pd.concat([self.res_df, df])

    def save_results(self):
        if self.res_name is None:
            res_path = cd.DIR_RES + 'res_df.p'
        else:
            res_path = cd.DIR_RES + self.res_name
        if self.res_df is None:
            logger.error("no results table to save! self.res_df is empty")
        else:
            pkl.dump(self.res_df, open(res_path, "wb"))
            logger.info("results saved to %s" % res_path)


def build_lr_func(decay):
    def lamda_mult(x):
        if x is None or x == 0.:
            return 1.0
        else:
            return decay * x
    return lamda_mult


def hparam_to_str(hparams, req_params):
    final_str = ''
    for key in sorted(hparams):
        if key in req_params:
            final_str += str(hparams[key]).replace('.', 'p').replace(':', '-') + "_"
    return final_str[:-1] + '.p'


def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()  # good practice to set the model to evaluation mode (no dropout)
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = tfunc.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return 100 * correct / total


def early_stop(val_acc_history, t=2, required_progress=0.01):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    if len(val_acc_history) >= t + 1 and val_acc_history[-t - 1] > max(val_acc_history[-t:]) - required_progress:
        return True
    return False


def _init_cur_res():
    """ initializes all results to '', does not need HPARAMS already defined in the config """
    return {
        'initial_val_acc': '',
        'epoch1_val_acc': '',
        'epoch2_val_acc': '',
        'epoch3_val_acc': '',
        'final_val_acc': '',
        'training_time': '',
        'total_data_iterated': '',
        'early_stopped': '',
        'pct_unk_train': '',
        'pct_unk_val': ''
    }


if __name__ == '__main__':
    mm = ModelManager()
    mm.load_data()
    mm.data_to_pipe()
    param_overrides = {'EARLY_STOP': False}
    mm.hparams.update(param_overrides)
    mm.train(epoch_override=2, reload_data=False)