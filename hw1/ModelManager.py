"""
manager for running the model with various hyperparameters

handles loading serialized data if available
"""
import logging
import os
import torch.nn as nn
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
import config_defaults as cd
import data_processor as dp
import BagOfWords as BoW
import ngrams
import pickle as pkl
from tqdm import tnrange
from tqdm import tqdm_notebook as tqdm
cd.init_logger()


class ModelManager:
    def __init__(self, hparams=None, save_pickles=True, load_pickles=True):
        self.hparams = cd.DEFAULT_HPARAMS
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        self.save_pickles = save_pickles
        self.load_pickles = load_pickles

        # init other vars
        self.data = {}
        self.loaders = {}
        self.model = None
        self.validation_acc_history = []

        assert self.hparams['VAL_SIZE'] < cd.TRAIN_AND_VAL_SIZE

        logging.info("initialized model with hyperparametrs:")
        for key in self.hparams:
            logging.info("%s: %s" % (key, self.hparams[key]))
        logging.info("allow pickle loads: %s, allow pickle saves: %s" % (self.load_pickles, self.save_pickles))

    def load_and_train(self):
        self.load_data()
        self.data_to_pipe()
        self.model_init()
        self.train()

    def load_data(self):
        """
        does the hard work of loading and processing the data
        considers if we can load the data by comparing current hparams with hparams on the pickle files
        """
        pickle_path_train_val = cd.DIR_PICKLE + 'trainval_' + hparam_to_str(self.hparams)
        pickle_path_test = cd.DIR_PICKLE + 'test_' + hparam_to_str(self.hparams)
        pickle_path_indexer = cd.DIR_PICKLE + 'idx_' + hparam_to_str(self.hparams)

        # first check if we can just load the pickles
        if self.load_pickles and os.path.isfile(pickle_path_train_val) \
                and os.path.isfile(pickle_path_test) \
                and os.path.isfile(pickle_path_indexer):
            # do load
            logging.info("found pickle files in %s, loading them instead of rebuilding ... " % cd.DIR_PICKLE)
            train_and_val_data = pkl.load(open(pickle_path_train_val, "rb"))
            test_data = pkl.load(open(pickle_path_test, "rb"))
            train_ngram_indexer = pkl.load(open(pickle_path_indexer, "rb"))
        else:
            logging.info("did not find pickle files in %s, rebuilding ..." % cd.DIR_PICKLE)

            logging.info("loading datasets ...")
            train_and_val_set = dp.construct_dataset(cd.DIR_TRAIN, cd.TRAIN_AND_VAL_SIZE)
            test_set = dp.construct_dataset(cd.DIR_TRAIN, cd.TEST_SIZE)

            logging.info("processing training and val set of size %s..." % cd.TRAIN_AND_VAL_SIZE)
            train_and_val_data, train_ngram_indexer, ngram_counter = \
                ngrams.process_text_dataset(train_and_val_set,
                                            self.hparams['NGRAM_SIZE'],
                                            self.hparams['VOC_SIZE'],
                                            mode=self.hparams['NGRAM_MODE'],
                                            val_size=self.hparams['VAL_SIZE'])

            logging.info("processing test set of size %s ..." % cd.TEST_SIZE)
            test_data, _, _ = ngrams.process_text_dataset(test_set,
                                                          self.hparams['NGRAM_SIZE'],
                                                          ngram_indexer=train_ngram_indexer,
                                                          mode=self.hparams['NGRAM_MODE'])

            if self.save_pickles:
                logging.info("saving pickled data to folder %s ..." % cd.DIR_PICKLE)
                pkl.dump(train_and_val_data, open(pickle_path_train_val, "wb"))
                pkl.dump(test_data, open(pickle_path_test, "wb"))
                pkl.dump(train_ngram_indexer, open(pickle_path_indexer, "wb"))

        self.data['train_data'] = train_and_val_data[:len(train_and_val_data) - self.hparams['VAL_SIZE']]
        self.data['val_data'] = train_and_val_data[-self.hparams['VAL_SIZE']:]
        self.data['test_data'] = test_data
        self.data['vocab'] = train_ngram_indexer

    def data_to_pipe(self):
        """
        coverts the data objects to the torch.*.DataLoader pipes
        """
        imdb_train = dp.IMDBDataset(self.data['train_data'])
        imdb_validation = dp.IMDBDataset(self.data['val_data'])
        imdb_test = dp.IMDBDataset(self.data['test_data'])

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

    def train(self):
        """
        main training loop
        """
        if self.model is None:
            self.model_init()

        op_constr = self.hparams['OPTIMIZER']
        optimizer = op_constr(self.model.parameters(), lr=self.hparams['LR'])
        criterion = nn.CrossEntropyLoss()

        stop_training = False
        for epoch in tnrange(self.hparams['NEPOCH'], desc='Epochs'):
            for i, (data, lengths, labels) in enumerate(tqdm(self.loaders['train'])):
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
                    logging.info('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                        epoch + 1, self.hparams['NEPOCH'], i + 1, len(self.loaders['train']), val_acc))

                    self.validation_acc_history.append(val_acc)
                    # check if we need to earily stop the model
                    stop_training = early_stop(self.validation_acc_history)
                    if stop_training:
                        logging.info("--- earily stop triggered ---")
                        break
                if stop_training:
                    break

    def test_model(self, loader=None):
        if loader is None:
            loader = self.loaders['test']
        if self.model is None:
            logging.error("ERROR: cannot test model, was not initialized")
        else:
            print("acc = %s" % test_model(loader, self.model))

    def save_results(self, df):
        """ tack results onto a pandas dataframe """


def hparam_to_str(hparams):
    final_str = ''
    for key in hparams:
        if key in cd.DATA_HPARAMS:
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
    if len(val_acc_history) > t + 1 and val_acc_history[-t - 1] > max(val_acc_history[-t - 1:]) + required_progress:
        return True
    return False
