"""
responsible for tokenizing and building dataset pipe
"""
import spacy
import string
from collections import Counter
from sklearn.feature_extraction import stop_words
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange
import logging

tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation

PAD_TOKEN, PAD_IDX = '<pad>', 0
UNK_TOKEN, UNK_IDX = '<unk>', 1


def tokenize(sent, remove_stopwords=True, remove_punc=True, mode='spacy'):
    """
    basic tokenizer method from spacy
    :param sent: input sentence
    :param remove_stopwords: whether to remove stopwords
    :param remove_punc: whether to remove punctuation
    :param mode: {'spacy', 'naive'}
    :return: list of tokens
    """
    if mode == 'spacy':
        tokens = tokenizer(sent)
        tokens = [token.text for token in tokens]
    elif mode == 'naive':
        tokens = sent.split(" ")
    else:
        tokens = []

    if remove_stopwords:  # only removed if small cap
        tokens = [token for token in tokens if token not in stop_words.ENGLISH_STOP_WORDS]

    if remove_punc:
        tokens = [token.lower() for token in tokens if (token not in punctuations)]
    else:
        tokens = [token.lower() for token in tokens]

    # returns lower case, scrubbed tokens
    return tokens


def extract_ngram_from_text(text, n, remove_stopwords=True, remove_punc=True, mode='spacy'):
    """
    Function that retrieves all n-grams from the input string
    :param text: raw string
    :param n: integer that tells the model to retrieve all k-gram where k<=n
    :param remove_stopwords: whether or not to remove stopwords from lib
    :param remove_punc: whether or not to remove punctuation from lib
    :param mode: {'spacy', 'naive'}
    :return ngram_counter: a counter that maps n-gram to its frequency
    :return tokens: a list of parsed ngrams
    """
    tokens = tokenize(text, remove_stopwords=remove_stopwords, remove_punc=remove_punc, mode=mode)
    all_ngrams = []
    for i in range(0, len(tokens) - n):
        for j in range(1, n + 1):
            all_ngrams.append(get_n_gram_at_position_i(j, i, tokens))
    ngram_counter = Counter(all_ngrams)
    return ngram_counter, all_ngrams


def construct_ngram_indexer(ngram_counter_list, topk):
    """
    Function that selects the most common topk ngrams
    index 0 reserved for <pad>
    index 1 reserved for <unk>
    :param ngram_counter_list: list of counters
    :param topk: int, # of words to keep in the vocabulary - not counting pad/unk
    :return ngram2idx: a dictionary that maps ngram to an unique index
    """
    rt_dict = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    i = 2  # the index to start the rest of the tokens
    final_count = Counter()

    for elem in tqdm(ngram_counter_list):
        for key, value in elem.items():
            final_count[key] += value

    for key in tqdm(dict(final_count.most_common(topk))):
        rt_dict[key] = i
        i += 1

    return rt_dict, final_count  # length topk + 2


def get_n_gram_at_position_i(n, i, tokens):
    """
    provided a list of tokens, gets the ngram starting at position i (0 indexed)
    :param n: ngram size
    :param i: ith position
    :param tokens: full list of tokens
    :return: tuple representing ngram
    """
    out_list = []
    if n == 1:
        return tokens[i]
    else:
        for j in range(i, i + n):
            out_list.append(tokens[j])
    return tuple(out_list)


def token_to_index(tokens, ngram_indexer):
    """
    Function that transform a list of tokens to a list of token index.
    index 0 reserved for <pad>
    index 1 reserved for <unk>
    :param tokens: list of ngram
    :param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    return [ngram_indexer[token] if token in ngram_indexer else UNK_IDX for token in tokens]


def process_text_dataset(dataset,
                         n,
                         topk=None,
                         ngram_indexer=None,
                         remove_stopwords=True,
                         remove_punc=True,
                         mode='spacy',
                         val_size=0):
    """
    Top level function that encodes each datum into a list of ngram indices
    :param dataset: list of IMDBDatum
    :param n: n in "n-gram"
    :param topk: #
    :param ngram_indexer: a dictionary that maps ngram to an unique index
    :param remove_stopwords: bool to remove stopwords in the tokenizer
    :param remove_punc: bool to remove punctuation in the tokenizer
    :param mode: {'spacy', 'naive'}
    :param val_size: size of validation so the indexer only gets created on non-validation data
    """
    ngram_counter = None
    # extract n-gram
    logging.info("extracting ngrams ...")
    for i in tnrange(len(dataset), desc='extract ngrams'):
        text_datum = dataset[i].raw_text
        ngrams, tokens = extract_ngram_from_text(text_datum, n, remove_stopwords, remove_punc, mode)
        dataset[i].set_ngram(ngrams)
        dataset[i].set_tokens(tokens)
    # select top k ngram
    if ngram_indexer is None:
        logging.info("constructing ngram_indexer ...")
        logging.info("indexer length %s" % len([datum.ngram for datum in dataset][:-val_size]))
        ngram_indexer, ngram_counter = construct_ngram_indexer([datum.ngram for datum in dataset][:-val_size], topk)
    else:
        logging.info("already have a passed ngram_indexer ...")
    # vectorize each datum
    logging.info("setting each dataset's token indexes")
    for i in tnrange(len(dataset), desc='token to index'):
        dataset[i].set_token_idx(token_to_index(dataset[i].tokens, ngram_indexer))
    return dataset, ngram_indexer, ngram_counter


if __name__ == "__main__":
    """ self testing """
    print("testing ngrams module")

    my_sent = r'this apple is as red as they can be, said Sally. Then Sally left and became a silly Sally.'
