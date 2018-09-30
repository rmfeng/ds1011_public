import spacy
import string
from collections import Counter
from sklearn.feature_extraction import stop_words
from tqdm import tqdm_notebook as tqdm

tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation

PAD_TOKEN, PAD_IDX = '<pad>', 0
UNK_TOKEN, UNK_IDX = '<unk>', 1


def tokenize(sent, remove_stopwords=True, remove_punc=True):
    """
    basic tokenizer method from spacy
    :param sent: input sentence
    :param remove_stopwords: whether to remove stopwords
    :param remove_punc: whether to remove punctuation
    :return: list of tokens
    """
    tokens = tokenizer(sent)
    if remove_punc:
        tokens = [token.text.lower() for token in tokens if (token.text not in punctuations)]

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words.ENGLISH_STOP_WORDS]

    return tokens


def extract_ngram_from_text(text, n, remove_stopwords=True, remove_punc=True):
    """
    Function that retrieves all n-grams from the input string
    @param text: raw string
    @param n: integer that tells the model to retrieve all k-gram where k<=n
    @param remove_stopwords: whether or not to remove stopwords from lib
    @param remove_punc: whether or not to remove punctuation from lib
    @return ngram_counter: a counter that maps n-gram to its frequency
    @return tokens: a list of parsed ngrams
    """
    # tokenize words - for simplicity just split by space
    tokens = tokenize(text, remove_stopwords=remove_stopwords, remove_punc=remove_punc)
    # print("all tokens", str(tokens))

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
    @param ngram_counter_list: list of counters
    @param topk, int: # of words to keep in the vocabulary - not counting pad/unk
    @return ngram2idx: a dictionary that maps ngram to an unique index
    """
    rt_dict = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    i = 2  # the index to start the rest of the tokens
    final_count = Counter()
    for ngc in ngram_counter_list:
        final_count += ngc

    for key in dict(final_count.most_common(topk)):
        rt_dict[key] = i
        i += 1

    return rt_dict  # length topk + 2


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
    @param tokens: list of ngram
    @param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    return [ngram_indexer[token] if token in ngram_indexer else UNK_IDX for token in tokens]


def process_text_dataset(dataset, n, topk=None, ngram_indexer=None):
    """
    Top level function that encodes each datum into a list of ngram indices
    @param dataset: list of IMDBDatum
    @param n: n in "n-gram"
    @param topk: #
    @param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    # extract n-gram
    for i in tqdm(range(len(dataset))):
        text_datum = dataset[i].raw_text
        ngrams, tokens = extract_ngram_from_text(text_datum, n)
        dataset[i].set_ngram(ngrams)
        dataset[i].set_tokens(tokens)
    # select top k ngram
    if ngram_indexer is None:
        ngram_indexer = construct_ngram_indexer([datum.ngram for datum in dataset], topk)
    # vectorize each datum
    for i in range(len(dataset)):
        dataset[i].set_token_idx(token_to_index(dataset[i].tokens, ngram_indexer))
    return dataset, ngram_indexer


if __name__ == "__main__":
    """ self testing """
    print("testing ngrams module")

    my_sent = r'this apple is as red as they can be, said Sally. Then Sally left and became a silly Sally.'
    print(extract_ngram_from_text(my_sent
                                  , 2
                                  , remove_stopwords=True
                                  , remove_punc=True), "\n")

    print(extract_ngram_from_text(my_sent
                                  , 2
                                  , remove_stopwords=False
                                  , remove_punc=True), "\n")
    print(extract_ngram_from_text(my_sent
                                  , 3
                                  , remove_stopwords=False
                                  , remove_punc=False), "\n")


