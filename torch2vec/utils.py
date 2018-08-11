import torch
import torch.nn.functional as F

import numpy as np

# Progress bar
from tqdm import *
import sys

# File I/O
from collections import Counter
from collections import OrderedDict
import itertools
import pickle

def make_skipgrams(sentence, window_length):
    """
    For returning skip grams out of a sequence.

    Args:
    list(sentence) or tuple(sentence): a sentence made of strings or ints or of any kind
    int(window): the window size from which the skipgrams will be built. (Number of words taken forward and backward)

    Returns:
    list(skipgrams)
    """

    skipgrams = []

    for i in range(0, len(sentence)):
        window = 1
        w_i = sentence[i]
        while window is not window_length+1:
            if (i-window) >= 0:
                w_c_front = sentence[i-window]
                skipgrams.append([w_i, w_c_front])
            if (i+window) < len(sentence):
                w_c_back = sentence[i+window]
                skipgrams.append([w_i, w_c_back])
            window += 1

    return skipgrams

def read_data(txt_path, window, mincount, line_limit, subsampling=True):
    """
    Corpus is assumed to be the 1bwc.txt one.

    Args:
    str(txt_path): path to the corpus
    int(window): the window size. Window size is: number of surrounding words + 1 - e.g. if the 2 surrounding words are wanted
    then the window size is 3. Should not be an even number.
    int(mincount): the minimum number of occurences for the considered word.
    int(line_limit): the number of lines to be read from the file.

    Returns:
    list(skip_grams): list of skipgrams built from the text.
    dict(intw, wint): dicts from int to word and from word to int.
    Counter(occurence_dict): words (in integer format) and their occurences.
    set(excluded_words): set of words that were excluded due to an occurence below the mincount threshold.
    int(total_word_count): total words counted, excluding words that did not meet the min count required.
    int(total_word_count_no_excluded_words): same as above, but including words that did not meet the min count.
    """
    occurence_dict = Counter()
    intw, wint = {}, {}
    excluded_words = {}

    # skipgrams will store sequences of integers.
    skip_grams = []
    i, total_word_count = 0, 0 # word indexer and word counter
    line_count = 0


    with open(txt_path, "r") as f:
        for l in tqdm(f, desc="Reading file", unit=" lines"):
            splitted_sentence = l.lower().replace("\n", "").split(" ")

            # Get the sentence into a integer sequence and update whether it is in excluded words
            integer_sequence = []
            for w in splitted_sentence:

                # update the total word count
                total_word_count += 1

                try:
                    # make word to integer conversion
                    integer_sequence.append(wint[w])
                except:
                    # Updating both dictionaries if word (key) is not present.
                    wint[w] = i
                    intw[i] = w
                    # Appending the integer sequence
                    integer_sequence.append(i)
                    # Update the word indexer
                    i += 1

                # Updating the list of excluded words (indexing sets or dict is an O(1) operation)
                if occurence_dict[wint[w]] < mincount:
                    excluded_words[wint[w]] = 1
                else:
                    try:
                        del excluded_words[wint[w]]
                    except KeyError:
                        pass

            # Update the occurence dict
            occurence_dict = occurence_dict + Counter(integer_sequence)

            if subsampling:pass # Subsampling here?

            # Get the skipgrams out of the integer sequence
            skip_grams.append(make_skipgrams(integer_sequence, window))

            if line_count == line_limit:
                break
            line_count += 1

    total_word_count_no_excluded_words = len(occurence_dict)

    # Get rid of the rarely occurring words, set should be rather small if count is not to high (usually 5)
    for i in excluded_words.keys(): # where i is an int representing a word.
        # update the total word count along with both the word to int and int to word dicts.
        total_word_count -= occurence_dict[i]
        del wint[intw[i]]
        del intw[i]
        del occurence_dict[i]

    return skip_grams, intw, wint, occurence_dict, excluded_words, total_word_count, total_word_count_no_excluded_words


def build_unigram_table(occ_dict, table_size):
    """
    Function for building the unigram table that will be used for negative sampling procedure.
    Use the same procedure as in the C implementation. Build a big table which we populate with the
    word index. Those index appear with a frequency weighted by their unigram probability.

    Args:
    Counter(occ_dict): occurence dict storing word frequencies.
    int(table_size): Size of the unigram table, should be large enough for proper sampling.

    Returns:
    list(unigram_table): list that will be used for negative sampling.
    """
    unigram_table = list()

    Z = sum(v**(3/4) for v in occ_dict.values())
    for k, v in occ_dict.items():
        p_wi = (v**(3/4))/Z
        unigram_table += [k]*int(p_wi*table_size)
    return unigram_table

def add_negative_samples(skipgram_data, unigrams_table, neg_examples_size=5):
    """
    Will modify *in place* the skipgram data and return a flattened list.

    Args:
    list(list(skipgram_data)): list of lists with skipgrams
    list(unigrams_table): table of unigrams from which negative examples will be sampled
    int(neg_examples_size): number of negative examples that will be added to the training instances

    Returns:
    list(sg_neg_examples): list containing the training data with skipgrams and their negative samples.
    """
    sg_neg_examples = []
    total_data = len(skipgram_data)
    for i, sg in tqdm(enumerate(skipgram_data), desc="Processing neg. samples ({} in total)".format((total_data-1)),
                      unit= " neg. samples"):
        for gram in sg:
            gram += negative_sampling(word_input=gram[0], target=gram[1],
                                      unigrams_table=unigrams_table, neg_examples_size=neg_examples_size)
            sg_neg_examples.append(gram)
    return sg_neg_examples

# Add negative samples to the end of each skip gram.
def negative_sampling(word_input, target, unigrams_table, neg_examples_size=5):
    """
    For extending the skipgrams with negatively sampled training examples. Words rarelyint here, but they
    could be of type str with the proper unigrams table.

    Args:
    int(word_input): the word for which we want to sample a negative example.
    int(target): the target of the word. Needed to avoid negative examples to be the same as target
    int(neg_examples_size): number of negative examples that will be added to the training instances

    Returns:
    list(negative_examples): list of negative examples along with the word input and target.
    """
    negative_examples = []
    while len(negative_examples) is not neg_examples_size:
        neg_sample = np.random.choice(unigrams_table)
        # Make sure that the negative example is not the same as the training or as the target.
        # This will block if there only is one value within the unigram table
        if (neg_sample != word_input) and (neg_sample != target):negative_examples.append(neg_sample)
        else:pass
    return negative_examples

def nearest_words(embedding, voc_size, word, wint, intw, n_words=10):
    """
    Small function for testing after training has been completed.

    Args:
    torch.nn.Embedding(embedding): usually the first weight matrix dim(V x D) of the single layer network
    int(voc_size): the V dimension of the matrix
    str(word): the word to recover the nearest words
    dict(wint) & dict(intw): the w 2 int and int 2 w dicts
    int(n_words): recover the n closest words of the current word

    Returns:
    tuple(sim, word): a list of n tuples consisting of the similarity score and the word.
    """
    similar_words = {}
    word_embed = embedding(torch.LongTensor([wint[word]]))
    for i in range(voc_size):
        emb = embedding(torch.LongTensor([i]))
        cos_sim = F.cosine_similarity(emb, word_embed)
        if len(similar_words) < n_words:
            similar_words[float(cos_sim)] = intw[i]
        else:
            if cos_sim > min(similar_words):
                min_key = min(similar_words)
                del similar_words[min_key]
                similar_words[float(cos_sim)] = intw[i]
            else:
                pass
    # Ordering dict based on the value of the cosine similarity
    return sorted(similar_words.items())[::-1]

def save_dict(path, dict_obj):
    with open(path, "wb") as f:
        pickle.dump(dict_obj, f)
    return None

def load_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)
