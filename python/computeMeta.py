# coding: utf8
from read_data import readData
from tools import *
from metaPlot import *
from math import log
from collections import defaultdict
from word_similarity import MyModel
import pandas as pd


def runMeta(sentences, wsent, char_list, job_labels):
    """
    Compute various metadata about characters in char_list
    :param sentences: list(dict)
        List of dicts. Each dict is a sentence
        and contains 'nostop', 'words', 'tags'
    :param wsetn: dictionary of sentences by character
    :param char_list: list(unicode)
        List of character names in unicode
        Compound names are concatenated as in sentences
    :param job_labels: dict of character -> [job label]
    """
    ### Define parameters

    # classifier_data_dict has keys [u'tromper', u'nutrition', u'\xe9motions', u'dormir', u'raison', u'\xe9tats', u'vouloir', u'tuer', u'gu\xe9rir', u'relations', u'm\xe9tiers', u'salutations', u'soupir', u'pens\xe9e', u'parole', u'foi']
    classifier_data_dict = readData()
    job_list = classifier_data_dict[u'm\xe9tiers']
    sents_by_char = wsent
    word2vec_model = MyModel()
    TOP_N_JOBS = 5
    char_list = list(reversed(char_list))
    N_CHARS = 10 # Num of chars to compute scores for -> default all
    char_threshold = len(char_list) / 3 ### Assumption 1/3 of chars are main chars

    # Stores for predictor scores
    count_full_const = {}
    count_full_decr = {}
    count_expo_const = {}
    count_expo_decr = {}
    proximity_full_const = {}
    proximity_full_decr = {}
    proximity_expo_const = {}
    proximity_expo_decr = {}

    # DataFrames for plotting
    cols = ['Character', 'Label-Guess', 'Similarity', 'Rank']
    df_count_full_const = pd.DataFrame(columns=cols)
    df_count_full_decr = pd.DataFrame(columns=cols)
    df_count_expo_const = pd.DataFrame(columns=cols)
    df_count_expo_decr = pd.DataFrame(columns=cols)
    df_proximity_full_const = pd.DataFrame(columns=cols)
    df_proximity_full_decr = pd.DataFrame(columns=cols)
    df_proximity_expo_const = pd.DataFrame(columns=cols)
    df_proximity_expo_decr = pd.DataFrame(columns=cols)

    

    # OLD Stores
    # TODO remove these after changing rest
    full_count_score = {}
    full_proximity_score = {}
    # Dataframes
    similarity_scores = pd.DataFrame(columns=['Character', 'Label-Guess', 'Similarity', 'Predictor', 'Rank'])


    for i, character in enumerate(char_list):
        # scores per character
        char_is_main = True if i < char_threshold else False
        count_score, proximity_score = jobPredictor(sentences, sents_by_char[character], character, char_is_main, job_list)

        full_count_score[character] = sortNTopByVal(count_score, TOP_N_JOBS, True)
        full_proximity_score[character] = sortNTopByVal(proximity_score, TOP_N_JOBS)

        # Choose best predictions for meta benchmark
        # top_preds is concatenation of the different scores - for plotting purposes
        # contains tuples with ((rank_in_list, pred), predictor)
        top_preds = zip(list(enumerate(full_count_score[character])), ['Count'] * len(full_count_score[character])) + zip(list(enumerate(full_proximity_score[character])), ['Proximity'] * len(full_count_score[character]))

        # Generate vector similarities
        if job_labels.get(character):
            for label in job_labels.get(character):
                for rank_pred, method in top_preds:
                    # Add rows to Dataframe iteratively
                    row_idx = similarity_scores.shape[0]
                    pred = rank_pred[1]
                    rank = rank_pred[0]
                    score = word2vec_model.compareWords(pred[0], label)
                    similarity_scores.loc[row_idx] = [character, (label, pred), score, method, rank]

    # PRINTING RESULTS
    print('===========LABELS=============')
    printStore(job_labels)
    print("")
    print('===========COUNT SCORE=============')
    printStore(full_count_score)
    print("")
    print('===========PROXIMITY SCORE=============')
    printStore(full_proximity_score)
    print("")
    print('===========SIMILARITY SCORE=============')
    plotJobScores(similarity_scores)

    # Computing job suggestion similarity with labels


def jobPredictor(sentences, char_sents, char_name, job_list, char_is_main, job_labels=defaultdict(int), decreasing_score=True):
    """
    Find potential jobs for candidate in char_name.
    Return list[(jobname, score)]
    """
    count_score = {}
    proximity_score = {}

    total_sents = len(sentences)
    total_char_sents = len(char_sents)

    # take neighbor sentence to the left and right to create window of sentences
    # take sentence index +-window
    window = 2
    n = len(sentences)
    job_count = defaultdict(int)

    # Compute score for all character sents
    for i in char_sents:
        w_range = sentenceWindow(sentences, i, window)
        sent_nostop = []
        sent_words = []
        sent_tags = []
        for s in w_range:
            sent_nostop += sentences[s]['nostop']
            sent_words += sentences[s]['words']
            sent_tags += sentences[s]['tags']
        if char_name in sent_nostop:
            for job in job_list:
                if unicode(job) in sent_nostop:
                    ##### count score
                    # +1 for each mention
                    # storeCount(count_score, job)
                    # -log(i/n) for each mention
                    proportion = float(i+1)/ n
                    storeIncrement(count_score, job, -log(proportion))

                    ##### mean proximity score
                    dist = abs(getIdxOfWord(sent_words, job) - getIdxOfWord(sent_words, char_name))
                    storeIncrement(proximity_score, job, dist)
                    job_count[job] += 1
            # divide by total matches to get mean proximity measure
            if job_count[job]:
                proximity_score[job] = float(proximity_score[job]) / float(job_count[job])
    proximity_score = {k: float(v) / job_count[k] for k,v in proximity_score.items()}

    # Get reduced sentence set -> assumption, either 10 or 10th of character mentions
    reduced_char_sents = char_sents[:max(10, len(char_sents)/10)]



    return count_score, proximity_score


def sentenceWindow(sentences, index, window):
    """
    :param sentences: List of dicts. Each dict is a sentence
    and contains 'nostop', 'words', 'tags'
    :param index: index with character mention in sentence
    :param window: window size
    :type: 'nostop', 'words' or 'tags'
    """
    min_idx = index-window if index-window >= 0 else 0
    max_idx = index+windwo if index+window < len(sentences) else len(sentences)-1
    w_range = (min_idx, max_idx)
    w_range


def printStore(store):
    for k, v in store.items():
        print(k, v)



# visualize with t-SNE http://homepage.tudelft.nl/19j49/t-SNE.html?
