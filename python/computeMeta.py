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

    ################ JOBS #################
    job_list = classifier_data_dict[u'm\xe9tiers']
    sents_by_char = wsent
    word2vec_model = MyModel()
    char_list = list(reversed(char_list))
    N_CHARS = 10 # Num of chars to compute scores for -> default all
    predictors = ['count', 'proximity']

    # Stores for predictor scores
    # count_full_const = {}
    # count_full_decr = {}
    # count_expo_const = {}
    # count_expo_decr = {}
    # proximity_full_const = {}
    # proximity_full_decr = {}
    # proximity_expo_const = {}
    # proximity_expo_decr = {}

    # DataFrames for plotting
    p = predictors[0]
    df_count_full_const = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=True, predictor=p)
    df_count_full_decr = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=True, full=True, predictor=p)
    df_count_expo_const = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=False, predictor=p)
    df_count_expo_decr = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=True, full=False, predictor=p)

    p = predictors[1]
    df_proximity_full_const = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=True, predictor=p)
    df_proximity_full_decr = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=True, full=True, predictor=p)
    df_proximity_expo_const = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=False, predictor=p)
    df_proximity_expo_decr = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=True, full=False, predictor=p)

    # PRINTING RESULTS
    # print('===========LABELS=============')
    # printStore(job_labels)
    # print("")
    # print('===========COUNT SCORE=============')
    # printStore(full_count_score)
    # print("")
    # print('===========PROXIMITY SCORE=============')
    # printStore(full_proximity_score)
    # print("")
    print('===========SIMILARITY SCORE=============')
    plotJobScores(df_count_full_const)

    # Computing job suggestion similarity with labels

def jobPredictor(sentences, sents_by_char, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=True, predictor='count'):
    """
    Computes predictions and scores for predictor with parameters decreasing and full
    """
    total_sents = len(sentences)
    full_score = {}
    TOP_N_JOBS = 5
    df = pd.DataFrame(columns=['Character', 'Label-Guess', 'Similarity', 'Rank'])
    job_count = defaultdict(int)

    # take neighbor sentence to the left and right to create window of sentences
    # take sentence index +-window
    window = 2

    # For each character
    for i, character in enumerate(char_list):
        char_sents = {}
        score = {}
        # For each sentence in which chaarater is mentioned
        char_sents = sents_by_char[character]
        if not full:
            # Get reduced sentence set
            # ASSUMPTION, either 10 or 10th of character mentions
            char_sents = char_sents[:int(max(10, len(char_sents)/10))]

        for i in char_sents:
            # Compute sentence window to look at
            w_range = sentenceWindow(sentences, i, window)
            sent_nostop = []
            sent_words = []
            sent_tags = []
            for s in w_range:
                sent_nostop += sentences[s]['nostop']
                sent_words += sentences[s]['words']
                sent_tags += sentences[s]['tags']
            # If character is mentioned in sentence window, add to score
            if character in sent_nostop:
                for job in job_list:
                    if unicode(job) in sent_nostop:
                        if predictor == 'count':
                            countPredict(score, decreasing, job, i, total_sents)
                        elif predictor == 'proximity':
                            proxPredict(score, decreasing, character, job, job_count, sent_words)

        if predictor == 'proximity':
            # divide by total matches to get mean proximity measure
            score = {k: float(v) / job_count[k] for k,v in score.items()}

        full_score[character] = sortNTopByVal(score, TOP_N_JOBS, descending=True)

        # contains tuples with (rank_in_list, pred)
        preds = list(enumerate(full_score[character]))

        df = get_df(df, character, preds, job_labels, word2vec_model)
    return df


def countPredict(score, decreasing, job, pos, total_sents):
    # COUNT SCORE
    # 1 per mention
    if not decreasing:
        storeIncrement(score, job, 1)
    # Decrease score increment as mentions progress
    else:
        # +1 for each mention
        # storeCount(count_score, job)
        # -log(i/n) for each mention
        proportion = float(pos+1)/ total_sents
        storeIncrement(score, job, -log(proportion))


def proxPredict(score, decreasing, character, job, job_count, sent_words):
    if not decreasing:
        dist = abs(getIdxOfWord(sent_words, job) - getIdxOfWord(sent_words, character))
        storeIncrement(score, job, dist)
        job_count[job] += 1
    # Decrease score increment as mentions progress
    else:
        pass


def get_df(df, character, preds, job_labels, word2vec_model):
    """
    Generate vector similarities and return DataFrame
    """
    if job_labels.get(character):
        for label in job_labels.get(character):
            for rank, pred in preds:
                # Add rows to Dataframe iteratively
                row_idx = df.shape[0]
                score = word2vec_model.compareWords(pred[0], label)
                df.loc[row_idx] = [character, (label, pred), score, rank]

    return df


def sentenceWindow(sentences, index, window):
    """
    :param sentences: List of dicts. Each dict is a sentence
    and contains 'nostop', 'words', 'tags'
    :param index: index with character mention in sentence
    :param window: window size
    :type: 'nostop', 'words' or 'tags'
    """
    min_idx = index-window if index-window >= 0 else 0
    max_idx = index+window if index+window < len(sentences) else len(sentences)-1
    return (min_idx, max_idx)


def printStore(store):
    for k, v in store.items():
        print(k, v)
