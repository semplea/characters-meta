# coding: utf-8
from read_data import readData
from tools import *
from math import log
from collections import defaultdict
from word_similarity import MyModel
import pandas as pd


def runMeta(sentences, char_list, job_labels):
    """
    Compute various metadata about characters in char_list
    :param sentences: list(dict)
        List of dicts. Each dict is a sentence
        and contains 'nostop', 'words', 'tags'
    :param char_list: list(unicode)jo
        List of character names in unicode
        Compound names are concatenated as in sentences
    :param job_labels: dict of character -> [job label]
    """
    TOP_N_JOBS = 2
    char_list = list(reversed(char_list))
    classifier_data_dict = readData()
    job_list = classifier_data_dict['metiers']
    full_count_score = {}
    full_proximity_score = {}
    sents_by_char = buildSentsByChar(char_list, sentences)
    word2vec_model = MyModel()
    N_CHARS = 10
    similarity_scores = pd.DataFrame(columns=['Character', 'Label-Guess', 'Similarity', 'Predictor'])

    for character in char_list[:N_CHARS]:
        # scores per character
        count_score, proximity_score = jobPredictor(sentences, sents_by_char[character], character, job_list)

        full_count_score[character] = sortNTopByVal(count_score, TOP_N_JOBS, True)
        full_proximity_score[character] = sortNTopByVal(proximity_score, TOP_N_JOBS)

        # Choose best predictions for meta benchmark
        top_preds = zip(full_count_score[character], ['Count'] * len(full_count_score[character])) + zip(full_proximity_score[character], ['Proximity'] * len(full_count_score[character]))

        # Generate vector similarities
        if job_labels.get(character):
            for label in job_labels.get(character):
                for pred, method in top_preds:
                    # Add rows to Dataframe iteratively
                    row_idx = similarity_scores.shape[0]
                    score = word2vec_model.compareWords(pred[0], label)
                    similarity_scores.loc[row_idx] = [character, (label, pred), score, method]

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


def jobPredictor(sentences, char_sents, char_name, job_list, job_labels=defaultdict(int)):
    """
    Find potential jobs for candidate in char_name.
    Return list[(jobname, score)]
    """
    count_score = {}
    proximity_score = {}

    # take neighbor sentence to the left and right to create window of sentences
    # take sentence index +-window
    window = 2
    n = len(sentences)
    job_count = defaultdict(int)
    for i in char_sents:
        sent_nostop = sentence_window(sentences, i, window, 'nostop')
        sent_words = sentence_window(sentences, i, window, 'words')
        if char_name in sent_nostop:
            for job in job_list:
                if unicode(job) in sent_nostop:
                    # +1 for each mention
                    # storeCount(count_score, job)
                    # -log(i/n) for each mention
                    proportion = float(i+1)/ n
                    storeIncrement(count_score, job, -log(proportion))
                    # mean proximity score
                    dist = abs(getIdxOfWord(sent_words, job) - getIdxOfWord(sent_words, char_name))
                    storeIncrement(proximity_score, job, dist)
                    job_count[job] += 1
                # divide by total matches to get mean proximity measure
            if job_count[job]:
                proximity_score[job] = float(proximity_score[job]) / float(job_count[job])
    proximity_score = {k:
            float(v) / job_count[k] for k,v in proximity_score.items()}
    return count_score, proximity_score


def sentence_window(sentences, index, window, s_type):
    """
    :param sentences: List of dicts. Each dict is a sentence
    and contains 'nostop', 'words', 'tags'
    :param index: index with character mention in sentence
    :param window: window size
    :type: 'nostop', 'words' or 'tags'
    """
    w_range = (index-window, index+window)
    res = []
    for s in w_range:
        res += sentences[s][s_type]
    return res


def printStore(store):
    for k, v in store.items():
        print(k, v)



# visualize with t-SNE http://homepage.tudelft.nl/19j49/t-SNE.html?
