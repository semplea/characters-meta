# coding: utf-8
from read_data import readData
from tools import *
from math import log
from collections import defaultdict
from word_similarity import MyModel


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
    TOP_N_JOBS = 10
    char_list = list(reversed(char_list))
    classifier_data_dict = readData()
    job_list = classifier_data_dict['metiers']
    full_count_score = {}
    full_proximity_score = {}
    sents_by_char = buildSentsByChar(char_list, sentences)
    word2vec_model = MyModel()
    N_CHARS = 2
    similarity_scores = dict.fromkeys(char_list[:N_CHARS], [])

    for character in char_list[:N_CHARS]:
        # scores per character
        # TODO
        count_score, proximity_score = jobPredictor(sentences, character, job_list)
        full_count_score[character] = sortNTopByVal(count_score, TOP_N_JOBS, True)
        full_proximity_score[character] = sortNTopByVal(proximity_score, TOP_N_JOBS)
        # Choose best predictions for meta benchmark
        top_preds = [full_count_score[character][0], full_proximity_score[character][0]]
        # Generate vector similarities
        print(job_labels[character])
        for label in job_labels[character]:
            for top in top_preds:
                similarity_scores[character].append(
                word2vec_model.compareWords(top[0], label))


    print("")
    print('===========COUNT SCORE=============')
    printScore(full_count_score)
    print("")
    print('===========PROXIMITY SCORE=============')
    printScore(full_proximity_score)
    print("")
    print('===========SIMILARITY SCORE=============')
    print(similarity_scores)

    # Computing job suggestion similarity with labels


def jobPredictor(sentences, char_name, job_list, job_labels=defaultdict(int)):
    """
    Find potential jobs for candidate in char_name.
    Return list[(jobname, score)]
    """
    count_score = {}
    proximity_score = {}
    # Check in previous and next sentences as well
    # TODO Go by index in sents instead
    # for i = 0 prev_sent is List() and same for end of sentences
    # TODO generalise to sentences +-i context
    # take neighbor to the left and right to create sliding window of sentences
    window_size = 3
    n = len(sentences)
    job_count = defaultdict(int)
    for i, sent in enumerate(sentences):
        sent_nostop = sent['nostop']
        sent_words = sent['words']
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


def printScore(store):
    for k, v in store.items():
        print(k, v)

# Load model
# model = Word2Vec.load_word2vec_format('frWac_postag_no_phrase_700_skip_cut50.bin', binary=True)
# model.similarity('femme_n', 'homme_n')
#
# visualize with t-SNE http://homepage.tudelft.nl/19j49/t-SNE.html?
