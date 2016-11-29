# coding: utf-8
from read_data import read_data
from tools import *
from math import log


def run_meta(sentences, char_list):
    """
    Compute various metadata about characters in char_list
    :param sentences: list(dict)
        List of dicts. Each dict is a sentence
        and contains 'nostop', 'words', 'tags'
    :param char_list: list(unicode)
        List of character names in unicode
        Compound names are concatenated as in sentences

    """
    TOP_N_JOBS = 10
    char_list = list(reversed(char_list))
    classifier_data_dict = read_data()
    job_list = classifier_data_dict['metiers']
    full_count_score = {}
    full_proximity_score = {}
    for character in char_list[0:10]:
        # scores per character
        count_score, proximity_score = job_coocurrence(sentences, character, job_list)
        full_count_score[character] = sortntop_byval(count_score, TOP_N_JOBS, True)
        full_proximity_score[character] = sortntop_byval(proximity_score, TOP_N_JOBS)
    print('===========COUNT SCORE=============')
    print_score(full_count_score)
    print('===========PROXIMITY SCORE=============')
    print_score(full_proximity_score)


def job_coocurrence(sentences, char_name, job_list):
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
    for i, sent in enumerate(sentences):
        sent_nostop = sent['nostop']
        sent_words = sent['words']
        if char_name in sent_nostop:
            for job in job_list:
                matched = 0
                if unicode(job) in sent_nostop:
                    # +1 for each mention
                    # storeCount(count_score, job)
                    # -log(i/n) for each mention
                    proportion = float(i+1)/ n
                    store_increment(count_score, job, -log(proportion))
                    # mean proximity score
                    dist = abs(getIdxOfWord(sent_words, job) - getIdxOfWord(sent_words, char_name))
                    store_increment(proximity_score, job, dist)
                    matched += 1
                # divide by total matches to get mean proximity measure
                if matched:
                    proximity_score[job] = float(proximity_score[job]) / float(matched)

    return count_score, proximity_score

def print_score(store):
    for k, v in store.items():
        print(k, v)
