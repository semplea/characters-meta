# coding: utf-8
from read_data import read_data
from tools import *

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
    TOP_N_JOBS = 4
    char_list = list(reversed(char_list))
    classifier_data_dict = read_data()
    job_list = classifier_data_dict['metiers']
    potential_char_jobs = {}
    for character in char_list[0:10]:
        potential_dict = find_job(sentences, character, job_list)
        potential_char_jobs[character] = sortntop_byval_desc(potential_dict, TOP_N_JOBS)

    print(potential_char_jobs)


def find_job(sentences, char_name, job_list):
    """
    Find potential jobs for candidate in char_name.
    Return dict(job name): count
    """
    potential_jobs = {}
    for sent in sentences:
        sent_nostop = sent['nostop']
        for job in job_list:
            if char_name in sent_nostop and unicode(job) in sent_nostop:
                storeCount(potential_jobs, job)
    return potential_jobs

def sortntop_byval_desc(tosort, top):
    """
    Sort dictionary by descending values (and return top elements)
    """
    return sorted(tosort, key=tosort.get, reverse=True)[:top]
