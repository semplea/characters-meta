# coding: utf8
from read_data import readData
from tools import *
from metaPlot import *
from math import log
from collections import defaultdict
from word_similarity import MyModel
import pandas as pd
import re
import pickle
from random import randrange
import requests


def runMeta(book, sentences, wsent, char_list, job_labels, gender_label, job=False, gender=False, sentiment=False):
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
    ### GLOBAL PARAMS
    # classifier_data_dict has keys [u'tromper', u'nutrition', u'\xe9motions', u'dormir', u'raison', u'\xe9tats', u'vouloir', u'tuer', u'gu\xe9rir', u'relations', u'm\xe9tiers', u'salutations', u'soupir', u'pens\xe9e', u'parole', u'foi']
    classifier_data_dict = readData()
    sents_by_char = wsent
    word2vec_model = MyModel()
    char_list = list(reversed(char_list))
    save_path = 'metadata/' + book + '_'

    ################ JOBS #################
    # Define parameters
    job_list = classifier_data_dict[u'm\xe9tiers']
    N_CHARS = 10 # Num of chars to compute scores for -> default all
    predictors = ['count', 'proximity']

    if job:
        # Compute predictions
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

        # Save to csv
        df_count_full_const.to_csv(save_path + 'count_full_const.csv', encoding='utf-8')
        df_count_full_decr.to_csv(save_path + 'count_full_decr.csv', encoding='utf-8')
        df_count_expo_const.to_csv(save_path + 'count_expo_const.csv', encoding='utf-8')
        df_count_expo_decr.to_csv(save_path + 'count_expo_decr.csv', encoding='utf-8')
        df_proximity_full_const.to_csv(save_path + 'proximity_full_const.csv', encoding='utf-8')
        df_proximity_full_decr.to_csv(save_path + 'proximity_full_decr.csv', encoding='utf-8')
        df_proximity_expo_const.to_csv(save_path + 'proximity_expo_const.csv', encoding='utf-8')
        df_proximity_expo_decr.to_csv(save_path + 'proximity_expo_decr.csv', encoding='utf-8')

    ################## GENDER ###################

    if gender:
        # Compute predictions
        gender_nosolo = genderPredictor(sentences, sents_by_char, char_list, gender_label, full=True, solo=False)
        gender_solo = genderPredictor(sentences, sents_by_char, char_list, gender_label, full=True, solo=True)

        # Save to csv
        gender_nosolo.to_csv(save_path + 'gender_nosolo.csv', encoding='utf-8')
        gender_solo.to_csv(save_path + 'gender_solo.csv', encoding='utf-8')

    if sentiment:
        sentimentPredictor(sentences, sents_by_char, char_list)


def sentimentPredictor(sentences, sents_by_char, char_list):
    full_char_text = []
    char_polarity = {}
    for s in sents_by_char:
        sentiment_sent = [w if w != '<unknown>' else sentences[s]['words'][i] for i, w  in enumerate(sentences[s]['lemma'])]

        # This limits the number of words submitted to the sentiment API
        sentiment_sent = [w for i, w in enumerate(sentences[s]['words']) if     sentences[s]['tags'].split(':')[0] in ['NOM', 'ADJ', 'PUN', 'VER', 'ADV']]

        full_char_text.append(sentiment_sent)

    full_char_text = [item for sublist in full_char_text for item in sublist]
    full_char_text = ' '.join(full_char_text)
    curr_len = len(full_char_text)

    # Max length supported by sentiment API
    while curr_len > 80000:
        new_text = full_char_text.split()
        new_text.pop(randrange(len(new_text)))
        full_char_text = ' '.join(new_text)
        curr_len = len(full_char_text)

    # TODO play with requests module to get polarity info on character



def genderPredictor(sentences, sents_by_char, char_list, gender_label, full=True, solo=False):
    # Try on full and exposition -> even more local? one of the first infos we get... window of first 2 mentions
    # include factor for proximity
    window = 5
    N_CHARS = 3
    df = pd.DataFrame(columns=['Character', 'Label', 'Title_score', 'Title_score_div',
                'Title_in_name', 'Adj_score', 'Adj_score_div', 'Pron_score', 'Pron_score_div',
                'Mention_count', 'Book_length', 'Span', 'Interaction_count', 'Char_count'])
    # df.loc[row_idx] = [
        # character, gender_label[character], title, title/len(char_sents),
        # title_in_name, adj, adj/len(char_sents), pron, pron/len(char_sents),
        # len(char_sents), len(sentences), abs(char_sents[0] - char_sents[-1]),
        # abs(len(char_sents) - len(solo_sents)), len(char_list)]
    for i, character in enumerate(char_list):
        char_sents = sents_by_char[character]

        f_title = [u'madame', u'mademoiselle', u'mme', u'mlle', u'm\xe8re', u'mrs', u'ms']
        m_title = [u'monsieur', u'm', u'mr', u'p\xe8re']
        title = 0.0
        title_in_name = 0.0
        # adj endings for feminin and masculin
        f_adj2 = [u've', u'ce', u'se']
        f_adj3 = [u'gue', u'sse', u'que', u'che', u'tte', u'\xe8te', u'\xe8re']
        f_adj4 = [u'elle', u'enne', u'onne', u'euse', u'cque']
        m_adj1 = [u'g', u'f', u'x', u'c', u't']
        m_adj2 = [u'el', u'er', u'en', u'on']
        m_adj3 = [u'eur']
        adj = 0.0
        # pronoun score
        pron = 0.0
        # List of sents with only char in them

        char_sents = set(char_sents)
        others = char_list[:i] + char_list[i+1:]
        other_sents = set()
        for other in others:
            other_sents = other_sents.union(set(sents_by_char[other]))
        solo_sents = char_sents.difference(other_sents)
        interaction_count = len(char_sents) - len(solo_sents)
        if solo:
            char_sents = solo_sents
        char_sents = list(char_sents)

        # Count pronouns
        for sent_idx in char_sents:
            word_idx = getIdxOfWord(sentences[sent_idx]['words'], character)

            # Full sentence
            w_range = getWindow(char_sents, i, 1)
            sent_words = []
            sent_tags = []
            for r in w_range:
                sent_words += sentences[sent_idx]['words']
                sent_tags += sentences[sent_idx]['tags']

            # Subrange of words in sentence
            # sent_words = []
            # sent_tags = []
            # w_range = getWindow(sentences[sent_idx]['words'], word_idx, 4)
            # for r in w_range:
            #     sent_words.append(sentences[sent_idx]['words'][r])
            #     sent_tags.append(sentences[sent_idx]['tags'][r])

            # Check if association with typical female or male titles
            prev_word = sentences[sent_idx]['words'][word_idx-1]
            if prev_word.lower() in f_title:
                title += 1.0
            elif prev_word.lower() in m_title:
                title -= 1.0

            # Check in name of character as well, if title included
            # Not indicative would be very unlikely
            # Almost cheat feature
            camel_split = re.sub('(?!^)([A-Z][a-z]+)', r' \1', character).split()
            if len(camel_split) > 1 and camel_split[0].lower() in f_title:
                title_in_name += 1.0
            elif len(camel_split) > 1 and camel_split[0].lower() in m_title:
                title_in_name -= 1.0

            # Score masculin and feminin personal pronouns
            for i, tag in enumerate(sent_tags):
                if tag == 'PRO:PER':
                    word = sent_words[i]
                    if word == u'il' or word == u'Il':
                        pron -= 1.0
                    elif word == u'elle' or word == u'Elle':
                        pron += 1.0

            # Score adjectives with masculin or feminin suffix
            for i, tag in enumerate(sent_tags):
                if tag == 'ADJ':
                    word = sent_words[i]
                    if word[-4:] in f_adj4:
                        adj += 1.0
                    elif word[-3:] in f_adj3:
                        adj += 1.0
                    elif word[-3:] in m_adj3:
                        adj += 1.0
                    elif word[-2:] in f_adj2:
                        adj += 1.0
                    elif word[-2:] in m_adj2:
                        adj -= 1.0
                    elif word[-1:] in m_adj1:
                        adj -= 1.0

        # print(character, 'pronoun score', pron, 'adj score', adj, 'title', title)

        ### DECIDE
        # Positive score -> f, negative -> m
        thresh = 0.0
        w1 =  w2 = 1.0 # default at 1.0, check if weight to maximise accuracy (and recall, precision)
        res = '' # If empty, undecidable
        if title > 0:
            res = 'f'
        elif title < 0:
            res = 'm'
        elif pron + adj > thresh:
            res = 'f'
        elif pron + adj < thresh:
            res = 'm'
        # print(character, res, gender_label[character], pron, adj, title)

        # Add to df (only 'features', no prediction)
        if gender_label.get(character) and gender_label[character] in [u'm', u'f']:
            div = 1.0
            span = 1.0
            # In case in solo mode some chars have 0 sents
            if len(char_sents) > 0:
                div = len(char_sents)
                span = abs(char_sents[0] - char_sents[-1])
            row_idx = df.shape[0]
            df.loc[row_idx] = [
                character, gender_label[character], title, title/div,
                title_in_name, adj, adj/div, pron, pron/div, div, len(sentences),
                span, interaction_count, len(char_list)]

    return df



def jobPredictor(sentences, sents_by_char, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=True, predictor='count'):
    """
    Computes predictions and scores for predictor with parameters decreasing and full
    """
    total_sents = len(sentences)
    full_score = {}
    TOP_N_JOBS = 5
    df = pd.DataFrame(columns=['Character', 'Label_Guess', 'Similarity', 'Rank', 'Predictor', 'Mention_Count', 'Increment', 'Size'])
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
            w_range = getWindow(sentences, i, window)
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

        df = get_df(df, character, preds, job_labels, word2vec_model, predictor, len(char_sents), full, decreasing)
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


def get_df(df, character, preds, job_labels, word2vec_model, predictor, mentions, full, decreasing):
    """
    Generate vector similarities and return DataFrame
    """
    if job_labels.get(character):
        for label in job_labels.get(character):
            for rank, pred in preds:
                # Add rows to Dataframe iteratively
                row_idx = df.shape[0]
                score = word2vec_model.compareWords(pred[0], label)
                incr = 'constant' if not decreasing else 'decreasing'
                size = 'full' if full else 'exposition'
                df.loc[row_idx] = [character, (label, pred), score, rank, predictor, mentions, incr, size]

    return df


def printStore(store):
    for k, v in store.items():
        print(k, v)
