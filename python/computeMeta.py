# coding: utf8
from read_data import readData
from tools import *
from metaPlot import *
from math import log
from collections import defaultdict
import wordSimilarity
import pandas as pd
import re
import pickle
from random import randrange
import requests
from ast import literal_eval
import random
import codecs
import csv


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
    word2vec_model = wordSimilarity.MyModel()
    char_list = list(reversed(char_list)) # by decreasing mention count
    save_path = 'metadata/' + book + '_'

    ################ JOBS #################
    # Define parameters
    job_list = classifier_data_dict[u'm\xe9tiers']
    N_CHARS = 10 # Num of chars to compute scores for -> default all
    predictors = ['count', 'proximity']

    if job:
        # Compute predictions
        df_job_full_const = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=True)
        df_job_full_decr = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=True, full=True)
        df_job_expo_const = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=False)
        df_job_expo_decr = jobPredictor(sentences, wsent, char_list, job_labels, job_list, word2vec_model, decreasing=True, full=False)

        # Save to csv
        df_job_full_const.to_csv(save_path + 'job_full_const.csv', encoding='utf-8')
        df_job_full_decr.to_csv(save_path + 'job_full_decr.csv', encoding='utf-8')
        df_job_expo_decr.to_csv(save_path + 'job_expo_decr.csv', encoding='utf-8')
        df_job_expo_const.to_csv(save_path + 'job_expo_const.csv', encoding='utf-8')

    ################## GENDER ###################

    # Load gender dict


    if gender:
        # Compute predictions
        gender_nosolo = genderPredictor(book, sentences, sents_by_char, char_list, gender_label, full=True, solo=False)
        gender_solo = genderPredictor(book, sentences, sents_by_char, char_list, gender_label, full=True, solo=True)
        gender_nosolo_w = genderPredictor(book, sentences, sents_by_char, char_list, gender_label, full=True, solo=False, weighted=True)
        gender_solo_w = genderPredictor(book, sentences, sents_by_char, char_list, gender_label, full=True, solo=True, weighted=True)

        # Save to csv
        gender_nosolo.to_csv(save_path + 'gender_nosolo.csv', encoding='utf-8')
        gender_solo.to_csv(save_path + 'gender_solo.csv', encoding='utf-8')
        gender_nosolo_w.to_csv(save_path + 'gender_nosolo_w.csv', encoding='utf-8')
        gender_solo_w.to_csv(save_path + 'gender_solo_w.csv', encoding='utf-8')


    if sentiment:
        # # Compute predictions
        sentiment_nosolo = sentimentPredictor(sentences, sents_by_char, char_list, reduced=False)
        sentiment_nosolo.to_csv(save_path + 'sentiment_nosolo_top.csv', encoding='utf-8')

        # sentiment_solo = sentimentPredictor(sentences, sents_by_char, char_list, reduced=False, solo=True)
        # sentiment_solo.to_csv(save_path + 'sentiment_solo_top.csv', encoding='utf-8')

    # Print stats
    tokens = 0
    job_len = len([item for item in job_labels.values() if item])
    job_tok = len([item for sublist in job_labels.values() for item in sublist])
    gender_len = len([item for item in gender_label.values() if item != '-'])
    for s in sentences:
        tokens += len(s['words'])
    print('{}, {}, {}, ({}, {}), {}'.format(book, tokens, len(char_list), job_len, job_tok, gender_len))


def sentimentPredictor(sentences, sents_by_char, char_list, solo=False, reduced=True):
    """
    Predict general sentiment for each character in char_list, and store in returned DataFrame
    """
    # Running takes a while (slow API calls?)
    print 'Predicting sentiment with solo:', solo

    # DataFrame for whole char_list
    df = pd.DataFrame(columns=['Character', 'Label', 'Pos_count', 'Pos_prob', 'Neg_count', 'Neg_prob', 'Neut_count', 'Neut_prob'])
    char_list = char_list[0:4] # To get only subset
    for idx, character in enumerate(char_list):
        pos_count = 0
        pos_probability = 0.0
        neg_count = 0
        neg_probability = 0.0
        neut_count = 0
        neut_probability = 0.0

        char_sents = sents_by_char[character]

        if solo:
            char_sents = getSoloSents(character, sents_by_char, char_list, idx)

        if reduced and len(char_sents) > 10:
            # Get subset of sents
            num_to_select = max(len(char_sents)/10, 10)
            char_sents = sorted(random.sample(char_sents, num_to_select))

        # For each character compute aggregate of sentence predictions
        for s in char_sents:
            sentiment_sent = [w if w != '<unknown>' else sentences[s]['words'][i] for i, w  in enumerate(sentences[s]['lemma'])]

            # This limits the number of words submitted to the sentiment API
            # The assumption is that sentiment specific words are typically not prepositions, names, conjunctions etc.
            sentiment_sent = [w for i, w in enumerate(sentences[s]['words']) if sentences[s]['tags'][i].split(':')[0] in ['NOM', 'ADJ', 'PUN', 'VER', 'ADV']]

            # API request to get sentiment classification for string
            sentiment_sent = ' '.join(sentiment_sent)
            r = requests.post('http://text-processing.com/api/sentiment/', data={'text':sentiment_sent, 'language':'french'})

            # make sure request got response
            assert r.status_code == 200

            # Get obect from byte object
            res = objFromByte(r)
            # PMM4FBHNND

            # Aggregate scores
            pos_probability += res['probability']['neg']
            neg_probability += res['probability']['pos']
            neut_probability += res['probability']['neutral']

            # Increment pos and neg count
            label = res['label']
            if label == 'pos':
                pos_count += 1
            elif label == 'neg':
                neg_count += 1
            else:
                neut_count += 1

        label = 'pos' if pos_count > neg_count else 'neg' if neg_count > pos_count else 'neutral'
        # Save to DataFrame
        row_idx = df.shape[0]
        div = len(sents_by_char[character])
        df.loc[row_idx] = [character, label,
            pos_count, pos_probability,
            neg_count, neg_probability,
            neut_count, neut_probability]

        print 'done {0}/{1}'.format(idx+1, len(char_list))

    return df


def genderPredictor(book, sentences, sents_by_char, char_list, gender_label, full=True, solo=False, weighted=False):
    """
    Get gender predictions for full list of chars. Return DataFrame
    """
    window = 0
    df = pd.DataFrame(columns=['Character', 'Label', 'Prediction', 'Score', 'Title_score',
            'Title_in_name', 'Adj_score', 'Pron_score', 'Art_score', 'Name_score'])

    # obvious titles for male and female characters
    f_title = [u'madame', u'mademoiselle', u'mme', u'mlle', u'm\xe8re', u'mrs', u'ms']
    m_title = [u'monsieur', u'm', u'mr', u'p\xe8re']

    # adj endings for feminin and masculin
    f_adj2 = [u've', u'ce', u'se']
    f_adj3 = [u'gue', u'sse', u'que', u'che', u'tte', u'\xe8te', u'\xe8re']
    f_adj4 = [u'elle', u'enne', u'onne', u'euse', u'cque']
    m_adj1 = [u'g', u'f', u'x', u'c', u't']
    m_adj2 = [u'el', u'er', u'en', u'on']
    m_adj3 = [u'eur']

    # articles
    f_art = [u'la', u'cette', u'ma']
    m_art = [u'le', u'ce', u'mon']
    art_tags = ['DET:ART', 'PRO:DEM', 'DET:POS'] # use this for sanity check

    # Load name gender scores if already existing
    try:
        name_scores = pd.read_csv('metadata/' + book + '_char_name_scores.csv')
        name_scores.set_index('Character', inplace=True)
        # with codecs.open('metadata/' + book + '_char_name_scores.csv', mode='r', encoding='utf8') as f:
            # reader = csv.reader(f)
            # next(reader)
            # name_scores = {rows[1]:float(rows[2]) for rows in reader}
    except IOError:
        name_scores = pd.DataFrame({'A' : []})

    for i, character in enumerate(char_list):
        char_sents = sents_by_char[character]

        art = 0.0 # article score
        adj = 0.0 # adjective score
        pron = 0.0 # pronoun score
        title = 0.0 # title score
        title_in_name = 0.0 # title contained in compound name of char score
        name = 0.0

        # First name
        split = camelSplit(character)

        # Use prequeried name gender scores if existing
        if not name_scores.empty and len(split[0]) > 1:
            tmp_character = codecs.encode(character, 'utf-8')
            try:
                name = name_scores.get_value(tmp_character, 'Name_score')
            except KeyError:
                pass
        else:
            # Limited to 1000 requests per day
            r = requests.get('https://api.genderize.io/', params={'name':split[0]})
            if r.status_code != 200:
                print(r.content)
            print(r.content)
            res = objFromByte(r)
            if res:
                n_score = res['probability']
                if res['gender'] == 'female' and len(split[0]) > 1:
                    name = n_score
                elif res['gender'] == 'male' and len(split[0]) > 1:
                    name = -n_score

        if solo:
            # Get list of sents with only char in them
            char_sents = getSoloSents(character, sents_by_char, char_list, i)

        # Count pronouns
        for sent_idx in char_sents:
            word_idx = getIdxOfWord(sentences[sent_idx]['words'], character)

            # Full sentence
            w_range = getWindow(char_sents, i, window)
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
            prev_word = sentences[sent_idx]['words'][word_idx-1].lower()
            prev_tag = sentences[sent_idx]['tags'][word_idx-1]
            if prev_word in f_title:
                title += 1.0
            elif prev_word in m_title:
                title -= 1.0
            elif prev_word in f_art and prev_tag in art_tags: # Check for obvious articles
                art += 1.0
            elif prev_word in m_art and prev_tag in art_tags:
                art -+ 1.0


            # Check in name of character as well, if title included
            # Not indicative would be very unlikely
            # Almost cheat feature
            camel_split = camelSplit(character)
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
        w1 = w2 = w3 = w4 = w5 = w6 = 1.0
        if weighted:
            if not solo:
            # trained on nosolo
                [w1, w2, w3, w4, w5, w6] = [2, 3, 1, 1, 2, 1]
            else:
            # trained on solo
                [w1, w2, w3, w4, w5, w6] = [2, 3, 1, 1, 3, 1]

        res = '' # If empty, undecidable
        score = w1 * title + w2 * title_in_name + w3 * adj + w4 * pron + w5 * art + w6 * name
        if name >= 0.95:
            res = 'f'
        elif name <= -0.95:
            res = 'm'
        elif score > 0:
            res = 'f'
        elif score < 0:
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
            df.loc[row_idx] = [character, gender_label[character], res, score, title/div, title_in_name/div, adj/div, pron/div, art/div, name]

    return df


def jobPredictor(sentences, sents_by_char, char_list, job_labels, job_list, word2vec_model, decreasing=False, full=True):
    """
    Computes predictions and scores for predictor with parameters decreasing and full
    """
    full_score = {}
    TOP_N_JOBS = 5
    df = pd.DataFrame(columns=['Character', 'Label_Guess', 'Similarity', 'Rank', 'Mention_Count', 'Increment', 'Size'])
    job_count = defaultdict(int)

    # take neighbor sentence to the left and right to create window of sentences
    # take sentence index +-window
    window = 2

    # For each character
    for i, character in enumerate(char_list):
        char_sents = {}
        count_score = defaultdict(lambda: 0.0)
        prox_score = defaultdict(lambda: 0.0)
        # For each sentence in which character is mentioned
        char_sents = sents_by_char[character]
        if not full:
            # Get reduced sentence set
            # ASSUMPTION, either 10 or 10th of character mentions
            char_sents = char_sents[:int(max(10, len(char_sents)/10))]

        total_sents = len(char_sents)
        for i, s in enumerate(char_sents):
            # Compute sentence window to look at
            w_range = getWindow(sentences, s, window)
            sent_nostop = []
            sent_words = []
            sent_tags = []
            for w in w_range:
                sent_nostop += sentences[w]['nostop']
                sent_words += sentences[w]['words']
                sent_tags += sentences[w]['tags']

            # If character is mentioned in sentence window, add to score
            if character in sent_nostop:
                for job in job_list:
                    if unicode(job) in sent_nostop:
                        countPredict(count_score, decreasing, job, i, total_sents)
                        proxPredict(prox_score, decreasing, job, i, total_sents, character, job_count, sent_words)

        # divide by total matches to get mean proximity measure
        prox_score = {k: float(v) / job_count[k] for k,v in prox_score.items() if job_count[k] > 0}

        # Normalize (and reverse, so that higher is better for proximity) for both scores
        if count_score and prox_score:
            max_val = max(count_score.values())
            count_score = {k: float(v) / max_val for k,v in count_score.items() if max_val > 0}
            max_val = max(prox_score.values())
            prox_score = {k: -(float(v) / max_val - 1) for k,v in prox_score.items() if max_val > 0}

        # Combine here
        w1 = 5
        w2 = 1
        merged = defaultdict(lambda: 0.0)
        for k,v in count_score.items():
            merged[k] = merged[k] + w1*v
        for k,v in prox_score.items():
            merged[k] = merged[k] + w2*v
        full_score[character] = sortNTopByVal(merged, TOP_N_JOBS, descending=True)

        # contains tuples with (rank_in_list, pred)
        preds = list(enumerate(full_score[character]))

        df = getCountDF(df, character, preds, job_labels, word2vec_model, len(char_sents), full, decreasing)
    return df


def countPredict(score, decreasing, job, pos, total_sents):
    """
    Increment function for job count measure. Increase score dict with `w` for `job`.
    If decreasing=True, `w` is exponentially decreasing.
    """
    # 1 per mention
    # w exponentially decreasing if decreasing=True
    w = 1
    if decreasing:
        w = -log(float(pos+1)/ total_sents)

    storeIncrement(score, job, w)


def proxPredict(score, decreasing, job, pos, total_sents, character, job_count, sent_words):
    """
    Increment function for job count measure. Increase score dict with 'w' for 'job'.
    If decreasing=True, `w` is exponentially decreasing.
    """
    w = 1
    if decreasing:
        w = -log(float(pos+1) / total_sents)

    dist = abs(getIdxOfWord(sent_words, job) - getIdxOfWord(sent_words, character))
    storeIncrement(score, job, w*dist)
    job_count[job] += w


def getSoloSents(character, sents_by_char, char_list, char_idx):
    """
    Get sentence indices with occurrences of ONLY given character.
    Return tuple with list of indices, difference count -> |nosolo| - |solo|
    """
    char_sents = set(sents_by_char[character])
    others = char_list[:char_idx] + char_list[char_idx+1:]
    other_sents = set()
    # Union of sents of all other chars
    for other in others:
        other_sents = other_sents.union(set(sents_by_char[other]))

    # Difference of character sentences with other sentences to get solo
    solo_sents = char_sents.difference(other_sents)

    return list(solo_sents)


def getCountDF(df, character, preds, job_labels, word2vec_model, mentions, full, decreasing):
    """
    Generate vector similarities and return DataFrame for count score
    """
    if job_labels.get(character):
        for label in job_labels.get(character):
            for rank, pred in preds:
                # Add rows to Dataframe iteratively
                row_idx = df.shape[0]
                score = word2vec_model.compareWords(pred[0], label)
                incr = 'constant' if not decreasing else 'decreasing'
                size = 'full' if full else 'exposition'
                df.loc[row_idx] = [character, (label, pred), score, rank, mentions, incr, size]

    return df


def printStore(store):
    for k, v in store.items():
        print(k, v)
