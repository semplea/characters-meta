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

    if gender:
        # Compute predictions
        gender_nosolo = genderPredictor(sentences, sents_by_char, char_list, gender_label, full=True, solo=False)
        gender_solo = genderPredictor(sentences, sents_by_char, char_list, gender_label, full=True, solo=True)

        # Save to csv
        gender_nosolo.to_csv(save_path + 'gender_nosolo.csv', encoding='utf-8')
        gender_solo.to_csv(save_path + 'gender_solo.csv', encoding='utf-8')

    if sentiment:
        # # Compute predictions
        # sentiment_nosolo = sentimentPredictor(sentences, sents_by_char, char_list)
        # sentiment_nosolo.to_csv(save_path + 'sentiment_nosolo.csv', encoding='utf-8')

        sentiment_solo = sentimentPredictor(sentences, sents_by_char, char_list, solo=True)
        sentiment_solo.to_csv(save_path + 'sentiment_solo.csv', encoding='utf-8')


def sentimentPredictor(sentences, sents_by_char, char_list, solo=False, reduced=True):
    """
    Predict general sentiment for each character in char_list, and store in returned DataFrame
    """
    # Running takes a while (slow API calls?)
    print 'Predicting sentiment with solo:', solo

    # DataFrame for whole char_list
    df = pd.DataFrame(columns=['Character', 'Label', 'Pos_count', 'Pos_prob', 'Neg_count', 'Neg_prob', 'Neut_count', 'Neut_prob'])

    for idx, character in enumerate(char_list):
        pos_count = 0
        pos_probability = 0.0
        neg_count = 0
        neg_probability = 0.0
        neut_count = 0
        neut_probability = 0.0

        char_sents = sents_by_char[character]

        if solo:
            char_sents, _ = getSoloSents(character, sents_by_char, char_list, idx)

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
            res = literal_eval(r.content.decode('utf-8'))

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


def genderPredictor(sentences, sents_by_char, char_list, gender_label, full=True, solo=False):
    """
    Get gender predictions for full list of chars. Return DataFrame
    """
    # TODO include factor for proximity
    window = 5
    df = pd.DataFrame(columns=['Character', 'Label', 'Title_score', 'Title_score_div',
                'Title_in_name', 'Adj_score', 'Adj_score_div', 'Pron_score', 'Pron_score_div',
                'Mention_count', 'Book_length', 'Span', 'Interaction_count', 'Char_count'])

    for i, character in enumerate(char_list):
        char_sents = sents_by_char[character]

        # obvious titles for male and female characters
        f_title = [u'madame', u'mademoiselle', u'mme', u'mlle', u'm\xe8re', u'mrs', u'ms']
        m_title = [u'monsieur', u'm', u'mr', u'p\xe8re']
        title = 0.0 # title score
        title_in_name = 0.0 # title contained in compound name of char score

        # adj endings for feminin and masculin
        f_adj2 = [u've', u'ce', u'se']
        f_adj3 = [u'gue', u'sse', u'que', u'che', u'tte', u'\xe8te', u'\xe8re']
        f_adj4 = [u'elle', u'enne', u'onne', u'euse', u'cque']
        m_adj1 = [u'g', u'f', u'x', u'c', u't']
        m_adj2 = [u'el', u'er', u'en', u'on']
        m_adj3 = [u'eur']

        adj = 0.0 # adjective score
        pron = 0.0 # pronoun score

        # Get list of sents with only char in them
        # TODO is interaction_count necessary, and useful computed for both solo and non_solo?
        solo_sents, interaction_count = getSoloSents(character, sents_by_char, char_list, i)

        if solo:
            char_sents = solo_sents


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
        w1 = w2 = 1.0
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
    """    w = 1
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

    interaction_count = len(char_sents) - len(solo_sents)

    return list(solo_sents), interaction_count


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
