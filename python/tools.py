# coding: utf8
# !/usr/bin/env python

import hunspell
import pandas as pd
from math import log
import matplotlib.pyplot as plt
import seaborn as sns
import codecs
import pickle

def getScriptPath():
	return "/home/alexis/Documents/EPFL/MS3/Project/python"

def getIdxOfWord(ws, w):
	"""Return index of word in sentence"""
	try:
		wIdx = ws.index(w)
	except:
		wIdx = -1
	return wIdx


def stem(stemmer, word):
	"""
	Computes a possible stem for a given word
	:param word: string
		The word to be stemmed
	:return: string
		The last possible stem in list, or the word itself if no stem found
	"""
	wstem = stemmer.stem(word)
	if len(wstem) > 0:  # and wstem[-1] not in stopwords
		return unicode(wstem[-1], 'utf8')
	else:
		return word



def storeCount(array, key):
	"""Increments value for key in store by one, or sets to 1 if key nonexistent."""
	if key in array:
		array[key] += 1
	else:
		array[key] = 1


def storeIncrement(store, key, incr):
	"""
	Increment value for key in store by given increment.
	:param incr: float
	"""
	if key in store:
		store[key] += incr
	else:
		store[key] = incr


def idxForMaxKeyValPair(array):
	maxV = array[0][1]
	i = 0
	maxVIdx = 0
	for k, v in array:
		if v > maxV:
			maxV = v
			maxVIdx = i
		i += 1
	return maxVIdx


def keyForMaxValue(_dict):
	maxK = ''
	maxV = 0
	for k, v in _dict.iteritems():
		if v > maxV:
			maxV = v
			maxK = k
	return maxK


def sortUsingList(tosort, reflist):
	"""
	Sorts tosort by order of reflist.
	Example: tosort: ['a', 'b', 'c'], reflist: [1, 3, 2]
	Return: ['a', 'c', 'b']
	:param tosort:
	:param reflist:
	:return:
	"""
	return [x for (y, x) in sorted(zip(reflist, tosort))]

def sortNTopByVal(tosort, top, descending=False):
    """
    Sort dictionary by descending values and return top elements.
	Return list of tuples.
    """
    return sorted([(k, v) for k, v in tosort.items()], key=lambda x: x[1], reverse=descending)[:top]

def buildSentsByChar(chars, sents):
	"""
	DEPRECATED-NOT NEEDED ANY MORE
	Build map of chars to list of indices where characters occur in sents.
	"""
	char_sent_map = dict.fromkeys(chars, list())
	for ix, sent in enumerate(sents):
		for char, ix_lst in char_sent_map.iteritems():
			if char in sent['nostop']:
				ix_lst.append(ix)
	return char_sent_map


def writeData(bookfile, char_list, wsent, sentences):
	"""
	Write data relevant to book to pickle files
	"""
	file_prefix = '../books-txt/predicted-data/'
	name_prefix = bookfile.split('/')[-1][:-4] # TODO get without .txt
	# write list to file, one element per line
	with codecs.open(file_prefix + name_prefix + '-chars.p', mode='wb') as f:
		pickle.dump(char_list, f)
	# write characters sentences dict to file in json format
	with codecs.open(file_prefix + 	name_prefix + '-charsents.p', mode='wb') as f:
		pickle.dump(wsent, f)
	# write sentences dict to file in json format
	with codecs.open(file_prefix + name_prefix + '-sents.p', mode='wb') as f:
		pickle.dump(sentences, f)


def getSurroundings(array, idx, window=2):
	"""
	Return words +-2 from idx
	"""
	surroundings = []
	if idx > 1:
		surroundings.append(array[idx - 2])
	else:
		surroundings.append('---')
	if idx > 0:
		surroundings.append(array[idx - 1])
	else:
		surroundings.append('---')
	if idx < len(array) - 1:
		surroundings.append(array[idx + 1])
	else:
		surroundings.append('---')
	if idx < len(array) - 2:
		surroundings.append(array[idx + 2])
	else:
		surroundings.append('---')
	return surroundings


def getWindow(lst, index, window):
    """
    :param lst: Some list
    :param index: index at senter of window
    :param window: window size -> +- window on each side
		Total size of 2*window+1
    """
    min_idx = index-window if index-window >= 0 else 0
    max_idx = index+window if index+window < len(lst) else len(lst)-1
    return range(min_idx, max_idx+1)
