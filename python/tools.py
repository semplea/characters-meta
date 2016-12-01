# coding: utf8
# !/usr/bin/env python

import hunspell
from math import log

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


def store_increment(array, key, incr):
	"""
	Increment value for key in store by given increment.
	:param incr: float
	"""
	if key in array:
		array[key] += incr
	else:
		array[key] = incr


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

def sortntop_byval(tosort, top, descending=False):
    """
    Sort dictionary by descending values and return top elements.
	Return list of tuples.
    """
    return sorted([(k, v) for k, v in tosort.items()], key=lambda x: x[1], reverse=descending)[:top]

def build_sents_by_char(chars, sents):
	"""
	Build map of chars to list of indices where characters occur in sents.
	"""
	print("Building character occurrence map")
	print(chars)
	char_sent_map = dict.fromkeys(chars, list())
	for ix, sent in enumerate(sents):
		for char, ix_lst in char_sent_map.iteritems():
			if char in sent['nostop']:
				ix_lst.append(ix)
	return char_sent_map
