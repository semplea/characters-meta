# coding: utf8
# !/usr/bin/env python

import hunspell

def getScriptPath():
	return "/home/alexis/Documents/EPFL/MS3/Project/python"


def stem(word):
	"""
	Computes a possible stem for a given word
	:param word: string
		The word to be stemmed
	:return: string
		The last possible stem in list, or the word itself if no stem found
	"""
	hunspellstemmer = hunspell.HunSpell(getScriptPath() + '/dictionaries/fr-toutesvariantes.dic', getScriptPath() + '/dictionaries/fr-toutesvariantes.aff')

	wstem = hunspellstemmer.stem(word)
	if len(wstem) > 0:  # and wstem[-1] not in stopwords
		return unicode(wstem[-1], 'utf8')
	else:
		return word


def storeCount(array, key):

	if key in array:
		array[key] += 1
	else:
		array[key] = 1


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
