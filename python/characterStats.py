# coding: utf8
# !/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

import codecs
import collections
import copy
import csv
import getopt
import hunspell
import json
import math
import operator
import os
import pickle
import re
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import imp
import mwclient
import hunspell
from computeMeta import runMeta
from tools import *
from nltk.internals import find_binary, find_file


if sys.version_info < (3, 0):
	reload(sys)
	sys.setdefaultencoding('utf8')

warnings.simplefilter("error")

os.environ["TREETAGGER_HOME"] = "/home/alexis/Documents/EPFL/MS3/Project/python/TreeTagger/cmd"

hunspellstemmer = hunspell.HunSpell(getScriptPath() + '/dictionaries/fr-toutesvariantes.dic', getScriptPath() + '/dictionaries/fr-toutesvariantes.aff')

def getScriptPath():
	return "/home/alexis/Documents/EPFL/MS3/Project/python"

sys.path.append(getScriptPath() + '/treetagger-python')
from treetagger3 import TreeTagger

tt = TreeTagger(encoding='utf-8', language='french')

stopwords = set(
	line.strip() for line in codecs.open(getScriptPath() + "/classifiersdata/stopwords.txt", 'r', 'utf8') if
	line != u'')
stopwords_pnouns = set(
	line.strip() for line in codecs.open(getScriptPath() + "/classifiersdata/stopwords_pnouns.txt", 'r', 'utf8') if
	line != u'')

structuralRules = []
rules_str = [line.strip() for line in codecs.open(getScriptPath() + "/classifiersdata/struct_rules.txt", 'r', 'utf8')]
for r in rules_str:
	prediction = r.split(':')[1]
	predicate = r.split(':')[0]
	pkeybuffer = ['']
	p = {int(p.split('=')[0]): p.split('=')[1] for p in predicate.split('&')}
	for i in range(4):
		if i in p:
			nbuffer = []
			for idx, pkey in enumerate(pkeybuffer):
				for ppart in p[i].split(','):
					nbuffer.append(pkey + ppart)
			pkeybuffer = nbuffer
		else:
			for idx, pkey in enumerate(pkeybuffer):
				pkeybuffer[idx] = pkey + '...'
	for pkey in pkeybuffer:
		rule = re.compile(pkey)
		structuralRules.append([rule, prediction])

# Names that are mentioned less than n times in the whole book will be ignored
# (adjusted automatically if dynamicFrequenceFilter = True)
WORD_FREQUENCE_THRESHOLD = 5
MIN_NOUN_LENGTH = 2  # Nouns shorter than that will be ignored
MINIMAL_MEDIAN_IDX = 1.0  # Names whose median position in sentences are ≤ than 1 will be ignored
MAX_CHARACTERS_GRAPH = 50  # Absolute max number of characters considered for final graph
dynamicFrequenceFilter = False

nobliaryParticles = [
	u'de', u'd', u"d'", u'del', u'dal', u'da', u'di', u'della', u'du', u'des', u'la', u'le', u'of', u'van', u'von',
	u'vom', u'zu', u'-']

# TOOLS ################################################################################################################

_names = {}
_tagnums = []
compoundNouns = {}


# BOT 5 ################################################################################################################

onlineDisambiguationClasses = {
	"character": [
		"personnage", "personnalité", "prénom", "animal", "saint", "naissance", "décès", "peuple", "ethni",
		"patronym"],
	"place": [
		"lieu", "ville", "commune", "pays", "région", "territoire", "province", "toponym", "géographi", "géolocalisé",
		"maritime"],
	"other": ["philosophi", "divinité", "dieu", "religion", "sigle", "code", "science", "nombre", "mathématique"]
}
# wikip:  We reached a general information page ("Wikipedia category", "Wikipedia disambiguation",...)
onlineDisambiguationStopwords = ["wikip", "article", "littérature", "littéraire"]
cachedResults = {}


def cachedOnlineDisambiguation(site_TODO, term):
	if term in cachedResults:
		return cachedResults[term]
	else:
		return False


def onlineDisambiguation(site, term, originalTerm=None, debug=False, iter=1, checkedClasses=[]):
	if debug:
		print("***** Online results for " + term + " *****")
	if originalTerm is None:
		originalTerm = term
	cachedResult = cachedOnlineDisambiguation(site, term)
	if cachedResult is not False and not debug:
		return cachedResult
	else:
		if site is not False:
			if iter < 5:
				pages = site.search(compoundNouns[originalTerm])
				for pageData in pages:
					page = site.Pages[pageData['title']]
					foundAtLeastOneCategory = False
					needToLookInText = False
					categoriesBasedDisambiguation = []
					for cat in page.categories():
						foundAtLeastOneCategory = True
						if debug:
							print(
								compoundNouns[originalTerm] + " (as " + term + ",iter=" + str(iter) + ")" + "\t" +
								pageData[
									'title'] + "\t" + cat.name)
						for k, cls in onlineDisambiguationClasses.iteritems():
							for cl in cls:
								if 'homonymie' in cat.name.lower():
									needToLookInText = True
								if cl in cat.name.lower():
									categoriesBasedDisambiguation.append([k, 0 if k == 'unknown' else 1])
					if needToLookInText:
						fullText = page.text().lower()
						tot_all = 0  # all occurences of all classification words found
						fullTextClasses = []
						for k, cls in classes_local.iteritems():
							tot_cl = 0  # all occurences of the words cls corresponding to class k
							for cl in cls:
								tot_cl = tot_cl + fullText.count(cl)
							fullTextClasses.append([k, tot_cl])
							tot_all += tot_cl
						if len(fullTextClasses) > 0:
							maxCountIdx = idxForMaxKeyValPair(fullTextClasses)  # Returns key yielding the highest count
							confidence = ((1 / (iter * (len(checkedClasses) + 1))) * (
								fullTextClasses[maxCountIdx][1] / tot_all) if tot_all > 0 else 0)
							# if (confidence==0):
							# 	print originalTerm
							# 	print term
							# 	print str(checkedClasses)
							# 	print str(fullTextClasses)
							foundDisambiguation = [fullTextClasses[maxCountIdx][0], confidence]
							if debug:
								print(
									originalTerm + " (" + term + ") -- full text disambiguation results: " + "\t" +
									foundDisambiguation[0] + "\t" + str(foundDisambiguation[1]) + "\t" +
									str(fullTextClasses))
							cachedResults[originalTerm] = foundDisambiguation
							updateCachedResults(site)
							return foundDisambiguation
					elif len(categoriesBasedDisambiguation) > 0:
						bestCat = bestChoice(categoriesBasedDisambiguation, [], debug)
						for c in categoriesBasedDisambiguation:
							bestCatCount = sum([k[1] for k in categoriesBasedDisambiguation if k[0] == bestCat[0]])
						foundDisambiguation = [bestCat[0], bestCatCount / len(categoriesBasedDisambiguation)]
						if bestCatCount == 0:
							print(originalTerm)
							print(term)
							print(bestCat[0])
							print(str(categoriesBasedDisambiguation))
						if debug:
							print(
								originalTerm + " (" + term + ") -- cat based disambiguation results: " + "\t" +
								foundDisambiguation[0] + "\t" + str(foundDisambiguation[1]) + "\t" +
								str(categoriesBasedDisambiguation))
						cachedResults[originalTerm] = foundDisambiguation
						updateCachedResults(site)
						return foundDisambiguation  # +" : "+cat.name

					for cat in page.categories():
						if (not cat.name in checkedClasses) and len(
								[w for w in onlineDisambiguationStopwords if w in cat.name.lower()]) == 0:
							checkedClasses.append(cat.name)
							return onlineDisambiguation(site, cat.name, originalTerm, debug, iter + 1, checkedClasses)
		elif debug:
			print("Wiki Lookup disabled")
		return [u'unknown', 0]


def readCachedResults(site):
	if os.path.isfile(getScriptPath() + "/cache/" + site.host + ".csv"):
		for row in csv.reader(codecs.open(getScriptPath() + "/cache/" + site.host + ".csv", 'r', 'utf8')):
			cachedResults[row[0]] = [row[1], float(row[2])]


def updateCachedResults(site):
	w = csv.writer(codecs.open(getScriptPath() + "/cache/" + site.host + ".csv", "w", 'utf8'))
	for key, val in cachedResults.items():
		w.writerow([key, val[0], val[1]])


# BOT 1 ################################################################################################################
classes_local = {}
for root, dirs, files in os.walk(getScriptPath() + "/classifiersdata/proximitywordclasses"):
	for file in files:
		if file.endswith(".txt"):
			wordsfile = codecs.open(os.path.join(root, file), 'r', 'utf8')
			classes_local[file.replace(".txt", "")] = [line.strip() for line in wordsfile if line[0] != b"#"]


def obviousPredictor(word, indexesOfSentencesContainingWord, sentences, debug=False):
	"""

	:param word:
	:param indexesOfSentencesContainingWord:
	:param sentences:
	:param debug:
	:return:
	"""
	if debug:
		print("***** Obvious results for " + word + " *****")
	scores = {}
	predictingWords = []
	obviousChars = [
		'm', 'm.', 'mr', 'monsieur', 'messieurs', 'mme', 'mrs', 'madame', 'mesdames', 'miss', 'mademoiselle',
		'mesdemoiselles', 'veuf', 'veuve', 'docteur', 'doctoresse', 'maître', 'maîtresse', 'professeur', 'professeure',
		'duc', 'duchesse', 'archiduc', 'archiduchesse', 'grand-duc', 'grande-duchesse', 'marquis', 'marquise', 'comte',
		'comtesse', 'vicomte', 'vicomtesse', 'baron', 'baronne', 'seigneur', 'sieur', 'dame', 'écuyeur', 'messire',
		'sir', 'lady', 'lord', 'émir', 'émira', 'chérif', 'chérifa', 'cheikh', 'cheykha', 'bey', 'calife', 'hadjib',
		'nizam', 'pervane', 'sultan', 'vizir', 'râja', 'rani', 'maharadjah', 'maharajah', 'maharaja', 'malik', 'shah',
		'chah', 'padishah', 'khan', 'altesse', 'excellence', 'majesté', 'dom', 'don', 'père', 'mère', 'soeur', 'frère',
		'fils', 'fille', 'abbé', 'curé', 'révérend', 'inquisiteur', 'inquisitrice', 'évêque', 'cardinal', 'monseigneur',
		'messeigneurs', 'éminence', 'sainteté', 'pharaon', 'despote', 'magnat', 'sire', 'pape', 'pontife', 'roi',
		'reine', 'prince', 'princesse', 'empereur', 'impératrice', 'infant', 'kronprinz', 'kaiser', 'aspirant',
		'caporal', 'colonel', 'commandant', 'commandante', 'lieutenant', 'maréchal', 'sergent', 'officier',
		'sous-officier', 'soldat']
	obviousPlaces = [
		'pays', 'région', 'département', 'ville', 'village', 'cité', 'avenue', 'allée', 'boulevard', 'rue', 'chemin',
		'quai', 'cathédrale', 'abbaye', 'église', 'chapelle', 'mont', 'colline', 'forêt', 'bois', 'océan', 'mer', 'lac',
		'étang']
	obviousOthers = ['dieu', 'déesse', 'jésus', 'marie', 'vierge']
	for index in indexesOfSentencesContainingWord:
		sentence = sentences[index]
		for wIdx, w in enumerate(sentence["words"]):
			if w == word:
				w1 = ''
				w2 = ''
				w3 = ''
				w0 = compoundNouns[w].split(' ')[0].lower()
				if wIdx > 1:
					w1 = sentence['words'][wIdx - 1].lower()
				if wIdx > 2:
					w2 = sentence['words'][wIdx - 2].lower()
				if wIdx > 3:
					w3 = sentence['words'][wIdx - 3].lower()
				if (w0 in obviousChars) or (w1 in obviousChars) or (w2 in obviousChars and w1 in nobliaryParticles):
					predictingWords.append([w0, w1, w2])
					storeCount(scores, 'character')
				if (w1 in obviousPlaces) or (w2 in obviousPlaces and w1 in ['de', 'du', "d'"]):
					predictingWords.append([w1, w2])
					storeCount(scores, 'place')
				if w.lower() in obviousOthers:
					predictingWords.append(w)
					storeCount(scores, 'other')
	if debug:
		print(str(predictingWords) + "\t" + str(scores))
	maxV = 0
	maxK = u'unknown'
	scoresSum = 0
	for k, v in scores.iteritems():
		scoresSum = scoresSum + max(0, v)
		if v > maxV:
			maxV = v
			maxK = k
	if scoresSum > (2 * len(scores)):
		# we trust the result only if we saw enough samples, that is on average more than two by category
		return [maxK, maxV / scoresSum]
	else:
		return [u'unknown', 0]


# BOT 2 ################################################################################################################

def positionPredictor(word, indexesOfSentencesContainingWord, sentences, debug=False):
	if debug:
		print("***** Position results for " + word + " *****")
	positions = []
	for index in indexesOfSentencesContainingWord:
		sentence = sentences[index]
		for wIdx, w in enumerate(sentence["words"]):
			if w == word:
				# if (sentence["tags"][wIdx]!='NAM'):
				positions.append(float(wIdx) / float(len(sentence["words"])))
	meanpos = np.mean(np.array(positions))
	if debug:
		print(word + "\tavg(pos)=" + str(meanpos) + "\tstd(pos)=" + str(np.std(positions)) + "\tcount=" + str(
			len(indexesOfSentencesContainingWord)))
	return ['place' if (meanpos > 0.45) else 'character', abs(0.45 - meanpos)]


# BOT 3 ################################################################################################################

classes_local = {}
for root, dirs, files in os.walk(getScriptPath() + "/classifiersdata/proximitywordclasses"):
	for file in files:
		if file.endswith(".txt"):
			wordsfile = codecs.open(os.path.join(root, file), 'r', 'utf8')
			classes_local[file.replace(".txt", "")] = [line.strip() for line in wordsfile if line[0] != b"#"]


def localProximityPredictor(word, surroundingTerms, debug=False):
	if debug:
		print("***** LocalProx results for " + word + " *****")
		print(word + " <-> " + ", ".join(surroundingTerms.keys()))
	class_probas = {}
	for possible_class in classes_local:
		class_probas[possible_class] = 0
		for class_word in classes_local[possible_class]:
			if class_word in surroundingTerms:
				class_probas[possible_class] = class_probas[possible_class] + surroundingTerms[class_word]
				if debug:
					print(word + "\t" + class_word + " --> " + possible_class + " (x" + str(
						surroundingTerms[class_word]) + ")")
	numberOfClues = sum(class_probas.values())
	maxProba = 0
	confidence = 0
	maxProbaClass = u"unknown"
	if numberOfClues > 2:
		for possible_class in class_probas:
			if class_probas[possible_class] > maxProba:
				maxProba = class_probas[possible_class]
				confidence = float(maxProba) / float(numberOfClues)
				maxProbaClass = possible_class
	if debug:
		print(word + "\t" + maxProbaClass + "\t" + str(confidence))
	return [maxProbaClass, confidence]


# BOT 4 ################################################################################################################

def structuralPredictor(word, indexesOfSentencesContainingWord, sentences, debug=False):
	if debug:
		print("***** Structural results for " + word + " *****")
	scores = {u"place": 0, u"character": 0, u"other": 0, u"unknown": 0}
	# Prediction score variable. If results turns out negative, we assume a place. If positive, a character.
	place_vs_char = 0.0
	# Noise score. If positive, discard result
	noise_score = 0.0
	positions = []
	for index in indexesOfSentencesContainingWord:
		sentence = sentences[index]
		for wIdx, w in enumerate(sentence["words"]):
			if w == word:
				if "VER:" in sentence["tags"][wIdx]:
					# if the word itself is tagged as a verb, we get highly suspicious…
					scores[u"unknown"] += 1.0
				else:
					surroundings = [tag.split(':')[0] for tag in getSurroundings(sentence["tags"], wIdx)]
					if debug:
						print(word + " [" + sentence["tags"][wIdx] + "]," + ",".join(surroundings))
					if "VER" == surroundings[2]:
						scores[u"character"] += 2.0
					elif "VER" in surroundings:
						scores[u"character"] += 0.5
					if "NAM" == surroundings[2]:
						scores[u"character"] += 1.0
					if surroundings[0] == "PRP" or surroundings[1] == "PRP":
						scores[u"place"] += 1.0
					if "VER" == surroundings[1]:
						scores[u"place"] += 0.5
					if surroundings[1] == "DET":
						scores[u"place"] += 0.5
						pass
					if surroundings[1] == "PRP" and surroundings[2] == "---":
						scores[u"other"] += 1.0
					if surroundings[1] == "PUN":  # noise detection (wrongly tokenized sentences).
						scores[u"unknown"] += 1.0
					else:
						scores[u"unknown"] -= 1.0
					# noise detection (wrongly tokenized sentences). If this happens, needs to be compensated 2 times
					if surroundings[0] == "---" and surroundings[1] == "---":
						scores[u"unknown"] += 2.0
					else:
						scores[u"unknown"] -= 1.0
	if debug:
		print(' --> ' + str(scores))
	maxV = 0
	maxK = u'unknown'
	scoresSum = 0
	for k, v in scores.iteritems():
		scoresSum = scoresSum + max(0, v)
		if v > maxV:
			maxV = v
			maxK = k
	return [maxK, maxV / scoresSum if scoresSum > 0 else 0]


def getQuotesPredictorThreshold(words, wsent, sentences, debug):
	speakMentionsRatios = []
	for w in words:
		quotesCount = 0
		for index in wsent[w]:
			if "PUN:cit" in sentences[index]["tags"]:
				quotesCount += 1
		speakMentionsRatios.append(quotesCount / len(wsent[w]))
	ratio = np.mean(speakMentionsRatios)
	if debug:
		print("***********************************************************")
		print("quotesPredictorThreshold = " + str(ratio))
		print("***********************************************************")
	return ratio


def quotesPredictor(word, indexesOfSentencesContainingWord, sentences, quotesPredictorThreshold, debug=False):
	if debug:
		print("***** Quotes/Mentions results for " + word + " *****")
	quotesCount = 0
	for index in indexesOfSentencesContainingWord:
		s = sentences[index]
		if "PUN:cit" in s["tags"]:
			quotesCount += 1
	if quotesCount > 0:
		score = quotesCount / len(indexesOfSentencesContainingWord)
		if debug:
			print("Quotes=" + str(quotesCount) + " / Mentions=" + str(
				len(indexesOfSentencesContainingWord)) + " / Score=" + str(score));
		if score >= quotesPredictorThreshold:
			return ["character", pow((score - quotesPredictorThreshold) / (1 - quotesPredictorThreshold), 2)]
		else:
			return ["place", pow((quotesPredictorThreshold - score) / quotesPredictorThreshold, 2)]
	else:
		return ["place", 0.9]

########################################################################################################################


def tokenizeAndStructure(text):
	taggedText = tt.tag(text)
	tagstats = {}
	chaps = collections.OrderedDict()
	cnum = ''
	chapter_sentences_idx = []
	allsentences = []
	sent_words = []
	sent_tags = []
	sent_lemma = []
	sent_lemmanostop = []
	for tag in taggedText:
		if "_CHAP_" in tag[0]:
			if cnum != '':
				chaps[cnum] = chapter_sentences_idx
				chapter_sentences_idx = []
			cnum = tag[0][6:]
		elif tag[1] == u"SENT":
			nostop = [w for w in sent_words if w not in stopwords]
			sent = {u"words": sent_words, u"tags": sent_tags, u"nostop": nostop, u"lemma": sent_lemma}
			chapter_sentences_idx.append(len(allsentences))
			allsentences.append(sent)
			sent_words = []
			sent_tags = []
			sent_lemma = []
		else:
			sent_words.append(tag[0])
			sent_tags.append(tag[1])
			sent_lemma.append(tag[2])
	return [chaps, allsentences]


########################################################################################################################

def bestChoice(_predictions, weights=[], debug=False):

	predictions = copy.deepcopy(_predictions)
	if len(weights) == 0:
		weights = [1 for p in predictions]
	if debug:
		print(" - Predictions: " + str(predictions))
	zeroProbas = []
	duplicates = []
	for idx, p in enumerate(predictions):
		# Check probabilities, remove predictions with p=0
		if p is None or len(p) != 2:
			print("prediction " + str(idx) + " invalid")
			print("    (len=" + str(len(p)) + "): [" + ",".join(p) + "]")
			exit()
		elif p[1] == 0:
			zeroProbas.append(idx)
		# Apply weighting
		elif weights[idx] == 0:
			zeroProbas.append(idx)
		elif (weights[idx] > 1) and not p[1] == 0:
			for n in range(1, weights[idx]):
				duplicates.append(p)
	for p in duplicates:
		predictions.append(p)
	zeroProbas.sort(reverse=True)
	for pIdx in zeroProbas:
		del predictions[pIdx]  # Remove predictions with probability 0
	if len(predictions) > 0:
		maxProbaIdx = idxForMaxKeyValPair(predictions)  # Returns key yielding the highest probabilities
	else:
		return ['unknown', 0]

	if len(predictions) == 0:
		# in case all the entries were removed, we return a copy of the former first item for compliance
		return copy.deepcopy(_predictions[0])

	allAgree = True
	agreeOnClass = predictions[0][0]
	for p in predictions:
		if p[0] != agreeOnClass:
			allAgree = False
	if allAgree:
		return predictions[maxProbaIdx]  # here we could also return [agreeOnClass, 1]
	else:
		predClasses = {}
		for prediction in predictions:
			storeCount(predClasses, prediction[0])
		# we have exactly as many classes as predictions (i.e. each predictor said something different)
		if len(predClasses) == len(predictions):
			return predictions[maxProbaIdx]
		else:
			mostRepresentedClassesCount = predClasses[max(predClasses.iteritems(), key=operator.itemgetter(1))[0]]
			for pred in predClasses.keys():
				if predClasses[pred] < mostRepresentedClassesCount:
					del predClasses[pred]
			validPredictions = [p for p in predictions if p[0] in predClasses.keys()]
			return validPredictions[idxForMaxKeyValPair(validPredictions)]


def detect_ucwords(fulltext, sentences, debug=False):
	_ucwords = {}
	# Get all the uppercase words that are not leading sentences

	for sent in sentences:
		s = sent[u"nostop"]
		if len(s) > 1:
			grams5 = zip(s[1:-4], s[2:-3], s[3:-2], s[4:-1], s[5:])
			grams3 = zip(s[1:-2], s[2:-1], s[3:])
			grams2 = zip(s[1:-1], s[2:])
			grams1 = zip(s[1:])
			sentUCWords = []
			for gram in grams5:
				if gram[0][0].isupper() and (gram[1] in [u'-', u"'"]) and (gram[3] in [u'-', u"'"]):
					sentUCWords.append(gram)
			for gram in grams3:
				if gram[0][0].isupper() and gram[2][0].isupper():
					if gram[1] in nobliaryParticles:
						sentUCWords.append(gram)
					elif gram[1] in [u"'"]:
						sentUCWords.append(gram)
					elif gram[1][0].isupper():
						sentUCWords.append(gram)
			for gram in grams2:
				if gram[0][0].isupper() and gram[1][0].isupper():
					sentUCWords.append(gram)
			sentUCWords_flat = [w for _tuple in sentUCWords for w in _tuple]
			for gram in grams1:
				if gram[0][0].isupper() and not (gram[0] in sentUCWords_flat):
					sentUCWords.append(gram)
			for gram in sentUCWords:
				gramStrRepresentation = u" ".join(gram).replace(u"' ", u"'")
				storeCount(_ucwords, gramStrRepresentation)
	if debug:
		print("***** UC Words found *****")
		print(", ".join(_ucwords.keys()))
		print("**************************")
	return _ucwords


########################################################################################################################

def getUseStats(word, ucwords, chapters, sentences, wprev, wnext, wsent):
	if len(wsent[word]) > 0:
		chaptersCovering = []
		frequenciesDiff = []
		chapterStart = [i for i in range(0, len(chapters)) if wsent[word][0] in chapters[chapters.keys()[i]]][0]
		chapterEnd = [i for i in range(0, len(chapters)) if wsent[word][-1] in chapters[chapters.keys()[i]]][0]
		for c, csidx in chapters.iteritems():
			intersect = [i for i in csidx if i in wsent[word]]
			chaptersCovering.append(len(intersect))
			expectedPerc = (len(csidx) / len(sentences))
			observedPerc = (len(intersect) / ucwords[word])
			frequenciesDiff.append(abs(expectedPerc - observedPerc))
		return {
			'firstsent': wsent[word][0],
			'lastsent': wsent[word][-1],
			'coverage': (wsent[word][-1] - wsent[word][0]) / len(sentences),
			'chapters': chaptersCovering,
			'chapterStart': chapterStart,
			'chapterEnd': chapterEnd,
			'dp': sum(frequenciesDiff) / 2
		}
	else:
		return {}


def getMainCharacters(ucwords, sentences, wprev, wnext, wsent):
	return ucwords


def sortbydescwordlengths(a, b):
	return len(b) - len(a)


def joinCompoundNouns(fulltext, ucwords):
	"""
	Joins all occurrences of compound uppercase words in fulltext.
	:param fulltext:
		Text in which to replace occurrences
	:param ucwords:
		dict of words and frequencies
	:return:
	"""
	allucwords = copy.deepcopy(ucwords.keys())
	allucwords.sort(sortbydescwordlengths)
	for w in allucwords:
		if (u" " in w) or (u"'" in w):
			wjoined = w.replace(u" ", u"").replace(u".", u"").replace(u"'", u"").encode("utf-8")
			if w.endswith("'"):
				wjoined += u"'"
			fulltext = fulltext.replace(w, wjoined)
			compoundNouns[wjoined] = w
		else:
			compoundNouns[w] = w
	return fulltext


def confirmProperNoun(word, wmedianidx, wsentences, ucwords):
	if (len(word) < MIN_NOUN_LENGTH) or (word.endswith("'") and len(word) < MIN_NOUN_LENGTH + 1):
		if debug:
			print("Word ignored: " + word + "  [len<" + str(MIN_NOUN_LENGTH) + "]")
		return False
	if word.lower() in stopwords:
		if debug:
			print("Word ignored: " + word + "  [in general stopwords" + "]")
		return False
	if word in stopwords_pnouns:
		if debug:
			print("Word ignored: " + word + "  [in proper nouns stopwords" + "]")
		return False
	if wmedianidx <= MINIMAL_MEDIAN_IDX:
		if debug:
			print("Word ignored: " + word + "  [median idx=" + str(wmedianidx) + "]")
		return False
	wordTags = []
	for s in wsentences:
		wordTags.append(s['tags'][s['words'].index(word)])
	if not ('NAM' in wordTags or 'NOM' in wordTags):
		if debug:
			print("Word ignored: " + word + "  [tagged " + str(wordTags) + "]")
		return False
	return True


def removeFalsePositives(sentences, wmedianidx, wprev, wnext, wsent, ucwords):
	for word, medianidx in wmedianidx.iteritems():
		proxWords = {}
		for w in [w for _sub in [wprev[word].keys(), wnext[word].keys()] for w in _sub]:
			storeCount(proxWords, w)
		rejected = False
		if not confirmProperNoun(word, medianidx, [sentences[i] for i in wsent[word]], ucwords):
			rejected = True
		if word.endswith('s') and word[:-1] in ucwords:
			rejected = True
			if debug:
				print("Word ignored: " + word + " supposed plural form of " + word[:-1])
		if rejected:
			del ucwords[word]
			del wprev[word]
			del wnext[word]
			del wsent[word]


def getNounsSurroundings(sentences, ucwords, fulltext):
	wprev = {}
	wnext = {}
	wsent = {}
	wmeanidx = {}
	wmedidx = {}
	allucwords = ucwords.keys()
	for word in allucwords:
		wprev[word] = {}
		wnext[word] = {}
		wsent[word] = []
		# wmeanidx[word] = 0.0
		wPositions = []
		i = 0.0
		for sentIdx, sent in enumerate(sentences):
			wpos = getIdxOfWord(sent["nostop"], word)
			if wpos > -1:
				wsent[word].append(sentIdx)
				wPositions.append(wpos)
				if wpos > 0:
					storeCount(wprev[word], stem(hunspellstemmer, sent["nostop"][wpos - 1]))
				if wpos < len(sent["nostop"]) - 1:
					storeCount(wnext[word], stem(hunspellstemmer, sent["nostop"][wpos + 1]))
				i += 1.0
		if len(wPositions) > 0:
			wmeanidx[word] = np.mean(np.array(wPositions))
			wmedidx[word] = np.median(np.array(wPositions))
		else:
			wmeanidx[word] = 0
			wmedidx[word] = 0
	return [wprev, wnext, wsent, wmeanidx, wmedidx]


def removeBelowThreshold(sentences, wmeanidx, wprev, wnext, wsent, ucwords):
	allucwords = ucwords.keys()
	for word in allucwords:
		if len(wsent[word]) >= WORD_FREQUENCE_THRESHOLD:
			ucwords[word] = len(wsent[word])
		else:
			del ucwords[word]
			del wprev[word]
			del wnext[word]
			del wsent[word]
			del wmeanidx[word]


########################################################################################################################

def processBook(bookfile, mwsite, focus, benchmark, debug=False, verbose=False, graphs=False, meta=False):
	"""
	Main processing function for a bookfile
	:param bookfile: string
		Path to a file containing the book text
	:param mwsite: boolean
	:param focus: string
	:param benchmark: dict(string):string
		Read from .corr file. Key is UC entity. Value is 'character' or 'place'
	:param debug: boolean
	:param verbose: boolean
	:param graphs: boolean
	:return:
	"""
	jsonOut = {}
	ucwords = {}
	sentences = []
	benchmarkValues = {"found": 0, "correct": 0, "predictors": [[], [], [], [], [], [], [], [], []]}
	finalWordClasses = {'character': [], 'place': []}
	allpredictions = {}
	with codecs.open(bookfile, 'r', 'utf8') as f:
		t1 = np.arange(0.0, 5.0, 0.1)
		t2 = np.arange(0.0, 5.0, 0.02)

		chapters_lines_buff = []
		for i, raw_line in enumerate(f):
			line_split = raw_line.split(u"\t")
			chapter_number = line_split[0]
			line = line_split[1]
			line = line.replace(u"’", u"'").replace(u"«", u" « ").replace(u"»", u" » ").replace(u"--", u" « ").replace(
				u"_", u" ").strip()  # .replace(u"-", u" ")
			chapters_lines_buff.append(u'. _CHAP_' + chapter_number + u'. ' + line)
		fulltext = u" ".join(chapters_lines_buff)

		if dynamicFrequenceFilter:
			global WORD_FREQUENCE_THRESHOLD
			allwords = len(re.findall(r'\w+', fulltext))
			# WORD_FREQUENCE_THRESHOLD = round(6+((math.log(math.log(allwords))*allwords)/10000)/5)
			WORD_FREQUENCE_THRESHOLD = round(6 + (allwords / 10000) / 4)

		[chapters, sentences] = tokenizeAndStructure(fulltext)
		if focus == '':
			ucwords = detect_ucwords(fulltext, sentences, debug)
			fulltext = joinCompoundNouns(fulltext, ucwords)
			[chapters, sentences] = tokenizeAndStructure(fulltext)
			ucwords = detect_ucwords(fulltext, sentences, debug)
		else:
			ucwords = {}
			focusWords = focus.split(u",")
			for w in focusWords:
				ucwords[w] = WORD_FREQUENCE_THRESHOLD
				compoundNouns[w] = w
		[wprev, wnext, wsent, wmeanidx, wmedidx] = getNounsSurroundings(sentences, ucwords, fulltext)
		removeFalsePositives(sentences, wmedidx, wprev, wnext, wsent, ucwords)

		ucwtotcount = sum(ucwords.values())
		ucwtotunique = len(ucwords)

		removeBelowThreshold(sentences, wmeanidx, wprev, wnext, wsent, ucwords)
		quotesPredictorThreshold = getQuotesPredictorThreshold(ucwords, wsent, sentences, debug)

		sorted_ucw = sorted(ucwords.items(), key=operator.itemgetter(1))
		sorted_ucw.reverse()
		weights = [3, 1, 1, 1, 1]
		if mwsite:
			weights.append(1)
		for word, wcount in sorted_ucw:
			if not word in compoundNouns:
				compoundNouns[word] = word
			proxWords = {}
			for w in [w for _sub in [wprev[word].keys(), wnext[word].keys()] for w in _sub]:
				storeCount(proxWords, w)
			allpredictions[word] = [
				obviousPredictor(word, wsent[word], sentences, debug),
				positionPredictor(word, wsent[word], sentences, debug),
				localProximityPredictor(word, proxWords, debug),
				structuralPredictor(word, wsent[word], sentences, debug),
				# structuralPredictor2(word, wsent[word], sentences, debug),
				quotesPredictor(word, wsent[word], sentences, quotesPredictorThreshold, debug)
			]
			if mwsite:
				allpredictions[word].append(onlineDisambiguation(mwsite, word, word, debug))
			if len(allpredictions[word]) != len(weights):
				print('ERROR: Weights and predictors mismatch.')
				exit()
			if debug:
				print('-----------------------------------')

		if saveResults:
			with codecs.open(getScriptPath() + u"/cache/results-" + bookfile.split(u"/")[-1], 'wb', 'utf8') as f:
				pickle.dump(allpredictions, f)
		for word, wcount in sorted_ucw:
			if debug: print(word)
			best = bestChoice(allpredictions[word], weights, debug)
			if debug: print(' --> ' + best[0])
			if best[0] in finalWordClasses.keys():
				finalWordClasses[best[0]].append(word)
			if len(benchmark) > 0:
				if word in benchmark.keys():
					benchmarkValues["found"] += 1
					if benchmark[word] == best[0]:
						benchmarkValues["correct"] += 1
					for idx, p in enumerate(allpredictions[word]):
						benchmarkValues["predictors"][idx].append(1 if p[0] == benchmark[word] else 0)
					if verbose:
						print(word + "\t" + best[0] + "\t" + str(benchmark[word] == best[0]) + "\t" + str(
							allpredictions[word]))
				else:
					if verbose:
						print(word + "\t" + best[0] + "\tN/A\t" + str(allpredictions[word]))
			else:
				if verbose:
					print(word + "\t" + best[0] + "\t" + str(best[1]) + "\t" + str(wcount))
			if debug:
				print('===================================')
		if len(benchmark) > 0:
			if verbose:
				print('=== PERFORMANCE EVALUATION ==============================')
			ncat = 0
			unknown_words = []
			correct_predictors = {}
			ref_count = {}  # reference (number of words that should fall in each category, by predictor; last idx=best choice)
			attr_count = {}  # attributions (number of words that fell in each category, by predictor; last idx=best choice)
			for cat in ['character', 'place']:
				ncat += 1
				correct_predictors[cat] = {}
				attr_count[cat] = {}
				ref_count[cat] = 0
				for pred_idx in range(0, len(weights) + 1):
					correct_predictors[cat][pred_idx] = []
					attr_count[cat][pred_idx] = []
				for word, word_predictions in allpredictions.iteritems():
					if word in benchmark.keys():
						if benchmark[word] == cat:  # we only consider the words from this effective category
							ref_count[cat] += 1
							for pred_idx, prediction in enumerate(word_predictions):
								correct_predictors[cat][pred_idx].append(1 if (prediction[0] == cat) else 0)
							correct_predictors[cat][pred_idx + 1].append(
								1 if (cat in finalWordClasses and word in finalWordClasses[cat]) else 0)
					else:
						unknown_words.append(word)  # we store away words that are not listed in the benchmark file
					for pred_idx, prediction in enumerate(word_predictions):
						attr_count[cat][pred_idx].append(1 if prediction[0] == cat else 0)
					attr_count[cat][pred_idx + 1].append(
						1 if (cat in finalWordClasses and word in finalWordClasses[cat]) else 0)
			precision_by_classes = {}
			recall_by_classes = {}
			for pred_idx in range(0, len(weights) + 1):
				precision_by_classes[pred_idx] = []
				recall_by_classes[pred_idx] = []
			for cat, cat_count in ref_count.iteritems():
				for idx, pred_correct in correct_predictors[cat].iteritems():
					precision_by_classes[idx].append(
						(sum(pred_correct) / sum(attr_count[cat][idx]) if sum(attr_count[cat][idx]) > 0 else 1))
					recall_by_classes[idx].append((sum(pred_correct) / cat_count if cat_count > 0 else 0))
			missing_words = list(set(benchmark.keys()) - set([w for ws in finalWordClasses.values() for w in ws]))
			if verbose:
				if len(unknown_words) > 0:
					print("! UNKNOWN WORDS: " + (", ".join(set(unknown_words))))
				if len(missing_words) > 0:
					print("! MISSING WORDS: " + (", ".join(missing_words)))
				for idx in precision_by_classes.keys():
					print(str(idx) + "\t" + "P=" + str(sum(precision_by_classes[idx]) / ncat) + "\t" + "R=" + str(
						sum(recall_by_classes[idx]) / ncat))
				print('===========================================================')
		sortKeys = []
		for v in finalWordClasses['character']:
			sortKeys.append(ucwords[v])
		finalWordClasses['character'] = sortUsingList(finalWordClasses['character'], sortKeys)
		sortKeys = []
		for v in finalWordClasses['place']:
			sortKeys.append(min(wsent[v]))
		finalWordClasses['place'] = sortUsingList(finalWordClasses['place'], sortKeys)
		if api:
			jsonOut['substitutions'] = compoundNouns
			jsonOut['classes'] = finalWordClasses

		if verbose:
			print('Total characters occurences: ' + str(sum([ucwords[x] for x in finalWordClasses['character']])))
			print('Total places occurences: ' + str(sum([ucwords[x] for x in finalWordClasses['place']])))

		if mwsite:
			updateCachedResults(mwsite)
		if len(benchmark) > 0:
			if benchmarkValues["found"] > 0:
				if verbose:
					print("========== BENCHMARK RESULTS ============")
					print("Overall score: " + str(benchmarkValues["correct"] / benchmarkValues["found"]))

		color_sequence = [
			"#000000", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5", "#7A4900",
			"#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6",
			"#1B4400", "#4FC601", "#FFFF00", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900",
			"#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
			"#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744",
			"#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
			"#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED",
			"#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
			"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7",
			"#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F",
			"#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55",
			"#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#001C1E", "#D1F7CE", "#004B28",
			"#C8D0F6", "#A3A489", "#806C66", "#222800", "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59",
			"#8ADBB4", "#1E0200", "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
			"#7ED379", "#012C58", "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393",
			"#943A4D", "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
			"#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5", "#E773CE",
			"#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4", "#00005F", "#A97399",
			"#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01", "#6B94AA", "#51A058", "#A45B02",
			"#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66", "#391406",
			"#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9",
			"#FFFFFE", "#C6DC99", "#203B3C", "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527",
			"#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C",
			"#B88183", "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
			"#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F", "#003109",
			"#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E", "#1A3A2A", "#494B5A",
			"#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F", "#BDC9D2", "#9FA064", "#BE4700",
			"#658188", "#83A485", "#453C23", "#47675D", "#3A3F00", "#061203", "#DFFB71", "#868E7E", "#98D058",
			"#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66", "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F",
			"#545C46", "#866097", "#365D25", "#252F99", "#00CCFF", "#674E60", "#FC009C",
			"#92896B"]  # http://godsnotwheregodsnot.blogspot.ru/
		nbCharacters = len(finalWordClasses['character'])
		if graphs:
			if nbCharacters > 0:
				if nbCharacters > MAX_CHARACTERS_GRAPH:
					finalWordClasses['character'] = [w[0] for w in sorted_ucw if w[0] in finalWordClasses['character']][
													0:MAX_CHARACTERS_GRAPH]

				chaptersPlaces = {}
				for cnum, chapsentencesidx in chapters.iteritems():
					chapterPlaces = {}
					for w2idx, w2 in enumerate(finalWordClasses['place']):
						chapterPlaces[w2] = [
							y for z in [sentences[idx]['words'] for idx in chapsentencesidx] for y in z].count(w2)
					chapterPlace = keyForMaxValue(chapterPlaces)
					chaptersPlaces[cnum] = (finalWordClasses['place'].index(chapterPlace) if chapterPlace != '' else -1)

				eventGraph = {}
				if not api:
					fig, ax = plt.subplots(1, 1, figsize=(18, 10))
					ax.get_xaxis().tick_bottom()
					ax.get_yaxis().tick_left()
					plt.xticks(
						range(0, len(chapters) * nbCharacters, nbCharacters), chapters.keys(), fontsize=10, rotation=90)
					plt.yticks(range(0, len(finalWordClasses['place']), 1), finalWordClasses['place'], fontsize=10)

					for w1idx, w1 in enumerate(finalWordClasses['character']):
						xs = []
						ys = []
						cidx = 0
						for cnum, chapsentencesidx in chapters.iteritems():
							if chaptersPlaces[cnum] != -1:
								intersect = list(set(wsent[w1]) & set(chapsentencesidx))
								if len(intersect) > 0:
									xs.append(cidx * nbCharacters + w1idx)
									ys.append(chaptersPlaces[cnum])
							cidx += 1
						# if the considered charactered is quoted more than once in this chapter, we add it to the list
						if len(xs) > 1:
							xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys), key=operator.itemgetter(0), reverse=False))
							plt.plot(
								xs_sorted, ys_sorted, 'o-', lw=2, color=color_sequence[w1idx % len(color_sequence)],
								label=w1, markersize=8, markeredgewidth=0.0, alpha=0.7)

					ax = plt.subplot(111)
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
					plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
					plt.show()
				else:
					eventGraph['chapters'] = chapters.keys()
					eventGraph['places'] = finalWordClasses['place']
					eventGraph['characters'] = {}
					for w1idx, w1 in enumerate(finalWordClasses['character']):
						xs = []
						ys = []
						cidx = 0
						for cnum, chapsentencesidx in chapters.iteritems():
							if chaptersPlaces[cnum] != -1:
								intersect = list(set(wsent[w1]) & set(chapsentencesidx))
								if len(intersect) > 0:
									xs.append(cidx)
									ys.append(chaptersPlaces[cnum])
							cidx += 1
						eventGraph['characters'][w1] = zip(
							*sorted(zip(xs, ys), key=operator.itemgetter(0), reverse=False))
					jsonOut['eventGraph'] = eventGraph

				intersects = []
				for w1 in finalWordClasses['character']:
					for w2 in [w for w in finalWordClasses['character'] if w != w1]:
						intersect = list(set(wsent[w1]) & set(wsent[w2]))
						if len(intersect) > 0:
							intersects.append([w1, w2, len(intersect)])
				if api:
					jsonOut['charsGraph'] = intersects
				else:
					print("__________ Characters graph ______________")
					print("graph characters {")
					print("   " + "graph[layout=neato, splines=true, overlap=prism];")
					for i in intersects:
						print("   " + i[0] + " -- " + i[1] + " [len=" + str(1 + 1 / i[2]) + ", penwidth=" + str(
							math.sqrt(i[2])) + "];")  # weight="+str(len(intersect))+",
					print("}")

				bipRelations = {}
				for w1 in finalWordClasses['character']:
					bipRelations[w1] = {}
					for cnum, chapsentencesidx in chapters.iteritems():
						if chaptersPlaces[cnum] != -1:
							if len(list(set(wsent[w1]) & set(chapsentencesidx))) > 0:
								storeCount(bipRelations[w1], finalWordClasses['place'][chaptersPlaces[cnum]])

				if api:
					jsonOut['bipGraph'] = bipRelations
				else:
					print("__________ Bipartite graph ______________")
					print("graph bip {")
					print("   " + "graph[layout=neato, splines=true, overlap=prism];")
					print('   "' + '","'.join(finalWordClasses['place']) + '"[shape=box,style=filled];')
					for w1 in bipRelations.keys():
						print('   "' + w1 + '"[fontsize=' + str(round(10 + math.log(ucwords[w1]))) + '];');
					for c, r in bipRelations.iteritems():
						for p, v in r.iteritems():
							print('   "' + c + '"--"' + p + '"[len=' + str(1 + (1 / v)) + ', penwidth=' + str(
								math.sqrt(v)) + '];')
					print("}")
			else:
				print("Plot impossible: no character found.");

		if len(benchmark) > 0:
			# print(bookfile+"\t"+str(sum(precision_by_classes[len(precision_by_classes)-1])/ncat)+"\t"+str(sum(
			# recall_by_classes[len(recall_by_classes)-1])/ncat))
			benchStr = bookfile + "\t" + str() + "\t" + str(
				# +"\t"+str(ucwtotcount)+"\t"+str(ucwtotunique)+"\t"+str(sorted_ucw[0][1])+
				# "\t"+str(len(re.findall(r'\w+', fulltext)))
				WORD_FREQUENCE_THRESHOLD)
			ps = []
			rs = []
			for idx in precision_by_classes.keys():
				p = sum(precision_by_classes[idx]) / ncat
				ps.append(p)
				r = sum(recall_by_classes[idx]) / ncat
				rs.append(r)
				benchStr = benchStr + u"\t" + '{:0.3f}'.format(p) + "\t" + '{:0.3f}'.format(r)
			# benchStr = benchStr+"\n--> Averages: "+str(sum(ps)/len(ps))+" / "+str(sum(rs)/len(rs))
			print(benchStr)

		if api:
			print(json.dumps(jsonOut))

		if meta:
			book = bookfile.split('/')[-1][:-4]
			# runMeta(book, sentences, wsent, finalWordClasses['character'], job_labels, gender_label)

			# To SAVE the data computed for a book to pickled files
			writeData(bookfile, finalWordClasses['character'], wsent, sentences)


########################################################################################################################

try:
	opts, args = getopt.getopt(
		sys.argv[1:], "abcdfgsxw:vm",
		["help", "benchmark", "graphs", "api", "file=", "focus=", "save", "mwclient=", "mincount=", "meta"])
except getopt.GetoptError as err:
	# print help information and exit:
	print(err)  # will print something like "option -a not recognized"
	sys.exit(2)
bookfile = u''
focus = u''
mwclienturl = u''
mwsite = False
benchmark = {}
job_labels = collections.defaultdict(lambda: [])
gender_label = collections.defaultdict(lambda: '')
dobenchmark = False
debug = False
verbose = False
graphs = False
api = False  # API Mode, enable Web (full JSON) output
saveResults = False
meta = False
for o, a in opts:
	if o == "-d":
		debug = True
	elif o in ("-b", "--benchmark"):
		dobenchmark = True
	elif o in ("-g", "--graphs"):
		graphs = True
	elif o in ("-h", "--help"):
		sys.exit()
	elif o in ("-f", "--file"):
		bookfile = a
	elif o in ("-x", "--focus"):
		focus = a
	elif o in ("-v", "--verbose"):
		verbose = True
	elif o in ("-a", "--api"):
		api = True
	elif o in ("-s", "--save"):
		saveResults = True
	elif o in ("-c", "--mincount"):
		if a == 'auto':
			dynamicFrequenceFilter = True
		else:
			WORD_FREQUENCE_THRESHOLD = int(a)
	elif o in ("-w", "--mwclient"):
		mwclienturl = a
		mwsite = mwclient.Site(mwclienturl)
		mwsite.compress = False
		readCachedResults(mwsite)
	elif o in("-m", "--meta"):
		meta = True
	else:
		assert False, "unhandled option"

if dobenchmark:
	with codecs.open(bookfile[:-4] + '.corr', 'r', 'utf8') as f:
		for i, raw_line in enumerate(f):
			line = raw_line.strip().split(u"\t")
			if len(line) > 2:
				# Line has name, type, count
				if int(line[2]) >= WORD_FREQUENCE_THRESHOLD:
					benchmark[line[0]] = (line[1] if line[1] in ['character', 'place'] else 'other')
					# job labels available
					if len(line) > 3 and line[1] == 'character':
						job_labels[line[0]] = line[3]
						# gender labels available
						if len(line) > 4:
							if line[4] in ['m', 'f']:
								gender_label[line[0]] = line[4]

			elif len(line) > 1:
				# Line has name, type
				benchmark[line[0]] = (line[1] if line[1] in ['character', 'place'] else 'other')
			else:
				print('Benchmark file error: line ' + str(i) + ' ignored.')
# If no benchmark, but meta -> create job_labels
elif meta:
	with codecs.open(bookfile[:-4] + '.corr', 'r', 'utf8') as f:
		for i, raw_line in enumerate(f):
			line = raw_line.strip().split(u"\t")
			# job labels
			if len(line) > 3 and line[1] == 'character':
				job_labels[line[0]] = re.findall(r'\w+', unicode(line[3]), re.UNICODE)
				# gender labels
				if len(line) > 4:
					if line[4] in ['m', 'f']:
						gender_label[line[0]] = line[4]

processBook(bookfile, mwsite, focus, benchmark, debug, verbose, graphs, meta)

########################################################################################################################
