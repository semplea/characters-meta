# coding: utf8
#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import division
import sys, glob, os
import getopt
import math
import collections
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from matplotlib.legend_handler import HandlerLine2D

if sys.version_info < (3,0):
	reload(sys)
	sys.setdefaultencoding('utf8')

#import nltk
#from nltk import word_tokenize

import re
import hunspell
import operator

import copy

import csv

'''
export TREETAGGER_HOME='/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd'
'''
os.environ["TREETAGGER_HOME"] = "/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd"

sys.path.append('treetagger-python')
from treetagger3 import TreeTagger
tt = TreeTagger(encoding='utf-8',language='french')


import urllib
import mwclient

import scipy
from scipy.cluster.vq import kmeans2
from scipy.interpolate import interp1d
import numpy as np
from numpy import matrix

stopwords = set(line.strip() for line in open("stopwords.txt", 'r') if line!='')
stopwords_pnouns = set(line.strip() for line in open("stopwords_pnouns.txt", 'r') if line!='')


structuralRules = []
rules_str = [line.strip() for line in open("struct_rules.txt", 'r')]
for r in rules_str:
	prediction = r.split(':')[1]
	predicate = r.split(':')[0]
	pkeybuffer = ['']
	p = {int(p.split('=')[0]):p.split('=')[1] for p in predicate.split('&')}
	for i in range(4):
		if i in p:
			nbuffer = []
			for idx, pkey in enumerate(pkeybuffer):
				for ppart in p[i].split(','):
					nbuffer.append(pkey+ppart)
			pkeybuffer = nbuffer
		else:
			for idx, pkey in enumerate(pkeybuffer):
				pkeybuffer[idx] = pkey+'...'
	for pkey in pkeybuffer:
		rule = re.compile(pkey)
		structuralRules.append([rule, prediction])

WORD_FREQUENCE_THRESHOLD = 5			# Names that are mentioned less than n times in the whole book will be ignored (adjusted automatically if dynamicFrequenceFilter = True)
MIN_NOUN_LENGTH = 2 					# Nouns shorter than that will be ignored
MINIMAL_MEAN_IDX = 3.0					# Names whose mean position in sentences are less than 3 will be ignored
MAX_CHARACTERS_GRAPH = 50               # Absolute max number of characters considered for final graph
dynamicFrequenceFilter = False

nobliaryParticles = [u'de',u'd',u"d'",u'del',u'dal',u'da',u'di',u'della',u'du',u'des',u'la',u'le',u'of',u'van',u'von',u'vom',u'zu',u'-']

### TOOLS ######################################################################################################################################################

_names = {}
_tagnums = []
compoundNouns = {}

hunspellstemmer = hunspell.HunSpell('dictionaries/fr-toutesvariantes.dic','dictionaries/fr-toutesvariantes.aff')
def stem(word):
	wstem = hunspellstemmer.stem(word)
	if len(wstem)>0:	# and wstem[-1] not in stopwords
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
	for k,v in array:
		if (v > maxV):
			maxV = v
			maxVIdx = i
		i = i+1
	return maxVIdx

def keyForMaxValue(_dict):
	maxK = ''
	maxV = 0
	for k,v in _dict.iteritems():
		if (v>maxV):
			maxV = v
			maxK = k
	return maxK

def sortUsingList(tosort, reflist):
	return [x for (y,x) in sorted(zip(reflist,tosort))]

### ######################################################################################################################################################

def tokenizeAndStructure(text):
	taggedText = tt.tag(text)
	tagstats = {}
	chaps = collections.OrderedDict()
	cnum = ''
	chapter_sentences_idx = []
	allsentences = []
	sent_words = []
	sent_tags = []
	for tag in taggedText:
		if ("_CHAP_" in tag[0]):
			if (cnum!=''):
				chaps[cnum] = chapter_sentences_idx
				chapter_sentences_idx = []
			cnum = tag[0][6:]
		elif (tag[1]==u"SENT"):
			nostop = [w for w in sent_words if w not in stopwords]
			sent = {u"words":sent_words,u"tags":sent_tags,u"nostop":nostop}
			chapter_sentences_idx.append(len(allsentences))
			allsentences.append(sent)
			sent_words = []
			sent_tags = []
		else:
			sent_words.append(tag[0])
			sent_tags.append(tag[1])
	chaps[cnum] = chapter_sentences_idx
	return [chaps, allsentences]


################################################################################################################################################################

def bestChoice(_predictions, weights = [], debug=False):
	predictions = copy.deepcopy(_predictions)
	if len(weights)==0:
		weights = [1 for p in predictions]
	if (debug):
		print(" - Predictions: "+str(predictions))
	zeroProbas = []
	duplicates = []
	for idx, p in enumerate(predictions):
		# Check probabilities, remove predictions with p=0
		if p is None or len(p)!=2:
			print("prediction "+str(idx)+" invalid")
			print("    (len="+str(len(p))+"): ["+",".join(p)+"]")
			exit()
		elif p[1]==0:
			zeroProbas.append(idx)
		# Apply weighting
		if (weights[idx]==0):
			zeroProbas.append(idx)
		elif (weights[idx]>1) and not p[1]==0:
			for n in range(1, weights[idx]):
				duplicates.append(p)
	for p in duplicates:
		predictions.append(p)
	zeroProbas.sort(reverse=True)
	for pIdx in zeroProbas:
		del predictions[pIdx]									# Remove predictions with probability 0
	maxProbaIdx = idxForMaxKeyValPair(predictions)				# Returns key yielding the highest probabilities

	if len(predictions)==0:
		return copy.deepcopy(_predictions[0])					# in case all the entries were removed, we return a copy of the former first item for compliance

	allAgree = True
	agreeOnClass = predictions[0][0]
	for p in predictions:
		if (p[0]!=agreeOnClass):
			allAgree = False
	if (allAgree):
		return predictions[maxProbaIdx]							# here we could also return [agreeOnClass, 1]
	else:
		predClasses = {}
		for prediction in predictions:
			storeCount(predClasses, prediction[0])
		if (len(predClasses)==len(predictions)):				# we have exactly as many classes as predictions (i.e. each predictor said something different)
			return predictions[maxProbaIdx]
		else:
			mostRepresentedClassesCount = predClasses[max(predClasses.iteritems(), key=operator.itemgetter(1))[0]]
			for pred in predClasses.keys():
				if predClasses[pred]<mostRepresentedClassesCount:
					del predClasses[pred]
			validPredictions = [p for p in predictions if p[0] in predClasses.keys()]
			return validPredictions[idxForMaxKeyValPair(validPredictions)]

def detect_ucwords(fulltext, sentences, debug=False):
	_ucwords = {}
	# Get all the uppercase words that are not leading sentences

	for sent in sentences:
		s = sent[u"nostop"]
		if (len(s)>1):
			grams5 = zip(s[1:-4], s[2:-3], s[3:-2], s[4:-1], s[5:])
			grams3 = zip(s[1:-2], s[2:-1], s[3:])
			grams2 = zip(s[1:-1], s[2:])
			grams1 = zip(s[1:])
			sentUCWords = []
			for gram in grams5:
				if (gram[0][0].isupper() and (gram[1] in [u'-', u"'"]) and (gram[3] in [u'-', u"'"])):
					sentUCWords.append(gram)
			for gram in grams3:
				if (gram[0][0].isupper() and gram[2][0].isupper()):
					if (gram[1] in nobliaryParticles):
						sentUCWords.append(gram)
					elif (gram[1] in [u"'"]):
						sentUCWords.append(gram)
					elif (gram[1][0].isupper()):
						sentUCWords.append(gram)
			for gram in grams2:
				if (gram[0][0].isupper() and gram[1][0].isupper()):
					sentUCWords.append(gram)
			sentUCWords_flat = [w for _tuple in sentUCWords for w in _tuple]
			for gram in grams1:
				if (gram[0][0].isupper() and not (gram[0] in sentUCWords_flat)):
					sentUCWords.append(gram)
			for gram in sentUCWords:
				gramStrRepresentation = u" ".join(gram).replace(u"' ", u"'")
				storeCount(_ucwords, gramStrRepresentation)
	if (debug):
		print("***** UC Words found *****")
		print(", ".join(_ucwords.keys()))
		print("**************************")
	return _ucwords

def sortbydescwordlengths(a,b):
	return len(b) - len(a)

def joinCompoundNouns(fulltext, ucwords):
	allucwords = copy.deepcopy(ucwords.keys())
	allucwords.sort(sortbydescwordlengths)
	for w in allucwords:
		if (u" " in w) or (u"'" in w):
			wjoined = w.replace(u" ", u"").replace(u".", u"").replace(u"'", u"").encode("utf-8")
			if (w.endswith("'")):
				wjoined = wjoined+u"'"
			fulltext = fulltext.replace(w, wjoined)
			compoundNouns[wjoined] = w
		else:
			compoundNouns[w] = w
	return fulltext

def confirmProperNoun(word, wmeanidx, wsentences, ucwords):
	if (len(word) < MIN_NOUN_LENGTH) or (word.endswith("'") and len(word) < MIN_NOUN_LENGTH+1):
		if debug:
			print("Word ignored: "+word+" len<"+str(MIN_NOUN_LENGTH))
		return False
	if (word.lower() in stopwords):
		if debug:
			print("Word ignored: "+word+" in general stopwords")
		return False
	if (word in stopwords_pnouns):
		if debug:
			print("Word ignored: "+word+" in proper nouns stopwords")
		return False
	if (wmeanidx<MINIMAL_MEAN_IDX):
		if debug:
			print("Word ignored: "+word+"\t"+str(wmeanidx))
		return False
	wordTags = []
	for s in wsentences:
		wordTags.append(s['tags'][s['words'].index(word)])
#		for i, w in enumerate(s['words']):
#			if w==word:
#				wordTags.append(s['tags'][i])
	if not ('NAM' in wordTags or 'NOM' in wordTags):
		if debug:
			print("Word ignored: "+word+" tagged "+str(wordTags))
		return False
	return True

def getIdxOfWord(ws, w):
	try:
		wIdx = ws.index(w)
	except:
		wIdx = -1
	return wIdx

def removeFalsePositives(sentences, wmeanidx, wprev, wnext, wsent, ucwords):
	for word, meanidx in wmeanidx.iteritems():
		proxWords = {}
		for w in [w for _sub in [wprev[word].keys(), wnext[word].keys()] for w in _sub]:
			storeCount(proxWords, w)
		rejected = False
		if (not confirmProperNoun(word, meanidx, [sentences[i] for i in wsent[word]], ucwords)):
			rejected = True
		if (word.endswith('s') and word[:-1] in ucwords):
			rejected = True
			if debug:
				print("Word ignored: "+word+" supposed plural form of "+word[:-1])
		if (rejected):
			del ucwords[word]
			del wprev[word]
			del wnext[word]
			del wsent[word]
#		else:
#			preditions = [positionPredictor(word, wsent[word], sentences, debug), localProximityPredictor(word, proxWords, debug), onlineDisambiguation(mwsite, word, word, debug)]
#			best = bestChoice(preditions)
#			print(best[0]+"\t"+str(best[1]))

def getNounsSurroundings(sentences, ucwords, fulltext):
	wprev = {}
	wnext = {}
	wsent = {}
	wmeanidx = {}
	allucwords = ucwords.keys()
	for word in allucwords:
		wprev[word] = {}
		wnext[word] = {}
		wsent[word] = []
		wmeanidx[word] = 0.0
		i = 0.0
		for sentIdx, sent in enumerate(sentences):
			wpos = getIdxOfWord(sent["nostop"], word)
			if (wpos > -1):
				wsent[word].append(sentIdx)
				wmeanidx[word] = (wmeanidx[word]*(i/(i+1.0)))+(float(wpos)/(i+1.0))		# update the mean position value (cumulative recompute)
				if wpos>0:
					storeCount(wprev[word], stem(sent["nostop"][wpos-1]))
				if wpos<len(sent["nostop"])-1:
					storeCount(wnext[word], stem(sent["nostop"][wpos+1]))
				i = i+1.0
	return [wprev, wnext, wsent, wmeanidx]

def removeBelowThreshold(sentences, wmeanidx, wprev, wnext, wsent, ucwords):
	allucwords = ucwords.keys()
	for word in allucwords:
		if (len(wsent[word])>=WORD_FREQUENCE_THRESHOLD):
			ucwords[word] = len(wsent[word])
		else:
			del ucwords[word]
			del wprev[word]
			del wnext[word]
			del wsent[word]
			del wmeanidx[word]

################################################################################################################################################################

def getUseStats(word, ucwords, chapters, sentences, wprev, wnext, wsent):
	if len(wsent[word])>0:
		chaptersCovering = []
		frequenciesDiff = []
		chapterStart = [i for i in range(0,len(chapters)) if wsent[word][0] in chapters[chapters.keys()[i]]][0]
		chapterEnd = [i for i in range(0,len(chapters)) if wsent[word][-1] in chapters[chapters.keys()[i]]][0]
		for c, csidx in chapters.iteritems():
			intersect = [i for i in csidx if i in wsent[word]]
			chaptersCovering.append(len(intersect))
			expectedPerc = (len(csidx)/len(sentences))
			observedPerc = (len(intersect)/ucwords[word])
			frequenciesDiff.append(abs(expectedPerc-observedPerc))
		return [wsent[word][0], wsent[word][-1], (wsent[word][-1]-wsent[word][0])/len(sentences), chapterStart, chapterEnd, sum(frequenciesDiff)/2, chaptersCovering]
		return {
				'firstsent':wsent[word][0],
				'lastsent':wsent[word][-1],
				'coverage':(wsent[word][-1]-wsent[word][0])/len(sentences),
				'chapters':chaptersCovering,
				'chapterStart':chapterStart,
				'chapterEnd':chapterEnd,
				'dp': sum(frequenciesDiff)/2
				}
	else:
		return {}

def getMainCharacters(ucwords, sentences, wprev, wnext, wsent):
	return ucwords

def processBook(bookfile, mwsite, focus, benchmark, debug=False, verbose=False, graphs=False):
	ucwords = {}
	sentences = []
	benchmarkValues = {"found":0,"correct":0,"predictors":[[],[],[],[],[],[],[],[],[]]}
	weights = [3, 1, 1, 1, 1, 1]
	finalWordClasses = {'character':[],'place':[]}
	allpredictions = {}
	with open(bookfile) as f:
		t1 = np.arange(0.0, 5.0, 0.1)
		t2 = np.arange(0.0, 5.0, 0.02)

		chapters_lines_buff = []
		for i, raw_line in enumerate(f):
			line_split = raw_line.split("\t")
			chapter_number = line_split[0]
			line = line_split[1]
			line = line.replace(u"--", u" ").replace(u"’", u"'").replace(u"«", u"").replace(u"»", u"").replace(u"_", u" ").strip()	#.replace(u"-", u" ")
			chapters_lines_buff.append('. _CHAP_'+chapter_number+'. '+line)
		fulltext = u" ".join(chapters_lines_buff)

		if (dynamicFrequenceFilter):
			global WORD_FREQUENCE_THRESHOLD
			allwords = len(re.findall(r'\w+', fulltext))
			WORD_FREQUENCE_THRESHOLD = round(6+(allwords/10000)/4)

		[chapters, sentences] = tokenizeAndStructure(fulltext)
		if (focus==''):
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
		[wprev, wnext, wsent, wmeanidx] = getNounsSurroundings(sentences, ucwords, fulltext)
		removeFalsePositives(sentences, wmeanidx, wprev, wnext, wsent, ucwords)

		ucwtotcount = sum(ucwords.values())
		ucwtotunique = len(ucwords)

		removeBelowThreshold(sentences, wmeanidx, wprev, wnext, wsent, ucwords)

		sorted_ucw = sorted(ucwords.items(), key=operator.itemgetter(1))
		for w, c in sorted_ucw:
			usestats = getUseStats(w, ucwords, chapters, sentences, wprev, wnext, wsent)
			print(w+"\t"+str(c)+"\t"+str(usestats[0])+"\t"+str(usestats[1])+"\t"+str(usestats[2])+"\t"+str(usestats[3])+"\t"+str(usestats[4])+"\t"+str(usestats[5])+"\t"+str(usestats[6]));


################################################################################################################################################################

try:
	opts, args = getopt.getopt(sys.argv[1:], "bcdfgxw:v", ["help", "benchmark", "graphs", "file=", "focus=", "mwclient=", "mincount="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err) # will print something like "option -a not recognized"
    sys.exit(2)
bookfile = ''
focus = ''
mwclienturl = ''
mwsite = False
benchmark = {}
dobenchmark = False
debug = False
verbose = False
graphs = False
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
	elif o in ("-c", "--mincount"):
		if a=='auto':
			dynamicFrequenceFilter = True
		else:
			WORD_FREQUENCE_THRESHOLD = int(a)
	elif o in ("-w", "--mwclient"):
		mwclienturl = a
		mwsite = mwclient.Site(mwclienturl)
		mwsite.compress = False
		readCachedResults(mwsite)
	else:
		assert False, "unhandled option"


if (dobenchmark):
	with open(bookfile[:-4]+'.corr') as f:
		for i, raw_line in enumerate(f):
			line = unicode(raw_line.strip(), 'utf8').split(u"\t")
			if (len(line)>2):
				if int(line[2])>=WORD_FREQUENCE_THRESHOLD:
					benchmark[line[0]] = (line[1] if line[1] in ['character','place'] else 'other')
			elif (len(line)>1):
				benchmark[line[0]] = (line[1] if line[1] in ['character','place'] else 'other')
			else:
				print('Benchmark file error: line '+str(i)+' ignored.')

processBook(bookfile, mwsite, focus, benchmark, debug, verbose, graphs)