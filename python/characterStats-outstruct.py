# coding: utf8
#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import division
import sys, glob, os
import getopt
reload(sys)
sys.setdefaultencoding('utf8')

import re
import nltk, hunspell
from nltk import word_tokenize
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
import numpy
from numpy import matrix

stopwords = [line.strip() for line in open("stopwords.txt", 'r')]

WORD_FREQUENCE_THRESHOLD = 10			# Names that are mentioned less than 10 times in the whole book will be ignored
MIN_NOUN_LENGTH = 2 					# Nouns shorter than that will be ignored
MINIMAL_MEAN_IDX = 3.0					# Names whose mean position in sentences are less than 3 will be ignored

### TOOLS ######################################################################################################################################################

_names = {}
_tagnums = []
compoundNouns = {}

hunspellstemmer = hunspell.HunSpell('dictionaries/fr-toutesvariantes.dic','dictionaries/fr-toutesvariantes.aff')
def stem(word):
	wstem = hunspellstemmer.stem(word)
	if len(wstem)>0:	# and wstem[-1] not in stopwords
		return wstem[-1]
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

#### BOT 4 #####################################################################################################################################################

#def tag2num(tag):
#	if tag in _tagnums:
#		return _tagnums.index(tag)
#	else:
#		_tagnums.append(tag)
#		return tag2num(tag)

def getSurroundings(array, idx):
	surroundings = []
	if (idx>1):
		surroundings.append(array[idx-2])
	else:
		surroundings.append('---')
	if (idx>0):
		surroundings.append(array[idx-1])
	else:
		surroundings.append('---')
	if (idx<len(array)-1):
		surroundings.append(array[idx+1])
	else:
		surroundings.append('---')
	if (idx<len(array)-2):
		surroundings.append(array[idx+2])
	else:
		surroundings.append('---')
	return surroundings

def structuralPredictor(word, indexesOfSentencesContainingWord, sentences, debug=False):
	if (debug):
		print "***** Structural results for "+word+" *****"
	scores = {"place":0,"character":0,"concept":0,"unknown":0}
	place_vs_char = 0.0			# Prediction score variable. If results turns out negative, we assume a place. If positive, a character.
	noise_score = 0.0			# Noise score. If positive, discard result
	positions = []
	for index in indexesOfSentencesContainingWord:
		sentence = sentences[index]
		verbs = []
		for wIdx, tag in enumerate(sentence["tags"]):
			if ("VER:" in tag):
				verbs.append(stem(sentence["words"][wIdx]))
		for wIdx, w in enumerate(sentence["words"]):
			if (w == word):
				positions.append(float(wIdx)/float(len(sentence["words"])))
				surroundings = [tag.split(':')[0] for tag in getSurroundings(sentence["tags"], wIdx)]
				if (debug):
					print word+" ["+sentence["tags"][wIdx]+"],"+",".join(surroundings)
				if ("VER" in surroundings):
					scores["character"] = scores["character"] + 0.5
				if ("VER" == surroundings[2]):
					scores["character"] = scores["character"] + 1.0
				if ("NAM" == surroundings[2]):
					scores["character"] = scores["character"] + 1.0
				if (surroundings[0]=="PRP" or surroundings[1]=="PRP"):
					scores["place"] = scores["place"] + 1.0
				if ("VER" == surroundings[1]):
					scores["place"] = scores["place"] + 0.5
				if (surroundings[1]=="DET"):
					scores["place"] = scores["place"] + 1.0
				if (surroundings[1]=="PRP" and surroundings[2]=="---"):
					scores["concept"] = scores["concept"] + 1.0
				if (surroundings[1]=="PUN"):								# noise detection (wrongly tokenized sentences).
					scores["unknown"] = scores["unknown"] + 1.0
				else:
					scores["unknown"] = scores["unknown"] - 1.0
				if (surroundings[0]=="---" and surroundings[1]=="---"):		# noise detection (wrongly tokenized sentences). If this happens, needs to be compensated 2 times
					scores["unknown"] = scores["unknown"] + 2.0
				else:
					scores["unknown"] = scores["unknown"] - 1.0
	maxV = 0
	maxK = ''
	scoresSum = 0
	for k,v in scores.iteritems():
		scoresSum = scoresSum+max(0, v)
		if (v>maxV):
			maxV = v
			maxK = k
	return [maxK, maxV/scoresSum if scoresSum>0 else 0]


### ######################################################################################################################################################

def tokenizeAndStructure(text):
	taggedText = tt.tag(text)
	tagstats = {}
	sentences = []
### Tokenize and detect uppercase words
#	sentences = nltk.sentences_tokenize(text)
	sent_words = []
	sent_tags = []
	for tag in taggedText:
		if (tag[1]=="SENT"):
			sent = {"words":sent_words,"tags":sent_tags,"nostop":[w for w in sent_words if w not in stopwords]}
			sentences.append(sent)
			sent_words = []
			sent_tags = []
		else:
			sent_words.append(tag[0])
			sent_tags.append(tag[1])
	return sentences


################################################################################################################################################################


def isUppercased(word):
	return word[0].isupper() # and len(word) > 1 and word[1].islower()

def detect_ucwords(fulltext, sentences, debug=False):
	ucwords = {}
	# Get all the uppercase words that are not leading sentences
	for sent in sentences:
		s = sent["words"]
#		s = [w for w in sent["words"] if not w in ['-',"'",'.']]			# Remove hyphens and other usual punctuation one may find in proper nouns
		if (len(s)>1):
#			grams5 = zip(s[1:-4], s[2:-3], s[3:-2], s[4:-1], s[5:])
#			grams4 = zip(s[1:-3], s[2:-2], s[3:-1], s[4:])
			grams3 = zip(s[1:-2], s[2:-1], s[3:])
			grams2 = zip(s[1:-1], s[2:])
			grams1 = zip(s[1:])
			sentUCWords = []
			for gram in grams3:
				if (isUppercased(gram[0]) and (gram[1] in ['de','d',"d'",'del','dal','da','di','della','du','des','la','le','of','van','von','vom','zu']) and isUppercased(gram[2])):
					sentUCWords.append(gram)
				elif (isUppercased(gram[0]) and isUppercased(gram[1]) and isUppercased(gram[2])):
					sentUCWords.append(gram)
			for gram in grams2:
				if (isUppercased(gram[0]) and isUppercased(gram[1])):
					sentUCWords.append(gram)
			for gram in grams1:
				if (isUppercased(gram[0]) and not (gram[0] in " ".join([w for _tuple in sentUCWords for w in _tuple]))):
					sentUCWords.append(gram)
			for gram in sentUCWords:
				gramStrRepresentation = " ".join(gram).replace("' ", "'")
				storeCount(ucwords, gramStrRepresentation)
	return ucwords


def joinCompoundNouns(fulltext, ucwords):
	for w in ucwords.keys():
		if " " in w or "'" in w:
			wjoined = w.replace(" ", "").replace(".", "").replace("'", "")
			fulltext = fulltext.replace(w, wjoined)
			compoundNouns[wjoined] = w
		else:
			compoundNouns[w] = w
	return fulltext

def confirmProperNoun(word, wmeanidx, wsentences):
	if (len(word) < MIN_NOUN_LENGTH):
		if debug:
			print "Word ignored: "+word+" len<"+str(MIN_NOUN_LENGTH)
		return False
	if (word.lower() in stopwords):
		if debug:
			print "Word ignored: "+word+" in stopwords"
		return False
	if (wmeanidx<MINIMAL_MEAN_IDX):
		if debug:
			print "Word ignored: "+word+"\t"+str(wmeanidx)
		return False
	wordTags = []
	for s in wsentences:
		for i, w in enumerate(s['words']):
			if w==word:
				wordTags.append(s['tags'][i])
	if not ('NAM' in wordTags or 'NOM' in wordTags):
		if debug:
			print "Word ignored: "+word+" tagged "+str(wordTags)
		return False
	return True

def getNounsSurroundigs(sentences, ucwords):
	wprev = {}
	wnext = {}
	wsent = {}
	wmeanidx = {}
	allucwords = ucwords.keys()
	for word in allucwords:
		wregex = re.compile(r'[^a-zA-Z]'+re.escape(word)+'[^a-zA-Z]')
		wcount = len(wregex.findall(fulltext))
		if (wcount>=WORD_FREQUENCE_THRESHOLD):
			ucwords[word] = wcount
			wprev[word] = {}
			wnext[word] = {}
			wsent[word] = []
			wmeanidx[word] = 0.0
			i = 0.0
			for sentIdx, sent in enumerate(sentences):
				for wpos, sent_word in enumerate(sent["words"]):
					if (sent_word==word):
						wsent[word].append(sentIdx)
			for sent in sentences:
				for wpos, sent_word in enumerate(sent["nostop"]):
					if (sent_word==word):
						wmeanidx[word] = (wmeanidx[word]*(i/(i+1.0)))+(float(wpos)/(i+1.0))
						if wpos>0:
							storeCount(wprev[word], stem(sent["nostop"][wpos-1]))
						if wpos<len(sent["nostop"])-1:
							storeCount(wnext[word], stem(sent["nostop"][wpos+1]))
						i = i+1.0
						break
		else:
			del ucwords[word]

	for word, meanidx in wmeanidx.iteritems():
		proxWords = {}
		for w in [w for _sub in [wprev[word].keys(), wnext[word].keys()] for w in _sub]:
			storeCount(proxWords, w)
		if (not confirmProperNoun(word, meanidx, [sentences[i] for i in wsent[word]])):
			del ucwords[word]
			del wprev[word]
			del wnext[word]
			del wsent[word]
#		else:
#			preditions = [positionPredictor(word, wsent[word], sentences, debug), localProximityPredictor(word, proxWords, debug), onlineDisambiguation(mwsite, word, word, debug)]
#			best = bestChoice(preditions)
#			print best[0]+"\t"+str(best[1])
		i=i+1
	return [wprev, wnext, wsent]

def get_stats(text, ucwords):
	words = nltk.word_tokenize(text)
	i = 0
	ci = len(words)-2
	while i<ci:
		if (words[i] in ucwords and words[i+1]=='de' and words[i+2] in ucwords):
			storeCount(_names, words[i]+' '+words[i+1]+' '+words[i+2])
			del words[i:i+3]
			ci=ci-3
		else:
			i=i+1
	i = 0
	ci = len(words)-1
	while i<ci:
		if (words[i] in ucwords and words[i+1] in ucwords):
			storeCount(_names, words[i]+' '+words[i+1])
			del words[i:i+2]
			ci=ci-2
		else:
			i=i+1
	for w in words:
		if (w in ucwords):
			storeCount(_names, w)

################################################################################################################################################################

try:
	opts, args = getopt.getopt(sys.argv[1:], "bcdfxw:v", ["help", "benchmarkfile=", "file=", "focus=", "mwclient=", "mincount="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err) # will print something like "option -a not recognized"
    sys.exit(2)
bookfile = ''
mwclienturl = ''
focus = ''
debug = False
benchmarkfile = False
mwsite = False
for o, a in opts:
	if o == "-d":
		debug = True
	elif o in ("-b", "--benchmarkfile"):
		benchmarkfile = a
	elif o in ("-h", "--help"):
		sys.exit()
	elif o in ("-f", "--file"):
		bookfile = a
	elif o in ("-x", "--focus"):
		focus = a
	elif o in ("-c", "--mincount"):
		WORD_FREQUENCE_THRESHOLD = int(a)
	elif o in ("-w", "--mwclient"):
		mwclienturl = a
		mwsite = mwclient.Site(mwclienturl)
		mwsite.compress = False
		readCachedResults(mwsite)
	else:
		assert False, "unhandled option"

benchmark = {}
if (benchmarkfile!=False):
	with open(benchmarkfile) as f:
		for i, raw_line in enumerate(f):
			line = unicode(raw_line.strip(), 'utf8').split("\t")
			if (len(line)>1):
				benchmark[line[0]] = line[1]

with open(bookfile) as f:
	chapters = []
	for i, raw_line in enumerate(f):
		line_split = raw_line.split("\t")
		chapter_number = line_split[0]
		line = line_split[1]
		line = line.replace("--", " ").replace("-", " ").replace("’", "'").replace("«", "").replace("»", "").replace("_", " ").strip()
		chapters.append(line)
	fulltext = " ".join(chapters)

	sentences = tokenizeAndStructure(fulltext)
	ucwords = detect_ucwords(fulltext, sentences, debug)
	fulltext = joinCompoundNouns(fulltext, ucwords)
	sentences = tokenizeAndStructure(fulltext)
	[wprev, wnext, wsent] = getNounsSurroundigs(sentences, ucwords)

	wtotcount = sum(ucwords.values())
	for word, wclass in benchmark.iteritems():
		if word in wsent:
			for index in wsent[word]:
				sentence = sentences[index]
				for wIdx, w in enumerate(sentence["words"]):
					if (w == word):
						surroundings = [tag.split(':')[0] for tag in getSurroundings(sentence["tags"], wIdx)]
						print wclass+"\t"+("\t".join(surroundings))


