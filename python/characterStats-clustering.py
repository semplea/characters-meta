#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
reload(sys)
sys.setdefaultencoding('utf8')

import nltk, hunspell
from nltk import word_tokenize
import operator

import urllib
import mwclient

import scipy
from scipy.cluster.vq import kmeans2
from numpy import matrix

'''
export TREETAGGER_HOME='/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd'
'''
os.environ["TREETAGGER_HOME"] = "/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd"

sys.path.append('treetagger-python')
from treetagger3 import TreeTagger
tt = TreeTagger(encoding='utf-8', language='french')

stopwords = [line.strip() for line in open("stopwords.txt", 'r')]

WORD_FREQUENCE_THRESHOLD = 10			# Names that are mentioned less than 10 times in the whole book will be ignored
MINIMAL_MEAN_IDX = 0.0					# Names whose mean position in sentences are less than 3 will be ignored

################################################################################################################################################################

_names = {}


def isUppercased(word):
	return len(word) > 1 and word[0].isupper() and word[len(word)-1].islower()


def storeCount(array, key):
	if key in array:
		array[key] += 1
	else:
		array[key] = 1

def detect_ucwords(fulltext, sentences, multiWords=False, debug=False):
	ucwords = {}
	# Get all the uppercase words that are not leading sentences
	for sent in sentences:
		s = sent["words"]
		if (len(s)>1):
			grams3 = zip(s[1:-2], s[2:-1], s[3:])
			grams2 = zip(s[1:-1], s[2:])
			grams1 = zip(s[1:])
			sentUCWords = []
			if (multiWords):
				for gram in grams3:
					if (isUppercased(gram[0]) and not gram[0].lower() in stopwords and (gram[1] in ['de',"d'",'d','d’','del','dal','da','di','della','du','des','-','of','van','von','vom','zu']) and isUppercased(gram[2])):
						sentUCWords.append(gram)
				for gram in grams2:
					if (isUppercased(gram[0]) and not gram[0].lower() in stopwords and isUppercased(gram[1])):
						sentUCWords.append(gram)
			for gram in grams1:
				if (isUppercased(gram[0]) and not (gram[0] in " ".join([w for _tuple in sentUCWords for w in _tuple]))):
					sentUCWords.append(gram)
			for gram in sentUCWords:
				storeCount(ucwords, " ".join(gram))
	return ucwords

def joinCompoundNouns(fulltext, ucwords):
	for w in ucwords.keys():
		if " " in w:
			wjoined = w.replace(" ", "")
			fulltext = fulltext.replace(w, wjoined)
	return fulltext

def getNounsSurroundigsNoStemming(sentences, ucwords):
	wprev = {}
	wnext = {}
	wsent = {}
	wmeanidx = {}
	allucwords = ucwords.keys()
	for word in allucwords:
		wcount = fulltext.count(word)
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
							storeCount(wprev[word], sent["nostop"][wpos-1])
						if wpos<len(sent["nostop"])-1:
							storeCount(wnext[word], sent["nostop"][wpos+1])
						i = i+1.0
						break
		else:
			del ucwords[word]

	for word, meanidx in wmeanidx.iteritems():
		proxWords = {}
		for w in [w for _sub in [wprev[word].keys(), wnext[word].keys()] for w in _sub]:
			storeCount(proxWords, w)
		i=i+1

	autoCluster(ucwords, wmeanidx)
	exit()

	return [wprev, wnext, wsent]

def autoCluster(ucwords, wmeanidx):
	to_cluster = []
	for word, meanidx in wmeanidx.iteritems():
		to_cluster.append([meanidx])		#, ucwords[word]
	clust = kmeans2(matrix(to_cluster), 2)
	i=0
	falsePositiveIdx = 0
	if (clust[0][0]>clust[0][1]):
		falsePositiveIdx = 1
	for word, meanidx in wmeanidx.iteritems():
		if (clust[1][i]==falsePositiveIdx):
			print "DEL:\t"+word+"\t"+str(meanidx)+"\t"+str(ucwords[word])
			del ucwords[word]
		else:
			print "OK: \t"+word+"\t"+str(meanidx)+"\t"+str(ucwords[word])
		i=i+1

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


classes = {"character":["personnage","personnalité","prénom","animal","divinité","dieu","saint","naissance","décès"],"place":["lieu","ville","pays","géographique","toponyme"],"unknown":["wikip"]}				# wikip:  We reached a general information page ("Wikipedia category", "Wikipedia disambiguation",...)

def classify(term):
	pages = site.search(term)
	found = ""
	for pageData in pages:
		page = site.Pages[pageData['title']]
		for cat in page.categories():
			if found!="":
				break
			for k, cls in classes.iteritems():
				for cl in cls:
					if cat.name.lower().find(cl) > -1:
						return k         #+" : "+cat.name
						break
#		if found=="":
		for cat in page.categories():
			if not cat.name in checkedClasses:
				checkedClasses.append(cat.name)
				found = classify(cat.name)
				if found!="":
					return found



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

### ######################################################################################################################################################

with open(sys.argv[1]) as f:
	chapters = []
	preprocessingReplace = {":":".", "--":"", "-":" - ", "’":"'", "_":"", "«":"", "»":""}
	for i, raw_line in enumerate(f):
		line_split = raw_line.split("\t")
		chapter_number = line_split[0]
		line = line_split[1]
		for p, r in preprocessingReplace.iteritems():
			line = line.replace(p,r)
		chapters.append(line.strip())
	fulltext = " ".join(chapters)
	sentences = tokenizeAndStructure(fulltext)
	ucwords = detect_ucwords(fulltext, sentences, True, False)
	fulltext = joinCompoundNouns(fulltext, ucwords)
	sentences = tokenizeAndStructure(fulltext)
	ucwords = detect_ucwords(fulltext, sentences, False, False)
	[wprev, wnext, wsent] = getNounsSurroundigsNoStemming(sentences, ucwords)


	print "Found "+str(len(ucwords))+" names. Starting disambiguation..."
	for i, chap in enumerate(chapters):
		sys.stdout.write('%d' % i)
		sys.stdout.flush()
		sys.stdout.write('\r')
		get_stats(chap, ucwords)
	names_view = [ (v,k) for k,v in _names.iteritems() ]
	names_view.sort(reverse=True)
	countMax = -1
	site = mwclient.Site(sys.argv[2])
	site.compress = False
	for key, value in names_view:
		if (countMax==-1):
			countMax = key
		wikiClass = ""
		if (key > countMax/10):
			checkedClasses = []
			wikiClass = classify(value)
			wikiClassConfidence = str(len(checkedClasses))
		print str(key)+"\t"+value+"\t"+wikiClass+"\t"+wikiClassConfidence
