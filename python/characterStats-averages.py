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

import re
import hunspell
import operator

import copy

import csv

import warnings
import pickle
warnings.simplefilter("error")

'''
export TREETAGGER_HOME='/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd'
'''
os.environ["TREETAGGER_HOME"] = "/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd"

sys.path.append('treetagger-python')
from treetagger3 import TreeTagger
tt = TreeTagger(encoding='utf-8',language='french')

#import nltk
#from nltk import word_tokenize
import nltk.app.wordnet_app as wnapp
from nltk.corpus import wordnet as wn

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
MINIMAL_MEDIAN_IDX = 1.0				# Names whose median position in sentences are less than 1 will be ignored
MAX_CHARACTERS_GRAPH = 50               # Absolute max number of characters considered for final graph
dynamicFrequenceFilter = False
averageMode = "meta"

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

### BOT 1 ######################################################################################################################################################

onlineDisambiguationClasses = {
			"character":["personnage","personnalité","prénom","animal","saint","naissance","décès","peuple","ethni","patronym"],
			"place":["lieu","ville","commune","pays","région","territoire","province","toponym","géographi","géolocalisé","maritime"],
			"other":["philosophi","divinité","dieu","religion","sigle","code","science","nombre","mathématique"]
			}
onlineDisambiguationStopwords = ["wikip","article","littérature","littéraire"]				# wikip:  We reached a general information page ("Wikipedia category", "Wikipedia disambiguation",...)

cachedResults = {}

def cachedOnlineDisambiguation(site_TODO, term):
	if term in cachedResults:
		return cachedResults[term]
	else:
		return False

def onlineDisambiguation(site, term, originalTerm=None, debug=False, iter=1, checkedClasses=[]):
	if (debug):
		print("***** Online results for "+term+" *****")
	if (originalTerm==None):
		originalTerm = term
	cachedResult = cachedOnlineDisambiguation(site, term)
	if (cachedResult!=False and not debug):
		return cachedResult
	else:
		if (site!=False):
			if (iter<5):
				pages = site.search(compoundNouns[originalTerm])
				for pageData in pages:
					page = site.Pages[pageData['title']]
					foundAtLeastOneCategory = False
					needToLookInText = False
					categoriesBasedDisambiguation = []
					for cat in page.categories():
						foundAtLeastOneCategory = True
						if (debug):
							print(compoundNouns[originalTerm]+" (as "+term+",iter="+str(iter)+")"+"\t"+pageData['title']+"\t"+cat.name)
						for k, cls in onlineDisambiguationClasses.iteritems():
							for cl in cls:
								if 'homonymie' in cat.name.lower():
									needToLookInText = True
								if cl in cat.name.lower():
									categoriesBasedDisambiguation.append([k, 0 if k=='unknown' else 1])
					if needToLookInText:
						fullText = page.text().lower()
						tot_all = 0			# all occurences of all classification words found
						fullTextClasses = []
						for k, cls in classes_local.iteritems():
							tot_cl = 0		# all occurences of the words cls corresponding to class k
							for cl in cls:
								tot_cl = tot_cl + fullText.count(cl)
							fullTextClasses.append([k, tot_cl])
							tot_all = tot_all+tot_cl
						if (len(fullTextClasses)>0):
							maxCountIdx = idxForMaxKeyValPair(fullTextClasses)		# Returns key yielding the highest count
							confidence = ((1/(iter*(len(checkedClasses)+1)))*(fullTextClasses[maxCountIdx][1]/tot_all) if tot_all>0 else 0)
#							if (confidence==0):
#								print originalTerm
#								print term
#								print str(checkedClasses)
#								print str(fullTextClasses)
							foundDisambiguation = [fullTextClasses[maxCountIdx][0], confidence]
							if (debug):
								print(originalTerm+" ("+term+") -- full text disambiguation results: "+"\t"+foundDisambiguation[0]+"\t"+str(foundDisambiguation[1])+"\t"+str(fullTextClasses))
							cachedResults[originalTerm] = foundDisambiguation
							updateCachedResults(site)
							return foundDisambiguation
					elif len(categoriesBasedDisambiguation)>0:
						bestCat = bestChoice(categoriesBasedDisambiguation, [], debug)
						for c in categoriesBasedDisambiguation:
							bestCatCount = sum([k[1] for k in categoriesBasedDisambiguation if k[0]==bestCat[0]])
						foundDisambiguation = [bestCat[0], bestCatCount/len(categoriesBasedDisambiguation)]
						if (bestCatCount==0):
							print originalTerm
							print term
							print bestCat[0]
							print str(categoriesBasedDisambiguation)
						if (debug):
							print(originalTerm+" ("+term+") -- cat based disambiguation results: "+"\t"+foundDisambiguation[0]+"\t"+str(foundDisambiguation[1])+"\t"+str(categoriesBasedDisambiguation))
						cachedResults[originalTerm] = foundDisambiguation
						updateCachedResults(site)
						return foundDisambiguation          #+" : "+cat.name


					for cat in page.categories():
						if (not cat.name in checkedClasses) and len([w for w in onlineDisambiguationStopwords if w in cat.name.lower()])==0:
							checkedClasses.append(cat.name)
							return onlineDisambiguation(site, cat.name, originalTerm, debug, iter+1, checkedClasses)
		elif (debug):
			print("Wiki Lookup disabled")
		return [u'unknown', 0]

def readCachedResults(site):
	if os.path.isfile(site.host+".csv"):
		for row in csv.reader(open(site.host+".csv")):
			cachedResults[row[0]] = [row[1], float(row[2])]

def updateCachedResults(site):
	w = csv.writer(open(site.host+".csv", "w"))
	for key, val in cachedResults.items():
	    w.writerow([key, val[0], val[1]])


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
		if "confident" in averageMode and p[1]!=0:
			predictions[idx][1] = 1
		# Check probabilities, remove predictions with p=0
		if p is None or len(p)!=2:
			print("prediction "+str(idx)+" invalid")
			print("    (len="+str(len(p))+"): ["+",".join(p)+"]")
			exit()
		elif p[1]==0:
			zeroProbas.append(idx)
		# Apply weighting
		elif (weights[idx]==0):
			zeroProbas.append(idx)
		elif (weights[idx]>1) and not p[1]==0:
			for n in range(1, weights[idx]):
				duplicates.append(p)
	for p in duplicates:
		predictions.append(p)
	zeroProbas.sort(reverse=True)
	for pIdx in zeroProbas:
		del predictions[pIdx]									# Remove predictions with probability 0
	if (len(predictions)>0):
		maxProbaIdx = idxForMaxKeyValPair(predictions)				# Returns key yielding the highest probabilities
	else:
		return ['unknown', 0]

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

def confirmProperNoun(word, wmedianidx, wsentences, ucwords):
	if (len(word) < MIN_NOUN_LENGTH) or (word.endswith("'") and len(word) < MIN_NOUN_LENGTH+1):
		if debug:
			print("Word ignored: "+word+"  [len<"+str(MIN_NOUN_LENGTH)+"]")
		return False
	if (word.lower() in stopwords):
		if debug:
			print("Word ignored: "+word+"  [in general stopwords"+"]")
		return False
	if (word in stopwords_pnouns):
		if debug:
			print("Word ignored: "+word+"  [in proper nouns stopwords"+"]")
		return False
	if (wmedianidx<MINIMAL_MEDIAN_IDX):
		if debug:
			print("Word ignored: "+word+"  [median idx="+str(wmedianidx)+"]")
		return False
	wordTags = []
	for s in wsentences:
		wordTags.append(s['tags'][s['words'].index(word)])
#		for i, w in enumerate(s['words']):
#			if w==word:
#				wordTags.append(s['tags'][i])
	if not ('NAM' in wordTags or 'NOM' in wordTags):
		if debug:
			print("Word ignored: "+word+"  [tagged "+str(wordTags)+"]")
		return False
	return True

def getIdxOfWord(ws, w):
	try:
		wIdx = ws.index(w)
	except:
		wIdx = -1
	return wIdx

def removeFalsePositives(sentences, wmedianidx, wprev, wnext, wsent, ucwords):
	for word, medianidx in wmedianidx.iteritems():
		proxWords = {}
		for w in [w for _sub in [wprev[word].keys(), wnext[word].keys()] for w in _sub]:
			storeCount(proxWords, w)
		rejected = False
		if (not confirmProperNoun(word, medianidx, [sentences[i] for i in wsent[word]], ucwords)):
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
	wmedidx = {}
	allucwords = ucwords.keys()
	for word in allucwords:
		wprev[word] = {}
		wnext[word] = {}
		wsent[word] = []
#		wmeanidx[word] = 0.0
		wPositions = []
		i = 0.0
		for sentIdx, sent in enumerate(sentences):
			wpos = getIdxOfWord(sent["nostop"], word)
			if (wpos > -1):
				wsent[word].append(sentIdx)
				wPositions.append(wpos)
#				wmeanidx[word] = (wmeanidx[word]*(i/(i+1.0)))+(float(wpos)/(i+1.0))		# update the mean position value (cumulative recompute)
				if wpos>0:
					storeCount(wprev[word], stem(sent["nostop"][wpos-1]))
				if wpos<len(sent["nostop"])-1:
					storeCount(wnext[word], stem(sent["nostop"][wpos+1]))
				i = i+1.0
		if (len(wPositions)>0):
			wmeanidx[word] = np.mean(np.array(wPositions))
			wmedidx[word] = np.median(np.array(wPositions))
		else:
			wmeanidx[word] = 0
			wmedidx[word] = 0
	return [wprev, wnext, wsent, wmeanidx, wmedidx]

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

'''
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
'''
################################################################################################################################################################

def computeKappa(mat, debug):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if debug:
        print n, "raters."
        print N, "subjects."
        print k, "categories."

    # Computing p[]
    p = [0.0] * k
    for j in xrange(k):
        p[j] = 0.0
        for i in xrange(N):
            p[j] += mat[i][j]
        p[j] /= N*n
    if debug: print "p =", p

    # Computing P[]
    P = [0.0] * N
    for i in xrange(N):
        P[i] = 0.0
        for j in xrange(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    if debug: print "P =", P

    # Computing Pbar
    Pbar = sum(P) / N
    if debug: print "Pbar =", Pbar

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    if debug: print "PbarE =", PbarE

    kappa = (Pbar - PbarE) / (1 - PbarE)
    if debug: print "kappa =", kappa

    return kappa

def checkEachLineCount(mat):
    """ Assert that each line has a constant number of ratings
        @param mat The matrix checked
        @return The number of ratings
        @throws AssertionError If lines contain different number of ratings """
    n = sum(mat[0])
    assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    return n



def processBook(bookfile, mwsite, focus, benchmark, debug=False, verbose=False, graphs=False):
	ucwords = {}
	sentences = []
	benchmarkValues = {"found":0,"correct":0,"predictors":[[],[],[],[],[],[],[],[],[]]}
	finalWordClasses = {'character':[],'place':[]}

	allpredictions = pickle.load( open( 'results-'+bookfile.split("/")[-1], "rb" ) )

	with open(bookfile) as f:
		t1 = np.arange(0.0, 5.0, 0.1)
		t2 = np.arange(0.0, 5.0, 0.02)
		if ("fixed2" in averageMode):
			weights = [2, 1, 1, 1, 1, 1]
		elif ("fixed3" in averageMode):
			weights = [3, 1, 1, 1, 1, 1]
		elif ("fixed4" in averageMode):
			weights = [4, 1, 1, 1, 1, 1]
		elif ("fixed5" in averageMode):
			weights = [5, 1, 1, 1, 1, 1]
		else:
			weights = [1, 1, 1, 1, 1, 1]


		if ("kappa" in averageMode):
			kappaMatrix = []
			kappaExcludedMatrix = []
			for pIdx in weights:
				kappaExcludedMatrix.append([])
			for w, pw in allpredictions.iteritems():
				kappaMatrix.append([len([x for x in pw if x[0]=='character']), len([x for x in pw if x[0]=='place']), len([x for x in pw if x[0]=='other']), len([x for x in pw if x[0]=='concept']), len([x for x in pw if x[0]=='unknown'])])
				for pIdx in range(0, len(weights)):
					pw_reduced = [p for p in pw]
					del pw_reduced[pIdx]
					kappaExcludedMatrix[pIdx].append([len([x for x in pw_reduced if x[0]=='character']), len([x for x in pw_reduced if x[0]=='place']), len([x for x in pw_reduced if x[0]=='other']), len([x for x in pw_reduced if x[0]=='concept']), len([x for x in pw_reduced if x[0]=='unknown'])])
			kappas = [computeKappa(kappaMatrix, debug)]
			for k in kappaExcludedMatrix:
				kappas.append(computeKappa(k, debug))
			if "kappad" in averageMode:
				print(str(kappas))
			discard = kappas.index(max(kappas))-1
			if (discard>0):
				weights[discard] = 0

#		Tweak weights according to allpredictions results. For instance, remove predictors whose % deviate too much from the others
		predStats = []
		if ("meta" in averageMode):
			charsPlacesRatio = []
			predictorRatioCounts = []
			entitiesCounts = []
			distances = []
			for i in range(0,len(weights)):
				entitiesCounts.append(len([1 for wp in allpredictions if allpredictions[wp][i][0]!='unknown']))
				charsPlacesRatio.append(len([1 for wp in allpredictions if allpredictions[wp][i][0]=='character'])/(len([1 for wp in allpredictions if allpredictions[wp][i][0]=='place'])+1))
				distances.append([])
				for j in range(0,len(weights)):
					dist = 0
					for word, wp in allpredictions.iteritems():
						if (wp[i][0]!=wp[j][0] and wp[i][1]>0 and wp[j][1]>0):
							dist = dist+1
					distances[i].append(dist)
			typicalEntCount=np.median(np.array(entitiesCounts))
			mean = np.mean(np.array(charsPlacesRatio))
			MAD = np.mean([abs(r - mean) for r in charsPlacesRatio])
			totAvg = sum([sum(x) for x in distances])/(len(weights)*len(weights))
			for pIdx, r in enumerate(charsPlacesRatio):
				predStats.append([])
				predStats[pIdx].append(entitiesCounts[pIdx])
				a = sum(distances[pIdx])/len(weights)
				predStats[pIdx].append(a)
				predStats[pIdx].append(r)
				predStats[pIdx].append(abs(r - mean))
				predStats[pIdx].append(entitiesCounts[pIdx]/typicalEntCount)
				predStats[pIdx].append(np.mean(np.array([allpredictions[pw][pIdx][1] for pw in allpredictions])))
#				if (pIdx>0 and (a > 1.4826*totAvg or a < totAvg/1.4826) and abs(r - mean) > 1.4826*MAD):
#					if ("metad" in averageMode):
#						print(str(pIdx)+" --> "+str(a)+"/"+str(totAvg)+"\t"+str(entitiesCounts[pIdx])+"/"+str(typicalEntCount))
#					weights[pIdx] = 0
				if (entitiesCounts[pIdx]/typicalEntCount > 1.15):
					weights[pIdx] = 0
			if ("metad" in averageMode):
				print(str(weights))

			'''
			charsPlacesRatio = []
			predictorRatioCounts = []
			entitiesCounts = []
			for pIdx in range(0,len(weights)):
				entitiesCounts.append(len([1 for wp in allpredictions if allpredictions[wp][pIdx][0]!='unknown']))
				charsPlacesRatio.append(len([1 for wp in allpredictions if allpredictions[wp][pIdx][0]=='character'])/(len([1 for wp in allpredictions if allpredictions[wp][pIdx][0]=='place'])+1))
			typicalEntCount=np.median(np.array(entitiesCounts))
			mean = np.mean(np.array(charsPlacesRatio))
			MAD = np.mean([abs(r - mean) for r in charsPlacesRatio])
			for rIdx, r in enumerate(charsPlacesRatio):
				if (debug):
					print(str(rIdx)+":"+str(r)+":"+str(entitiesCounts[rIdx]))
				if (rIdx>0 and entitiesCounts[rIdx]<typicalEntCount/1.4826):		# abs(r - mean) > 1.4826*MAD
					weights[rIdx] = 0
			if (debug):
				print('Adjusted predictors weights: '+str(weights))
			'''


		ucwords = allpredictions.keys()
		sorted_ucw = sorted(ucwords, key=operator.itemgetter(1))

		for word in sorted_ucw:
			if (debug): print(word)
			if (not "eval" in averageMode):
				best = bestChoice(allpredictions[word], weights, debug)
				if (best[0] in finalWordClasses.keys()):
					finalWordClasses[best[0]].append(word)
				if len(benchmark)>0:
					if (word in benchmark.keys()):
						benchmarkValues["found"] = benchmarkValues["found"]+1
						if (benchmark[word] == best[0]):
							benchmarkValues["correct"] = benchmarkValues["correct"]+1
						for idx, p in enumerate(allpredictions[word]):
							benchmarkValues["predictors"][idx].append(1 if p[0]==benchmark[word] else 0)
						if verbose:
							print(word+"\t"+best[0]+"\t"+str(benchmark[word] == best[0])+"\t"+str(allpredictions[word]))
					else:
						if verbose:
							print(word+"\t"+best[0]+"\tN/A\t"+str(allpredictions[word]))
		if len(benchmark)>0:
			if verbose:
				print('=== PERFORMANCE EVALUATION ==============================')
			ncat = 0
			unknown_words = []
			correct_predictors = {}
			ref_count = {}				# reference (number of words that should fall in each category, by predictor; last idx=best choice)
			attr_count = {}				# attributions (number of words that fell in each category, by predictor; last idx=best choice)
			nbWeights = len(weights) if ("eval" in averageMode) else len(weights)+1
			for cat in ['character','place']:
				ncat = ncat+1
				correct_predictors[cat] = {}
				attr_count[cat] = {}
				ref_count[cat] = 0
				for pred_idx in range(0,nbWeights):
					correct_predictors[cat][pred_idx] = []
					attr_count[cat][pred_idx] = []
				for word, word_predictions in allpredictions.iteritems():
					if word in benchmark.keys():
						if (benchmark[word]==cat):										# we only consider the words from this effective category
							ref_count[cat] = ref_count[cat]+1
							for pred_idx, prediction in enumerate(word_predictions):
								correct_predictors[cat][pred_idx].append(1 if (prediction[0]==cat) else 0)
							if (not "eval" in averageMode):
								correct_predictors[cat][pred_idx+1].append(1 if (cat in finalWordClasses and word in finalWordClasses[cat]) else 0)
					else:
						unknown_words.append(word)										# we ignore words that are not listed in the benchmark file
					for pred_idx, prediction in enumerate(word_predictions):
						attr_count[cat][pred_idx].append(1 if prediction[0]==cat else 0)
					if (not "eval" in averageMode):
						attr_count[cat][pred_idx+1].append(1 if (cat in finalWordClasses and word in finalWordClasses[cat]) else 0)
			precision_by_classes = {}
			recall_by_classes = {}
			for pred_idx in range(0, nbWeights):
				precision_by_classes[pred_idx] = []
				recall_by_classes[pred_idx] = []
			for cat, cat_count in ref_count.iteritems():
				for idx, pred_correct in correct_predictors[cat].iteritems():
					precision_by_classes[idx].append((sum(pred_correct)/sum(attr_count[cat][idx]) if sum(attr_count[cat][idx])>0 else 1))
					recall_by_classes[idx].append((sum(pred_correct)/cat_count if cat_count>0 else 0))
			missing_words = list(set(benchmark.keys()) - set([w for ws in finalWordClasses.values() for w in ws]))
			if (verbose):
				if (len(unknown_words)>0):
					print("! UNKNOWN WORDS: "+(", ".join(set(unknown_words))))
				if (len(missing_words)>0):
					print("! MISSING WORDS: "+(", ".join(missing_words)))
				for idx in precision_by_classes.keys():
					print(str(idx)+"\t"+"P="+str(sum(precision_by_classes[idx])/ncat)+"\t"+"R="+str(sum(recall_by_classes[idx])/ncat))
				print('===========================================================')

		if (mwsite!=False):
			updateCachedResults(mwsite)
		if len(benchmark)>0:
			if (benchmarkValues["found"]>0):
				if verbose:
					print("========== BENCHMARK RESULTS ============")
					print("Overall score: "+str(benchmarkValues["correct"]/benchmarkValues["found"]))
	#		for idx, b in enumerate([b for b in benchmarkValues["predictors"] if len(b)>0]):
	#			print("Prediction #"+str(idx+1)+": "+str( (sum(b)/len(b))))

		# These are the colors that will be used in the plot
	#	color_sequence = ['#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF80', '#FFFF00', '#FF5005', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB', '#426600', '#FF0010', '#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31', '#2BCE48', '#FFCC99', '#808080']
	#	color_sequence = ['#1f77b4', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
		color_sequence = ["#000000", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#FFFF00", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",  "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800", "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",  "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D", "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176", "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5", "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4", "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01", "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C", "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183", "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433", "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F", "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E", "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F", "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00", "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",  "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25", "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]	# http://godsnotwheregodsnot.blogspot.ru/

		nbCharacters = len(finalWordClasses['character'])
		if graphs:
			if (nbCharacters>0):
				if (nbCharacters>MAX_CHARACTERS_GRAPH):
					finalWordClasses['character'] = [w[0] for w in sorted_ucw if w[0] in finalWordClasses['character']][0:MAX_CHARACTERS_GRAPH]

				fig, ax = plt.subplots(1, 1, figsize=(18, 10))
				ax.get_xaxis().tick_bottom()
				ax.get_yaxis().tick_left()
				plt.xticks(range(0, len(chapters)*nbCharacters, nbCharacters), chapters.keys(), fontsize=10, rotation=90)
				plt.yticks(range(0, len(finalWordClasses['place']), 1), finalWordClasses['place'], fontsize=10)

				chaptersPlaces = {}
				for cnum, chapsentencesidx in chapters.iteritems():
					chapterPlaces = {}
					for w2idx, w2 in enumerate(finalWordClasses['place']):
						chapterPlaces[w2] = [y for z in [sentences[idx]['words'] for idx in chapsentencesidx] for y in z].count(w2)
					chapterPlace = keyForMaxValue(chapterPlaces)
					chaptersPlaces[cnum] = (finalWordClasses['place'].index(chapterPlace) if chapterPlace!='' else -1)

				for w1idx, w1 in enumerate(finalWordClasses['character']):
					xs = []
					ys = []
					cidx = 0
					for cnum, chapsentencesidx in chapters.iteritems():
						if (chaptersPlaces[cnum]!=-1):
							intersect = list(set(wsent[w1]) & set(chapsentencesidx))
							if len(intersect)>0:
								xs.append(cidx*nbCharacters+w1idx)
								ys.append(chaptersPlaces[cnum])
						cidx = cidx+1
					# if the considered charactered is quoted more than once in this chapter, we add it to the list
					if (len(xs)>1):
						xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys), key=operator.itemgetter(0), reverse=False))
						plt.plot(xs_sorted, ys_sorted, 'o-', lw=2, color=color_sequence[w1idx % len(color_sequence)], label=w1, markersize=8, markeredgewidth=0.0, alpha=0.7)
			#			x_linspace = np.linspace(min(xs_sorted), max(xs_sorted), num=100, endpoint=True)
			#			xs_sorted = np.array(xs_sorted)
			#			ys_sorted = np.array(ys_sorted)
			#			if (len(xs_sorted)>3):
			#				f2 = interp1d(xs_sorted, ys_sorted, kind='cubic')
			#			else:
			#				f2 = interp1d(xs_sorted, ys_sorted, kind='slinear')
			#			plt.plot(xs_sorted, ys_sorted, 'o', x_linspace, f2(x_linspace), '-', lw=2, color=color_sequence[w1idx % len(color_sequence)], label=w1, markersize=8, markeredgewidth=0.0, alpha=0.7)

				ax = plt.subplot(111)
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
				plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
				plt.show()

				print("__________ Characters graph ______________")
				print("graph characters {")
				print("   "+"graph[layout=neato, splines=true, overlap=prism];")
				for w1 in finalWordClasses['character']:
					for w2 in [w for w in finalWordClasses['character'] if w!=w1]:
						intersect = list(set(wsent[w1]) & set(wsent[w2]))
						if (len(intersect)>0):
							print("   "+w1+" -- "+w2+" [len="+str(1+1/len(intersect))+", penwidth="+str(math.sqrt(len(intersect)))+"];")		#weight="+str(len(intersect))+",
				print("}")

				print("__________ Bipartite graph ______________")
				print("graph bip {")
				print("   "+"graph[layout=neato, splines=true, overlap=prism];")
				print('   "'+'","'.join(finalWordClasses['place'])+'"[shape=box,style=filled];')
				relations = {}
				for w1 in finalWordClasses['character']:
					print('   "'+w1+'"[fontsize='+str(round(10+math.log(ucwords[w1])))+'];');
					relations[w1] = {}
					for cnum, chapsentencesidx in chapters.iteritems():
						if (chaptersPlaces[cnum]!=-1):
							if len(list(set(wsent[w1]) & set(chapsentencesidx)))>0:
								storeCount(relations[w1], finalWordClasses['place'][chaptersPlaces[cnum]])
				for c, r in relations.iteritems():
					for p, v in r.iteritems():
						print('   "'+c+'"--"'+p+'"[len='+str(1+(1/v))+', penwidth='+str(math.sqrt(v))+'];')
				print("}")
			else:
				print("Plot impossible: no character found.");


		if (len(benchmark)>0):
#			print(bookfile+"\t"+str(sum(precision_by_classes[len(precision_by_classes)-1])/ncat)+"\t"+str(sum(recall_by_classes[len(recall_by_classes)-1])/ncat))
			benchStr = bookfile+"\t"+str()+"\t"+str(WORD_FREQUENCE_THRESHOLD)   #+"\t"+str(ucwtotcount)+"\t"+str(ucwtotunique)+"\t"+str(sorted_ucw[0][1])+"\t"+str(len(re.findall(r'\w+', fulltext)))
			ps = []
			rs = []
			for idx in precision_by_classes.keys():
				p = sum(precision_by_classes[idx])/ncat
				ps.append(p)
				r = sum(recall_by_classes[idx])/ncat
				rs.append(r)
				f1 = 2*(p*r)/(p+r)
				benchStr = benchStr+"\t"+'{:0.3f}'.format(p)+"\t"+'{:0.3f}'.format(r)+"\t"+'{:0.4f}'.format(f1)
#			benchStr = benchStr+"\n--> Averages: "+str(sum(ps)/len(ps))+" / "+str(sum(rs)/len(rs))
			if ("eval" in averageMode):
				for idx in precision_by_classes.keys():
					p = sum(precision_by_classes[idx])/ncat
					r = sum(recall_by_classes[idx])/ncat
					f1 = 2*(p*r)/(p+r)
					cat = "C" if (p < 0.7) else "B" if (p < 0.9) else "A"
					print(str(idx)+"\t"+'{:0.3f}'.format(p)+"\t"+'{:0.3f}'.format(r)+"\t"+'{:0.4f}'.format(f1)+"\t"+cat+"\t"+str(predStats[idx][0])+"\t"+str(predStats[idx][1])+"\t"+str(predStats[idx][2])+"\t"+str(predStats[idx][3])+"\t"+str(predStats[idx][4])+"\t"+str(predStats[idx][5]))
			else:
				print(benchStr)

################################################################################################################################################################

try:
	opts, args = getopt.getopt(sys.argv[1:], "bcdfgxw:v", ["help", "benchmark", "graphs", "file=", "focus=", "mwclient=", "mincount=", "avg="])
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
	elif o in ("-a", "--avg"):
		averageMode = a
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

'''
	fig, ax = plt.subplots(1, 1, figsize=(12, 8))
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.xticks(range(0, len(sentences), int(round(len(sentences)/20))), fontsize=10)
	plt.yticks(range(0, len(finalWordClasses['place']), 1), finalWordClasses['place'], fontsize=10)
	for w1idx, w1 in enumerate(finalWordClasses['character']):
		xs = []
		ys = []
		for w2idx, w2 in enumerate(finalWordClasses['place']):
			intersect = list(set(wsent[w1]) & set(wsent[w2]))
			for i in intersect:
				xs.append(i)
				ys.append(w2idx)
		if (len(xs)>0):
			xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys), key=operator.itemgetter(0), reverse=True))
			line, = plt.plot(xs_sorted, ys_sorted, 'o-', lw=1, color=color_sequence[w1idx % len(color_sequence)], label=w1, markersize=10, markeredgewidth=0.0)
	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
	plt.show()
'''

'''
	print("Found "+str(len(ucwords))+" names. Starting disambiguation...")
	for i, chap in enumerate(chapters):
		sys.stdout.write('%d' % i)
		sys.stdout.flush()
		sys.stdout.write('\r')
		get_stats(chap, ucwords)
	names_view = [ (v,k) for k,v in _names.iteritems() ]
	names_view.sort(reverse=True)
	countMax = -1
	for key, value in names_view:
		if (countMax==-1):
			countMax = key
		wikiClass = ""
		if (key > countMax/10):
			wikiClass = onlineDisambiguation(mwsite, value, value, debug)
'''
