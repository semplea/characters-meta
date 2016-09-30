# coding: utf8
#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import division
import sys, glob, os
import getopt
import math
import collections
import matplotlib.pyplot as plt
import re
from scipy.interpolate import spline
from matplotlib.legend_handler import HandlerLine2D


def keyForMaxValue(_dict):
	maxK = ''
	maxV = 0
	for k,v in _dict.iteritems():
		if (v>maxV):
			maxV = v
			maxK = k
	return maxK

try:
	opts, args = getopt.getopt(sys.argv[1:], "df", ["file="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err) # will print something like "option -a not recognized"
    sys.exit(2)

bookfile = ''
debug = False
for o, a in opts:
	if o in ("-f", "--file"):
		bookfile = a
	if o in ("-d"):
		debug = True

benchmark = {}
with open(bookfile+'.corr') as f:
	for i, raw_line in enumerate(f):
		line = unicode(raw_line.strip(), 'utf8').split(u"\t")
		benchmark[line[0]] = line[1]

finalWordClasses = {'character':[],'place':[]}
openerScores = {}
with open(bookfile+'-opener.txt') as f:
	for i, raw_line in enumerate(f):
		line = unicode(raw_line.strip(), 'utf8').split(u"\t")
		if not line[0] in openerScores:
			openerScores[line[0]] = {"place":0,"character":0,"other":0}
		if line[1]=="LOCATION":
			openerScores[line[0]]["place"] = openerScores[line[0]]["place"]+1
		elif line[1]=="PERSON" or line[1]=="ORGANIZATION":
			openerScores[line[0]]["character"] = openerScores[line[0]]["character"]+1
		else:
			openerScores[line[0]]["other"] = openerScores[line[0]]["other"]+1

allpredictions = {}

fulltext = ''
with open (bookfile+'.txt') as f:
    data = f.readlines()
allwords = len(re.findall(r'\w+', fulltext))
WORD_FREQUENCE_THRESHOLD = round(6+(allwords/10000)/4)

for p, values in openerScores.iteritems():
	kmv = keyForMaxValue(values)
	if (sum(values.values())>WORD_FREQUENCE_THRESHOLD):
		allpredictions[p] = [[kmv, values[kmv]/sum(values.values())]]

for wp in allpredictions.keys():
	for wb in benchmark.keys():
		if wb in wp and wb!=wp and not wp in benchmark.keys():
			print("WARN: "+wp+" = "+wb+"?")

if (debug):
	for p, values in allpredictions.iteritems():
		print(p+"\t"+values[0][0]+"\t"+str(values[0][1]))

weights = [1]
ncat = 0
unknown_words = []
correct_predictors = {}
ref_count = {}				# reference (number of words that should fall in each category, by predictor; last idx=best choice)
attr_count = {}				# attributions (number of words that fell in each category, by predictor; last idx=best choice)
for cat in ['character','place']:
	ncat = ncat+1
	correct_predictors[cat] = {}
	attr_count[cat] = {}
	ref_count[cat] = 0
	for pred_idx in range(0,len(weights)+1):
		correct_predictors[cat][pred_idx] = []
		attr_count[cat][pred_idx] = []
	for word, word_predictions in allpredictions.iteritems():
		if word in benchmark.keys():
			if (benchmark[word]==cat):										# we only consider the words from this effective category
				ref_count[cat] = ref_count[cat]+1
				for pred_idx, prediction in enumerate(word_predictions):
					correct_predictors[cat][pred_idx].append(1 if (prediction[0]==cat) else 0)
#					if (prediction[0]==cat):
#						print('OKK: '+word+' ('+cat+')')
#					else:
#						print('ERR: '+word+' ('+prediction[0]+' instead of '+cat+')')
				correct_predictors[cat][pred_idx+1].append(1 if (cat in finalWordClasses and word in finalWordClasses[cat]) else 0)
		else:
			unknown_words.append(word)										# we ignore words that are not listed in the benchmark file
		for pred_idx, prediction in enumerate(word_predictions):
			attr_count[cat][pred_idx].append(1 if prediction[0]==cat else 0)
		attr_count[cat][pred_idx+1].append(1 if (cat in finalWordClasses and word in finalWordClasses[cat]) else 0)
precision_by_classes = {}
recall_by_classes = {}
for pred_idx in range(0,len(weights)+1):
	precision_by_classes[pred_idx] = []
	recall_by_classes[pred_idx] = []
for cat, cat_count in ref_count.iteritems():
	for idx, pred_correct in correct_predictors[cat].iteritems():
		precision_by_classes[idx].append((sum(pred_correct)/sum(attr_count[cat][idx]) if sum(attr_count[cat][idx])>0 else 1))
		recall_by_classes[idx].append((sum(pred_correct)/cat_count if cat_count>0 else 0))
for idx in precision_by_classes.keys():
	print(str(idx)+"\t"+"P="+str(sum(precision_by_classes[idx])/ncat)+"\t"+"R="+str(sum(recall_by_classes[idx])/ncat))