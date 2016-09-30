#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
reload(sys)
sys.setdefaultencoding('utf8')

import nltk, hunspell
from nltk import word_tokenize
import operator

'''
export TREETAGGER_HOME='/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd'
'''
os.environ["TREETAGGER_HOME"] = "/Users/cybor/Sites/3n-tools/python/tree-tagger/cmd"

sys.path.append('treetagger-python')
import treetagger
from treetagger import TreeTagger
tt = TreeTagger(encoding='utf-8',language='french')


import urllib
import mwclient

################################################################################################################################################################

_names = {}


def storeName(name):
	if name in _names:
		_names[name] += 1
	else:
		_names[name] = 1


def split_line(text):
	tags = tt.tag(text)
	i = 0
	ci = len(tags)-2
	while i<ci:
		if (tags[i][1]=='NAM' and tags[i+1][2]=='de' and tags[i+2][1]=='NAM'):
			storeName(tags[i][0]+' '+tags[i+1][0]+' '+tags[i+2][0])
			del tags[i:i+3]
			ci=ci-3
		else:
			i=i+1
	i = 0
	ci = len(tags)-1
	while i<ci:
		if (tags[i][1]=='NAM' and tags[i+1][1]=='NAM'):
			storeName(tags[i][0]+' '+tags[i+1][0])
			del tags[i:i+2]
			ci=ci-2
		else:
			i=i+1
	for tag in tags:
		if (tag[1]=='NAM'):
			storeName(tag[0])
'''
	sentences = nltk.sent_tokenize(text)
	# for each word in the line:
	for sent in sentences:
		words = word_tokenize(sent)
		for word in words:
			if len(word) > 0:
				if word[0].isupper():
					print(word)
'''


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



with open(sys.argv[1]) as f:
	for i, line in enumerate(f):
		sys.stdout.write('%d' % i)
		sys.stdout.flush()
		sys.stdout.write('\r')
		clean_text = line.replace("’", "'").replace("_", "")
		split_line(clean_text)
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
