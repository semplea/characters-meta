#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Invocation:    python lookup.py fr.wikipedia.org "Gillenormand"

import sys, os
import mwclient

reload(sys)
sys.setdefaultencoding('utf8')

classes = {"character":["personnage","personnalité","prénom","animal","divinité","dieu","saint","naissance","décès"],"place":["lieu","ville","pays","géographique","toponyme"],"unknown":["wikip"]}				# wikip:  We reached a general information page ("Wikipedia category", "Wikipedia disambiguation",...)
checkedClasses = []

def classify(term):
	pages = site.search(term)
	found = ""
	for pageData in pages:
		page = site.Pages[pageData['title']]
		for cat in page.categories():
			if found!="":
				break
			print cat.name
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

site = mwclient.Site(sys.argv[1])
site.compress = False

#print 'Running configured site', sys.argv[1]
#print 'Site has writeapi:', getattr(site, 'writeapi', False)

print classify(sys.argv[2])+":"+str(len(checkedClasses))