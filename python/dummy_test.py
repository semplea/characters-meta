#!/usr/bin/env python
import os
import sys
import nltk
import numpy
import scipy

os.environ["TREETAGGER_HOME"] = "/home/alexis/Documents/EPFL/MS3/Project/python/TreeTagger/cmd"

sys.path.append('/home/alexis/Documents/EPFL/MS3/Project/python/treetagger-python/')

from treetagger3 import TreeTagger
tt = TreeTagger(encoding='utf-8', language='french')
print(str(tt.tag('A simple sentence')))
print(tt)

