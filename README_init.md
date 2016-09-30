# 3n-tools
## A set of simple novel authoring and analysis tools

### importFromHTML.php

Needs clean XML as input:

	tidy -asxhtml -bare -clean -utf8 -o ~/Desktop/book_dirty.htm -f ~/Desktop/html_errs.txt  ~/Desktop/book_ok.htm

Mandatory markup :

- chapter titles tagged with &lt;h2/&gt;

### genSplitFiles.php




### chapterFeatures.php




##Python dependencies

#matplotlib

~$  sudo apt-get install python-matplotlib

#scipy

~$  sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

#hunspell

~$	sudo apt-get update
~$	sudo apt-get install python2.7-dev
~$	sudo apt-get install libhunspell-dev
~$	sudo pip install hunspell

#treetagger3

http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/

#nltk (needed for treetagger3)

~$	sudo pip install -U nltk
~$	sudo pip install treetaggerwrapper

#treetagger-python

Update os.environ["TREETAGGER_HOME"] in characterStats.py
