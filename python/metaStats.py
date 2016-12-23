# coding: utf8

# Main script to generate meta data about characters
# start from pickled data
import pickle
import codecs
import sys
import getopt
from computeMeta import runMeta
import re

try:
    opts, args = getopt.getopt(sys.argv[1:], "b:", ["book="])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

book = ''

for opt, arg in opts:
    if opt in ("-b", "--book"):
        book = arg

directory = '../books-txt/predicted-data/'

def loadData(suffix):
    with open(directory + book + '-' + suffix + '.p', mode='rb') as f:
        data = pickle.load(f)
    return data

sents = loadData('sents')
char_sents = loadData('charsents')
char_list = loadData('chars')

# get job_labels
job_labels = {}
with codecs.open('../books-txt/books-txt/' + book + '.corr', 'r', 'utf8') as f:
    for i, raw_line in enumerate(f):
        line = raw_line.strip().split(u"\t")
        if len(line) > 3 and line[1] == 'character':
            job_labels[line[0]] = re.findall(r'\w+', unicode(line[3]), re.UNICODE)

runMeta(sents, char_sents, char_list, job_labels)
