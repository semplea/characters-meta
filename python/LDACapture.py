# encoding: utf8
#!usr/bin/python

from gensim import models, corpora
from computeMeta import getSoloSents
import codecs


def runLDA(book, sentences, wsent, char_list, solo=False):
    stopwords = set(
    	line.strip() for line in codecs.open("classifiersdata/stopwords.txt", 'r', 'utf8') if line != u'')

    character = 'Passepartout'
    # Get sentences for only character
    char_sent = wsent[character]
    if solo:
        char_sent, _ = getSoloSents(character, wsent, char_list, char_list.index(character))

    # Get LDA for character
    charLDA(sentences, stopwords, character=character, char_sent=char_sent)
    # charLDA(sentences, stopwords)


def charLDA(sentences, stopwords, character='', char_sent=[]):
    doc = []
    idx = [i for i, w in enumerate(sentences)]
    if char_sent:
        idx = char_sent
    for s in idx:
        # Replace <unknown> tags by unlemmatized word
        lemma_sent = [w.lower() if w != '<unknown>' else sentences[s]['words'][i].lower() for i, w in enumerate(sentences[s]['lemma'])]
        # Keep only nouns and names
        lemma_sent = [w for i, w in enumerate(lemma_sent) if sentences[s]['tags'][i] in ['NOM', 'NAM']]
        # Remove stopwords and character name
        lemma_sent = [w for w in lemma_sent if w not in [character]]
        lemma_sent = [w for w in lemma_sent if len(w) > 2]

        doc.append(lemma_sent)


    # texts = [[word for word in document.lower().split()] for document in documents]
    # dictionary = corpora.Dictionary(texts)

    # Dictionary is list of pairs (token, count) generated from document
    dictionary = corpora.Dictionary(doc)
    dictionary.filter_extremes(no_below = 3, no_above=0.5)

    corpus = [dictionary.doc2bow(d) for d in doc]
    lda = models.LdaModel(corpus, num_topics=10, id2word = dictionary)
    print character
    for i in range(0, lda.num_topics):
        print "Topic", str(i), "-", lda.print_topic(i)
