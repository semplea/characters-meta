# coding: utf-8

from gensim.models import Word2Vec
from read_data import removeAccents

class MyModel:
    """
    Wrapper for the gensim Word2Vec model.
    """

    def __init__(self):
        self.model_bin = 'frWac_postag_no_phrase_700_skip_cut50.bin'
        self.model = Word2Vec.load_word2vec_format(self.model_bin, binary=True)

    def compareWords(self, w1, w2):
        """
        Return tuple of words with similarity (cosine distance) between two word embeddings.
        Note, words need to be given type. Assume nouns.
        """
        try:
            return self.model.similarity(w1 + '_n', w2 + '_n')
        except KeyError:
            return None



# visualize with t-SNE http://homepage.tudelft.nl/19j49/t-SNE.html?
# model['computer'] gives raw numpy vector of word 'computer'
