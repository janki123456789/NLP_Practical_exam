
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

class AverageWord2VecTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model_path='word2vec_skipgram.model'):
        self.model_path = model_path

    def fit(self, X, y=None):
        self.w2v = Word2Vec.load(self.model_path)
        return self

    def transform(self, X):
        result = []

        for text in X:
            words = str(text).split()

            vectors = [
                self.w2v.wv[word]
                for word in words
                if word in self.w2v.wv
            ]

            if len(vectors) == 0:
                result.append(np.zeros(100))
            else:
                result.append(np.mean(vectors, axis=0))

        return np.array(result)
