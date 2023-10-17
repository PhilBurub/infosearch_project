import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as simfunc
from matrices import Matrices
from time import time
from sys import getsizeof


class Search:
    def __init__(self, path, bm25, w2v, ft, ai):
        self.matrices = Matrices(path, bm25, w2v, ft, ai)

    def bm25_scores(self, query):
        bm25_query = self.matrices.lemmatize(query)
        return self.matrices.matrix_bm25.get_scores(bm25_query)

    def w2v_scores(self, query):
        w2v_query = self.matrices.tokenize(query)
        text_query = []
        for word in w2v_query:
            if word in self.matrices.w2v:
                text_query.append(self.matrices.w2v[word].reshape(1, -1))
        if len(text_query) == 0:
            query_vec = np.zeros((1, 100))
        else:
            query_vec = np.concatenate(text_query, axis=0).sum(axis=0).reshape(1, -1)
        return simfunc(self.matrices.matrix_w2v, query_vec)[:, 0]

    def ft_scores(self, query):
        ft_query = self.matrices.tokenize(query)
        text_query = []
        for word in ft_query:
            if word in self.matrices.ft:
                text_query.append(self.matrices.ft[word].reshape(1, -1))
        if len(text_query) == 0:
            query_vec = np.zeros((1, 300))
        else:
            query_vec = np.concatenate(text_query, axis=0).sum(axis=0).reshape(1, -1)
        return simfunc(self.matrices.matrix_ft, query_vec)[:, 0]

    def ai_scores(self, query):
        query_vec = self.matrices.ai.get_embeddings(query)
        return simfunc(self.matrices.matrix_ai, query_vec)[:, 0]

    def gettop(self, query, method, num):
        starttime = time()
        if method == 'bm25':
            scores = self.bm25_scores(query)
            size = getsizeof(self.matrices.matrix_bm25)
        elif method == 'word2vec':
            scores = self.w2v_scores(query)
            size = getsizeof(self.matrices.w2v) + getsizeof(self.matrices.matrix_w2v)
        elif method == 'fasttext':
            scores = self.ft_scores(query)
            size = getsizeof(self.matrices.ft) + getsizeof(self.matrices.matrix_ft)
        elif method == 'ai':
            scores = self.ai_scores(query)
            size = getsizeof(self.matrices.ai) + getsizeof(self.matrices.matrix_ai)
        texts = []
        i = 1
        for text_id in np.argsort(scores)[-1:-(num + 1):-1]:
            texts.append(f'{i}. {self.matrices.gettexts()[text_id]}')
            i += 1
        texts.extend(['---', f'Результат был получен за {round(time() - starttime, 3)} секунд.',
                      f'Затраты памяти составляют {size} байт.', ''])
        return texts
