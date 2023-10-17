from gensim.models import KeyedVectors
from rank_bm25 import BM25Okapi
import numpy as np
from corpora_build import Corpora
from tqdm import tqdm
from ai_embedding import AIEmbed

class Matrices:
    def __init__(self, path, bm25, w2v, ft, ai):
        self.corpora = Corpora(path)
        total = sum([bm25, w2v, ft, ai])
        i = 1
        if bm25:
            print(f'Идет векторизация корпуса... ({i}/{total}): BM25')
            self.shape_bm25()
            i += 1
        if w2v:
            print(f'Идет векторизация корпуса... ({i}/{total}): Word2Vec')
            self.shape_w2v()
            i += 1
        if ft:
            print(f'Идет векторизация корпуса... ({i}/{total}): FastText')
            self.shape_ft()
            i += 1
        if ai:
            print(f'Идет векторизация корпуса... ({i}/{total}): AI')
            self.shape_ai()
            i += 1
        print('Готово!')

    def shape_bm25(self):
        self.matrix_bm25 = BM25Okapi(self.corpora.lemmatized_corpora)

    def shape_w2v(self):
        self.w2v = KeyedVectors.load_word2vec_format('./models/Word2Vec/word2vec.bin', binary=True)
        self.matrix_w2v = []
        for text in tqdm(self.corpora.tokenized_corpora):
            text_query = []
            for word in text:
                if word in self.w2v:
                    text_query.append(self.w2v[word].reshape(1, -1))
            if len(text_query) == 0:
                text_query = np.zeros((1, 100))
            else:
                text_query = np.concatenate(text_query, axis=0).sum(axis=0).reshape(1, -1)
            self.matrix_w2v.append(text_query)
        self.matrix_w2v = np.concatenate(self.matrix_w2v, axis=0)

    def shape_ft(self):
        self.ft = KeyedVectors.load('./models/FastText/FastText.model')
        self.matrix_ft = []
        for text in tqdm(self.corpora.tokenized_corpora):
            text_query = []
            for word in text:
                if word in self.ft:
                    text_query.append(self.ft[word].reshape(1, -1))
            if len(text_query) == 0:
                text_query = np.zeros((1, 300))
            else:
                text_query = np.concatenate(text_query, axis=0).sum(axis=0).reshape(1, -1)
            self.matrix_ft.append(text_query)
        self.matrix_ft = np.concatenate(self.matrix_ft, axis=0)

    def shape_ai(self):
        self.ai = AIEmbed()
        self.matrix_ai = self.ai.get_embeddings(self.corpora.texts)

    def gettexts(self):
        return self.corpora.texts

    def tokenize(self, query):
        return self.corpora.tokenize(query)

    def lemmatize(self, query):
        return self.corpora.lemmatize(query)