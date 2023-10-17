from pymorphy3 import MorphAnalyzer
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
nltk.download('stopwords')

class Corpora:
    def __init__(self, path):
        self.analyzer = MorphAnalyzer()
        self.stopwords_ru = stopwords.words('russian')
        self.token_pattern = r'[\d\w\-]+'

        with open(path, 'r', encoding='utf-8') as book_file:
            self.texts = book_file.read().split('\n')
        self.lemmatized_corpora = []
        self.tokenized_corpora = []
        print('Идёт обработка корпуса...')
        for text in tqdm(self.texts):
            text_lemmatized = []
            text_tokenized = []
            for word in re.findall(self.token_pattern, text.lower()):
                lemma = self.analyzer.parse(word)[0].normal_form
                if lemma not in self.stopwords_ru:
                    text_lemmatized.append(lemma)
                    text_tokenized.append(word)
            self.lemmatized_corpora.append(text_lemmatized)
            self.tokenized_corpora.append(text_tokenized)

    def lemmatize(self, query):
        newtext = []
        for word in re.findall(self.token_pattern, query.lower()):
            lemma = self.analyzer.parse(word)[0].normal_form
            if lemma not in self.stopwords_ru:
                newtext.append(lemma)
        return newtext

    def tokenize(self, query):
        newtext = []
        for word in re.findall(self.token_pattern, query.lower()):
            lemma = self.analyzer.parse(word)[0].normal_form
            if lemma not in self.stopwords_ru:
                newtext.append(word)
        return newtext