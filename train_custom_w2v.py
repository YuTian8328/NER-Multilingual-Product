from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from preprocess import prepare_text
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv('dataset.csv')
    df['combined'] = df.title +', '+df.description.fillna(' ')
    nltk.download('punkt')
    sentences = []
    for i in df.combined:
        sents = sent_tokenize(prepare_text(i))
        for sent in sents:
            sentences.append(word_tokenize(sent))

    model = Word2Vec(sentences, size=50, window=5, min_count=1, workers=4)
    model.save("w2v-model/word2vec.model")