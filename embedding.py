import re
from config import MAX_LEN, TAG_TO_NUM, VolumeVec,WeightVec
from gensim.models import Word2Vec
import numpy as np
w2vModel = Word2Vec.load("word2vec.model")
def encode_and_pad(wordsANDtags):
    max_sentence_len = 36
    encoded_words, encoded_tags = [], []
    for w, t in zip(wordsANDtags[0], wordsANDtags[1]):
        if bool(re.search(r'^\d+[,\.\/\-x]*\d*m?c?l$',w)):
            encoded_words.append(VolumeVec)
            encoded_tags.append(TAG_TO_NUM['VOLUME'])
        elif bool(re.search(r'^\d+[,\.\/\-x]*\d*k?g$',w)):
            encoded_words.append(WeightVec)
            encoded_tags.append(TAG_TO_NUM['WEIGHT'])
        elif w in w2vModel.wv:
            encoded_words.append(w2vModel.wv[w])
            encoded_tags.append(TAG_TO_NUM[t])
        else:
            encoded_words.append(np.ones(50))
            encoded_tags.append(0)
    pad_length = max_sentence_len - len(encoded_words)
    if pad_length > 0:
        x = np.vstack((pad_length * [np.zeros(50)],encoded_words))
        y = np.array((pad_length * [0]) + encoded_tags)
    else:
        x = encoded_words[0:36]
        y = encoded_tags[0:36]
    return x, y