import re
import numpy as np
from tensorflow import keras
from config import COLORS, GENDERS, AGES, TAG_TO_NUM, NUM_TO_TAG, w2vModel, WeightVec,VolumeVec
def annotate(row,df,i):
    '''annotate function to label COLOR,GENDER,AGE,VOLUME,WEIGHT'''
    pattern = re.compile(r'[a-z]+')
    raw_w, raw_t = [], []
    for word, tag in row:
        if word.lower() in COLORS:
            raw_w.append(word)
            raw_t.append('COLOR')
        elif word.lower() in pattern.findall(df.color[i].lower()):
            raw_w.append(word)
            raw_t.append('COLOR')
        elif word.lower() in pattern.findall(df.brand[i].lower()):
            raw_w.append(word)
            raw_t.append('BRAND')        
        elif word.lower() in GENDERS:
            raw_w.append(word)
            raw_t.append('GENDER')
        elif word.lower() in AGES:
            raw_w.append(word)
            raw_t.append('AGE')    
        elif bool(re.search(r'^\d+[,\.\/\-x]*\d*m?c?l$',word)):
            raw_w.append(word)
            raw_t.append('VOLUME')
        elif bool(re.search(r'^\d+[,\.\/\-x]*\d*k?g$',word.lower())):
            raw_w.append(word)
            raw_t.append('WEIGHT')
        else:
            raw_w.append(word)
            raw_t.append(tag)
            
    return(raw_w,raw_t)
def encode_and_pad(wordsANDtags):
    '''convert word tokens to word embedding vectors and pad sentences to max length'''
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
class DataGenerator(keras.utils.Sequence):
    '''Generate data for Keras Model'''
    def __init__(self,dataset,batch_size=64, dim=(36,50),
                 n_classes=9, shuffle=True):
#         Initialization
        self.dataset = dataset
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
#         'Updates indexes after each epoch'
        self.indexes = np.arange(self.dataset.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
#         'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset.shape[0]) / self.batch_size)

    def __getitem__(self, index):
#         'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp,max_sentence_len = 36)

        return X, y
    
    def __data_generation(self,list_IDs_temp,max_sentence_len = 36):
        X = np.empty((64,36,50))
        Y = np.empty((64,36), dtype=int)
        for i,ID in enumerate(list_IDs_temp):
            wordsANDtags = annotate(row=self.dataset.combined[ID],df=self.dataset,i=ID)
            x,y = encode_and_pad(wordsANDtags)
            X[i,]=x
            Y[i,]=y
        return X, Y
    