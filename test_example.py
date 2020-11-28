import re
import numpy as np
from nltk.tokenize import word_tokenize
from preprocess import prepare_text
from tensorflow.keras.models import load_model
from config import VolumeVec, WeightVec,w2vModel,NUM_TO_TAG
def size_extract_via_regex(text):
    '''extract size information from a given text by using regex
    '''
    size = []
    text = text.lower()
   
    size.extend(re.findall(r'R\d+[,\./]*\d*[\W]',text.title()))
    size.extend(re.findall(r'([\w\.]*[\Wø]+\d+[x\d\.,\-\/cm\s]*m?c?m)\b',text))
    size.extend(re.findall(r'kokoj?a?[\W\d\sdeuksx]*\d',text))
    size.extend(re.findall(r'koossa[\W\d\sdeuksx]*\d',text))
    size.extend(re.findall(r'mitat?[\W\d\sdeuksx]*\d',text))
    size.extend(re.findall(r'\bø\d[\d\.,\-\/\s]*\b',text))
    SIZES = ['s','m','l','xs','xxs','xl','xxl','2xl','3xl','ø','takapituus','pituus','korkeus','leveys','etupituus',
        'paksuus','de','eu','us','uk','xxxl','large','small','medium','middle','yksi','standard']
    word_list = word_tokenize(text)
    for i,word in enumerate(word_list):
        if word in ['koko','kokoja','kokoa','size','sizes','koossa']:
            for j in range(i,len(word_list)):
                if word_list[j] in SIZES:
                    if word_list[j] not in size:
                        size.append(word_list[j])
    print(' '.join(size))
def entity_extract(sentence):
    '''extract BRAND,COLOR,GENDER,VOLUME,WEIGHT information by using the NER model trained previously'''
    brand = []
    gender = []
    age = []
    color = []
    volume = []
    weight = []
    word_list = word_tokenize(prepare_text(sentence))
    encoded_input = []
    for w in word_list:
        if bool(re.search(r'^\d+[,\.\/\-]*\d*m?c?l$',w)):
            encoded_input.append(VolumeVec)
        elif bool(re.search(r'^\d+[,\.\/\-]*\d*k?g$',w)):
            encoded_input.append(WeightVec)
        elif w in w2vModel.wv:
            encoded_input.append(w2vModel.wv[w])
        else:
            encoded_input.append(np.ones(50))
    encoded_input = np.array(encoded_input)
    pad_length = 36 - encoded_input.shape[0]
    if pad_length > 0:
        test_input=np.vstack((pad_length * [np.zeros(50)],encoded_input))
    else:
        test_input=encoded_input[0:36]
    pred=np.argmax(evaluate_model.predict(np.array([test_input])),axis=-1)[0][-len(word_list):]
    pred_tag=[NUM_TO_TAG[p] for p in pred]
    for w,t in zip(word_list,pred_tag):
        if t == 'BRAND':
            if w.title() not in brand:
                brand.append(w.title())
        elif t == 'COLOR':
            color.append(w)
        elif t == 'GENDER':
            gender.append(w)
        elif t == 'AGE':
            age.append(w)
        elif t == 'VOLUME':
            volume.append(w)
        elif t == 'WEIGHT':
            weight.append(w)
    print({'brand':' '.join(set(brand)),'age':' '.join(set(age)),'gender':' '.join(set(gender)),'color':' '.join(set(color)),
            'weight':' '.join(set(weight)),'volume':' '.join(set(volume))})

if __name__ == '__main__':
    evaluate_model = load_model('checkpoint')
    text = """ Adidas naisten kengät musta koko 3,5
    """
    entity_extract(text)
    size_extract_via_regex(text)


    