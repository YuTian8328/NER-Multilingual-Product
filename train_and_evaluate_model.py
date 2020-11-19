import config
import numpy as np
import pandas as pd
from nltk import word_tokenize
from datagenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, Bidirectional

#Randomly split 670k observations into training set,validation set and test set
df = pd.read_csv('after_branding.csv',converters={'combined':eval})
test_set = df.sample(frac=0.1,random_state=200) 
train_val = df.drop(test_set.index)
val_set = train_val.sample(frac=0.3,random_state=200)
train_set = train_val.drop(val_set.index)

train_set.reset_index(drop=True,inplace=True)
val_set.reset_index(drop=True,inplace=True)
test_set.reset_index(drop=True,inplace=True)
#Train ner model and save best weights together with the model architecture into checkpoint
params = {'dim': (36,50),
          'batch_size': 64,
          'n_classes': 9,
          'shuffle': True}
training_generator = DataGenerator(train_set,**params)
validation_generator = DataGenerator(val_set,**params)
test_generator = DataGenerator(test_set,**params)
model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(50, activation="relu"))) 
model.add(Dense(9,activation='softmax'))
model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# checkpoint = ModelCheckpoint('checkpoint', monitor='val_accuracy', verbose=1, save_best_only=True)
# callbacks_list = [checkpoint]

if __name__ == '__main__':
    model.fit_generator(generator=training_generator,callbacks=None,
                validation_data=validation_generator,epochs=3,verbose=1,
                    workers=-1)
    loss,accuracy = model.evaluate_generator(generator=test_generator,workers=-1, verbose=0)
    print("Accuracy on test set:",accuracy)