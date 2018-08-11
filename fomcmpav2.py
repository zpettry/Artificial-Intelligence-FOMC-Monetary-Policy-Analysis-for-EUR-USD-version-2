from __future__ import print_function
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers.core import Dense, Dropout, Activation
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import numpy as np
import pandas as pd

'''
Best performing model chosen hyper-parameters:
{'Dense': 64, 'Dense_1': 16, 'Dropout_1': 0.3838088604298333, 'Dense_2': 8, 'Dropout': 0.9128294469805703}
'''


files = ['Datasets/feb99', 'Datasets/july99', 'Datasets/feb00', 'Datasets/july00',
            'Datasets/feb01', 'Datasets/july01', 'Datasets/feb02', 'Datasets/july02',
            'Datasets/feb03', 'Datasets/july03', 'Datasets/feb04', 'Datasets/july04',
            'Datasets/feb05', 'Datasets/july05', 'Datasets/feb06', 'Datasets/july06',
            'Datasets/feb07', 'Datasets/july07', 'Datasets/feb08', 'Datasets/july08',
            'Datasets/feb09', 'Datasets/july09', 'Datasets/feb10', 'Datasets/july10',
            'Datasets/feb11', 'Datasets/july11', 'Datasets/feb12', 'Datasets/july12',
            'Datasets/feb13', 'Datasets/july13', 'Datasets/feb14', 'Datasets/july14',
            'Datasets/feb15', 'Datasets/july15']

mydata = "FOMCSentimentAnalysis.xlsx"

maxlen = 20000
training_samples = 30
validation_samples = 4
max_words = 20000
embedding_dim = 256

samples = []
for x in files:
    sample = open(x, 'r', encoding='utf-8')
    sample = sample.read()
    samples.append(sample)

labels = []
excel_data = pd.ExcelFile(mydata)
excel_sheet = excel_data.parse('Sheet1')
eur_usd_change = excel_sheet["EUR/USD_Change"]
labels = np.asarray(eur_usd_change)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.9128294469805703))
model.add(Dense(16, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3838088604298333))
model.add(Dense(8, activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
    filepath='my_model_test.h5',
    monitor='val_loss',
    save_best_only=True
        )
    ]

model.fit(x_train, y_train,
    epochs=50,
    batch_size=32,
    verbose=2,
    callbacks=callbacks_list,
    validation_data=(x_val, y_val))

files_test = ['Datasets/feb16', 'Datasets/july16', 'Datasets/feb17', 'Datasets/july17','Datasets/feb18']

mydata_test = "FOMCSentimentAnalysis_Test.xlsx"

samples_test = []
for x in files_test:
    sample_test = open(x, 'r',encoding='utf-8')
    sample_test = sample_test.read()
    samples_test.append(x)

labels_test = []
excel_data_test = pd.ExcelFile(mydata_test)
excel_sheet_test = excel_data_test.parse('Sheet1')
eur_usd_change_test = excel_sheet_test["EUR/USD_Change"]
labels_test = np.asarray(eur_usd_change_test)

sequences_test = tokenizer.texts_to_sequences(samples_test)

x_test = pad_sequences(sequences_test, maxlen=maxlen)
y_test = np.asarray(labels_test)

model.load_weights('my_model_test.h5')
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
