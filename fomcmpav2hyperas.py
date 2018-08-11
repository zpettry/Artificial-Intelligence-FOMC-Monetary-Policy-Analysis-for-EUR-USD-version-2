from __future__ import print_function
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

def data():


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

    return x_train, y_train, x_val, y_val, embedding_dim, maxlen, max_words


def model(x_train, y_train, x_val, y_val, embedding_dim, maxlen, max_words):

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense({{choice([8, 16, 32, 64, 128])}}, activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([4, 8, 16, 32, 64])}}, activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([2, 4, 8, 16, 32])}}, activation='relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    #model.summary()

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model.fit(x_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=2,
        validation_data=(x_val, y_val))
    #callbacks=callbacks_list,
    score, acc = model.evaluate(x_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                          eval_space=True)

    x_train, y_train, x_val, y_val, embedding_dim, maxlen, max_words = data()

    print("Evalutation of best performing model:")
    score = best_model.evaluate(x_val, y_val)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
