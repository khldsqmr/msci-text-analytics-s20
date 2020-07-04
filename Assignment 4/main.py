import os
import sys
import re
import json
import keras
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers import Flatten
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import model_from_json

#READ CSV FUNCTION
def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]

#LOAD DATA FUNCTION
def load_data(data_dir):
    #READ CSV FILES
    out = read_csv(os.path.join(data_dir, 'out.csv'))
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test,out

#MAIN FUNCTION DEFINITION
def main(data_path):

    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name,'data')

    #LOAD THE DATA (VARIABLES)
    x_train, x_val, x_test, y_train, y_val, y_test, out = load_data(data_path)
    
    #TOKENIZER
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(x_train)

    #DUMP THE TOKENIZER IN PICKLE FILE
    p = os.path.join(file_path,"tokenizer.pkl")
    with open(p, 'wb') as file:
        pickle.dump(tokenizer, file)

    xx_train = tokenizer.texts_to_sequences(x_train)
    xx_test = tokenizer.texts_to_sequences(x_test)
    xx_val = tokenizer.texts_to_sequences(x_val)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 26

    xx_train = pad_sequences(xx_train, padding='post', maxlen = maxlen)
    xx_test = pad_sequences(xx_test, padding='post', maxlen = maxlen)
    xx_val = pad_sequences(xx_val, padding='post', maxlen = maxlen)

    #LOAD WORD2VEC MODEL
    w2v = Word2Vec.load(os.path.join(data_path, 'w2v.model'))

    #CREATE EMBEDDING MATRIX
    w2v_dict=dict({})
    for idx, key in enumerate(w2v.wv.vocab):
        w2v_dict[key] = w2v.wv[key]

    embedding_matrix = []
    vocabulary_size = min(vocab_size,20000)

    embedding_matrix = np.zeros((vocabulary_size, w2v.vector_size)) 
    for word, i in tokenizer.word_index.items():
        if i>=20000:
            continue
        try:
            embedding_vector = w2v_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            embeddings_vector = None
    
    #PRINT THE SHAPE OF EMBEDDINGS MATRIX
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))

    #CREATE MODEL
    print('Creating Model...')
    model = Sequential()
    print('Adding Layers...')
    model.add(Embedding(input_dim = vocabulary_size, 
                        output_dim = 100, 
                        input_length = 26, 
                        weights = [embedding_matrix], 
                        trainable = True))

    model.add(Flatten())
    model.add(Dense(50, activation = 'sigmoid', kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'softmax', name = 'output_layer'))

    print(model.summary())
    
    #COMPILING MODEL
    print('Compiling Model...')
    model.compile(optimizer = 'Adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    #FITTING MODEL ON TRAIN DATA
    print('Fitting Model on Train data...')
    model.fit(xx_train, 
              y_train, 
              epochs = 10, 
              verbose = 1, 
              batch_size = 200, 
              validation_data = (xx_val, y_val))

    #CALCULATE LOSS AND ACCURACY METRICS
    loss, accuracy = model.evaluate(xx_test, y_test, verbose=0)
    print("Accuracy on Test Data: ", accuracy)

    #SERIALIZE MODEL TO MODEL
    print('Saving Model...')
    m1 = os.path.join(file_path,"nn_sigmoid.model")
    model.save(m1)

    #SERIALIZE MODEL TO JSON
    model_json = model.to_json()
    m2 = os.path.join(file_path,"nn_sigmoid.json")
    with open(m2, 'w') as file:
        file.write(model_json)

    #SERIALIZE weights to HDF5
    m3 = os.path.join(file_path,"nn_sigmoid.h5")
    model.save_weights(m3)

    print("Sigmoid model saved to the Data Folder!")

#MAIN FUNCTION
if __name__ == "__main__":
    print('Reading files...')
    data_path = sys.argv[1] #Data folder location
    main(data_path)
