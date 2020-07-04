import os
import sys
import re
import keras
import pickle
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#LOAD MODEL FUNCTION    
def load_model(file_path, x):
    fp1 = os.path.join(file_path,'{}.json'.format(x))
    json_file = open(fp1, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    json_file.close()
    # load weights into new model
    fp2 = os.path.join(file_path,'{}.h5'.format(x))
    loaded_model.load_weights(fp2)
    return loaded_model

#REMOVE PUNCTUATIONS FUNCTION
def remove_punctuations(x):
    for i, line in enumerate(x):
        for c in '!"#$%&()*+/:;<=>@[\\]^`,{|}~\t':
            line = [word.replace(c,'') for word in line]
            x[i] = line
    #print(listfile)
    elist=[]
    for lf in x:
        x = [' '.join(line.strip().split(',')) for line in lf]
        elist.append(str(x).replace('[','').replace(']','').replace('\'',''))
    return elist

#MAIN FUNCTION
def main(data_path, model_name):

    #LOAD THE MODEL
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name,'data')

    if 'relu' in model_name.lower():
        loaded_model = load_model(file_path, 'nn_relu')
        print("Loaded Relu model from the Data Folder!")
    elif 'sigmoid' in model_name.lower():
        loaded_model = load_model(file_path, 'nn_sigmoid')
        print("Loaded Sigmoid model from the Data Folder!")
    elif 'tanh' in model_name.lower():
        loaded_model = load_model(file_path, 'nn_tanh')
        print("Loaded Tanh model from the Data Folder!")
   
    #LOAD THE TEXT FILE
    textfile = open(os.path.join(data_path)).read()
    textfile = textfile.split('\n')
    
    #PRINT THE TEXT FILE
    print('------------------------------------------')
    print('')
    print('Text file sentences:')
    print(textfile)
    
    listfile=[]
    for line in textfile:
        listfile.append(line.splitlines())
    
    #REMOVE SPECIAL CHARACTERS FROM THE TEXT FILE SENTENCES
    elist = remove_punctuations(listfile)
    
    #PRINT THE PROCESSED TEXT FILE
    print('------------------------------------------')
    print('')
    print('Processed text file sentences:')
    print(elist)

    file_path = os.path.join(file_path,"tokenizer.pkl")
    tokenizer = pickle.load(open(file_path,'rb'))

    elist = tokenizer.texts_to_sequences(elist)
    
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 26
    elist = pad_sequences(elist, padding='post', maxlen=maxlen)

    #PRINT THE PREDICTED LIST
    print('------------------------------------------')
    print('')
    print('Predicted List:')
    print(loaded_model.predict_classes(elist))
    
if __name__ == "__main__":
    print('Scanning through the Text file...')
    data_path = sys.argv[1] #sys.argv[1] Text file Location
    model_name = sys.argv[2] #sys.argv[2] Model Name
    main(data_path, model_name)
