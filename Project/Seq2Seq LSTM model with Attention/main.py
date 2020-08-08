#Import Libraries
from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
import re
import nltk
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional
from keras.models import Model, load_model
import tensorflow as tf

def main(data_path):

    #Read the Files
    with open(os.path.join(data_path, 'movie_lines.txt'), encoding = 'utf-8', errors = 'ignore') as f:
        movieLines = f.read().split('\n')
    with open(os.path.join(data_path, 'movie_conversations.txt'), encoding = 'utf-8', errors = 'ignore') as f:
        movieConversations = f.read().split('\n')

    #Data Preprocessing
    #Map each line's id with its text by creating a dictionary
    id2line = {}
    for l in movieLines:
        L = l.split(' +++$+++ ')
        if len(L) == 5:
            id2line[L[0]] = L[4]

    # Create a list of all of the conversations' lines' ids.
    conv_ids = []
    for c in movieConversations[:-1]:
        C = c.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        conv_ids.append(C.split(','))

    #Sort the sentences into questions (inputs) and answers (targets)
    questions = []
    answers = []
    for c in conv_ids:
        for i in range(len(c)-1):
            questions.append(id2line[c[i]])
            answers.append(id2line[c[i+1]])

    #Print length of question set and answer set
    print('Total Number of questions: ', len(questions))
    print('Total Number of answers  :', len(answers))

    #Removing the punctuations and cleaning
    def removePuncAndClean(txt):
        txt = txt.lower()
        txt = re.sub(r"i'm", "i am", txt)
        txt = re.sub(r"he's", "he is", txt)
        txt = re.sub(r"she's", "she is", txt)
        txt = re.sub(r"it's", "it is", txt)
        txt = re.sub(r"that's", "that is", txt)
        txt = re.sub(r"what's", "that is", txt)
        txt = re.sub(r"where's", "where is", txt)
        txt = re.sub(r"how's", "how is", txt)
        txt = re.sub(r"\'ll", " will", txt)
        txt = re.sub(r"\'ve", " have", txt)
        txt = re.sub(r"\'re", " are", txt)
        txt = re.sub(r"\'d", " would", txt)
        txt = re.sub(r"won't", "will not", txt)
        txt = re.sub(r"can't", "cannot", txt)
        txt = re.sub(r"n't", " not", txt)
        txt = re.sub(r"n'", "ng", txt)
        txt = re.sub(r"'bout", "about", txt)
        txt = re.sub(r"'til", "until", txt)
        txt = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", txt)
        txt = " ".join(txt.split())
        return txt

    #Cleaning the data
    cleanQuestions = []
    for q in questions:
        cleanQuestions.append(removePuncAndClean(q))
    cleanAnswers = []    
    for a in answers:
        cleanAnswers.append(removePuncAndClean(a))

    #Determining the length of sentences
    lengths = []
    for q in cleanQuestions:
        lengths.append(len(q.split()))
    for a in cleanAnswers:
        lengths.append(len(a.split()))

    # Remove questions and answers that are shorter than 1 word and longer than 25 words.
    smallQuestions = []
    smallAnswers = []
    for i, q in enumerate(cleanQuestions):
        if len(q.split()) >= 2 and len(q.split()) <= 25:
            smallQuestions.append(q)
            smallAnswers.append(cleanAnswers[i])

    #Filtering out the answers that are too short or long
    cleanQuestions = []
    cleanAnswers = []

    for i, a in enumerate(smallAnswers):
        if len(a.split()) >= 2 and len(a.split()) <= 25:
            cleanAnswers.append(a)
            cleanQuestions.append(smallQuestions[i])

    #choosing number of samples
    SampleSize = 15000
    cleanQuestions = cleanQuestions[:SampleSize]
    cleanAnswers = cleanAnswers[:SampleSize]

    import nltk
    #tokenizing the questions and answers
    allInputWords = [nltk.word_tokenize(sent) for sent in cleanQuestions]
    allTargetWords = [nltk.word_tokenize(sent) for sent in cleanAnswers]

    #train-validation split
    dataSize = len(allInputWords)

    # We will use the first 0-80th %-tile (80%) of data for the training
    X_train  = allInputWords[:round(dataSize*(80/100))]
    X_train  = [tr_input[::-1] for tr_input in X_train] #reverseing input seq for better performance
    y_train = allTargetWords[:round(dataSize*(80/100))]

    # We will use the remaining for validation
    X_test = allInputWords[round(dataSize*(80/100)):]
    X_test  = [val_input[::-1] for val_input in X_test] #reverseing input seq for better performance
    y_test = allTargetWords[round(dataSize*(80/100)):]

    print('Train data size is: ', len(X_train))
    print("Test data size is : ", len(X_test))

    #Actual Train Sentences
    XX_train = cleanQuestions[:round(dataSize*(80/100))]
    yy_train = cleanAnswers[:round(dataSize*(80/100))]
    #Actual Test Sentences
    XX_test = cleanQuestions[round(dataSize*(80/100)):]
    yy_test = cleanAnswers[round(dataSize*(80/100)):]

    #Convert to dataframe
    dfData =  list(zip(cleanQuestions, cleanAnswers))
    data = pd.DataFrame(dfData, columns = ['input' , 'target'])

    # Add start and end tokens to target sequences
    data.target = data.target.apply(lambda x : 'START '+ x + ' END')
    print('---')
    print('Random Sample Data: ')
    print(data.sample(6))
    print('---')

    # Create a dictionary for the frequency of the vocabulary
    vocabulary = {}
    for question in allInputWords:
        for word in question:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

    for answer in allTargetWords:
        for word in answer:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1     

    #Reducing vocabulary size and replace with UNK.
    threshold = 15
    count = 0
    for k,v in vocabulary.items():
        if v >= threshold:
            count += 1

    print("Size of total vocabulary:", len(vocabulary))

    #word_num 1 is for START tage for decoder
    word_num  = 2 
    encodingDict = {}
    decodingDict = {1: 'START'}
    #Vocabularies that appear above threshold count
    for word, count in vocabulary.items():
        if count >= threshold: 
            encodingDict[word] = word_num 
            decodingDict[word_num ] = word
            word_num += 1

    print("No. of vocabulary used:", word_num)

    #include unknown token for words not in dictionary
    decodingDict[len(encodingDict)+2] = 'UNK'
    encodingDict['UNK'] = len(encodingDict)+2

    dictSize = word_num+1

    #encodingDict: encoding dictionary
    #data: list of strings
    #vector_size: size of an encoded vector
    def modify(encodingDict, data, vector_size=20):
        transformedData = np.zeros(shape=(len(data), vector_size))
        for i in range(len(data)):
            for j in range(min(len(data[i]), vector_size)):
                try:
                    transformedData[i][j] = encodingDict[data[i][j]]
                except:
                    transformedData[i][j] = encodingDict['UNK']
        return transformedData

    #encoding training set
    inputLength = 25
    ouputLength = 25
    encodedTrainInput = modify(encodingDict, X_train, vector_size=inputLength)
    encodedTrainOutput = modify(encodingDict, y_train, vector_size=ouputLength)
    #encoding validation set
    encodedValInput = modify(encodingDict, X_test, vector_size=inputLength)
    encodedValOutput = modify(encodingDict, y_test, vector_size=ouputLength)

    print('Train data size is: ', encodedTrainInput.shape)
    print('Test data size is : ', encodedValInput.shape)

    #Building the Seq2Seq model
    import tensorflow as tf
    tf.keras.backend.clear_session()
    from keras.layers import SimpleRNN

    encoderInput = Input(shape=(inputLength,))
    decoderInput = Input(shape=(ouputLength,))

    #Encoder
    neuronDim = 512
    encoderEmbeddings = Embedding(dictSize, 128, input_length=inputLength, mask_zero=True)(encoderInput)
    encoderLstm = LSTM(neuronDim, return_sequences=True, unroll=True)(encoderEmbeddings)
    encoderState = encoderLstm[:,-1,:]

    print('encoderLstm: ', encoderLstm)
    print('encoderState: ', encoderState)

    #Decoder, with encoderStates as initial state
    decoderEmbeddings = Embedding(dictSize, 128, input_length=ouputLength, mask_zero=True)(decoderInput)
    decoderLstm = LSTM(neuronDim, return_sequences=True, unroll=True)(decoderEmbeddings, initial_state=[encoderState, encoderState])

    print('decoderLstm: ', decoderLstm)

    #Attention Mechanism
    from keras.layers import Activation, dot, concatenate

    attention = dot([decoderLstm, encoderLstm], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    print('attention: ', attention)

    contextVector = dot([attention, encoderLstm], axes=[2,1])
    print('contextVector: ', contextVector)

    decoderCombinedContext = concatenate([contextVector, decoderLstm])
    print('decoderCombinedContext: ', decoderCombinedContext)

    #Another weight and tanh layer
    decoderOutput = TimeDistributed(Dense(neuronDim, activation="tanh"))(decoderCombinedContext)
    decoderOutput = TimeDistributed(Dense(dictSize, activation="softmax"))(decoderOutput)
    print('decoderOutput: ', decoderOutput)


    #Define the model
    model = Model(inputs=[encoderInput, decoderInput], outputs=[decoderOutput])
    #Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    #preparing data for encoder and decoder
    trainEncoderInput = encodedTrainInput
    trainDecoderInput = np.zeros_like(encodedTrainOutput)
    trainDecoderInput[:, 1:] = encodedTrainOutput[:,:-1]
    trainDecoderInput[:, 0] = 1
    trainDecoderOutput = np.eye(dictSize)[encodedTrainOutput.astype('int')]

    testEncoderInput = encodedValInput
    testDecoderInput = np.zeros_like(encodedValOutput)
    testDecoderInput[:, 1:] = encodedValOutput[:,:-1]
    testDecoderInput[:, 0] = 1
    testDecoderOutput = np.eye(dictSize)[encodedValOutput.astype('int')]

    '''
    #UNCOMMENT to TRAIN THE MODEL
    for i in range(10):
      model.fit(x=[trainEncoderInput, trainDecoderInput], y=[trainDecoderOutput],
                    validation_data=([testEncoderInput, testDecoderInput], [testDecoderOutput]),
                    #validation_split=0.05,
                    batch_size=64, epochs=10)

    model.save('model_attention_weights.h5')
    '''
    #Load the model
    print('Loading the Model')
    model = load_model('model_attention_60.h5')

    #Define prediction function
    def prediction(raw_input):
        cleanInput = removePuncAndClean(raw_input)
        inputToken = [nltk.word_tokenize(cleanInput)]
        inputToken = [inputToken[0][::-1]]  #reverseing input seq
        encoderInput = modify(encodingDict, inputToken, 25)
        decoderInput = np.zeros(shape=(len(encoderInput), ouputLength))
        decoderInput[:,0] = 1
        for i in range(1, ouputLength):
            decoderOutput = model.predict([encoderInput, decoderInput]).argmax(axis=2)
            decoderInput[:,i] = decoderOutput[:,i]
        return decoderOutput

    def decodeSequence(decodingDict, vector):
        txt = ''
        for i in vector:
            if i == 0:
                break
            txt += ' '
            txt += decodingDict[i]
        return txt

    print("FIVE EXAMPLES: TRAIN SENTENCE PREDICTIONS: ")
    print('---')
    for i in range(5):
        seq_index = np.random.randint(1, len(XX_train))
        output = prediction(XX_train[seq_index])
        print('Question           :', XX_train[seq_index])
        print('Actual Response    : ', yy_train[seq_index])
        print('Predicted Response : ', decodeSequence(decodingDict, output[0]))
        print('----')

    print('----')
    print("FIVE EXAMPLES: TEST SENTENCE PREDICTIONS: ")
    print('---')
    for i in range(5):
        seq_index = np.random.randint(1, len(XX_test))
        output = prediction(XX_test[seq_index])
        print('Question           :', XX_test[seq_index])
        print('Actual Response    : ', yy_test[seq_index])
        print('Predicted Response : ', decodeSequence(decodingDict, output[0]))
        print('----')
    print('----')
    print('Importing libraries to calculate Bleu score...')
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.translate.bleu_score import sentence_bleu

    c = SmoothingFunction()
    print('---')
    print("Calculating Bleu Score for Train data...")
    print('---')
    bleuScoresTrain = []
    for x,y in zip(XX_train[:5], yy_train[:5]):

        output = prediction(x)

        actualOutput = y
        predictedOutput = decodeSequence(decodingDict, output[0])

        ref = actualOutput.split(' ')
        pred = predictedOutput.split(' ')

        if len(ref) >= 4 and len(pred) >= 4:
            BLEUscore = sentence_bleu([ref], pred, smoothing_function = c.method2)
        elif len(ref) >= 3 and len(pred) >= 3:
            BLEUscore = sentence_bleu([ref], pred, weights = (1.0/3, 1.0/3, 1.0/3), smoothing_function = c.method2)
        elif len(ref) >= 2 and len(pred) >= 2:
            BLEUscore = sentence_bleu([ref], pred, weights = (0.5, 0.5), smoothing_function = c.method2)
        else:
            BLEUscore = sentence_bleu([ref], pred, weights = [1], smoothing_function = c.method2)
        bleuScoresTrain.append(BLEUscore)

    print("Bleu Score for Train data: ", sum(bleuScoresTrain)/float(len(bleuScoresTrain)))
    print('---')
    print("Calculating Bleu Score for Test data...")
    print('---')
    bleuScoresTest = []
    for x,y in zip(XX_test[:5], yy_test[:5]):

        output = prediction(x)

        actualOutput = y
        predictedOutput = decodeSequence(decodingDict, output[0])

        ref = actualOutput.split(' ')
        pred = predictedOutput.split(' ')

        if len(ref) >= 4 and len(pred) >= 4:
            BLEUscore = sentence_bleu([ref], pred, smoothing_function = c.method2)
        elif len(ref) >= 3 and len(pred) >= 3:
            BLEUscore = sentence_bleu([ref], pred, weights = (1/3, 1/3, 1/3), smoothing_function = c.method2)
        elif len(ref) >= 2 and len(pred) >= 2:
            BLEUscore = sentence_bleu([ref], pred, weights = (0.5, 0.5), smoothing_function = c.method2)
        else:
            BLEUscore = sentence_bleu([ref], pred, weights = [1], smoothing_function = c.method2)
        bleuScoresTest.append(BLEUscore)

    print("Bleu Score for Test data: ", sum(bleuScoresTest)/float(len(bleuScoresTest)))
    print('---')
    print("END!")


#MAIN FUNCTION
if __name__ == "__main__":
    print('Scanning through the corpus...')
    nltk.download('punkt')
    data_path = sys.argv[1]
    main(data_path)
