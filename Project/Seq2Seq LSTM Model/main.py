#Import Libraries
import os
import sys
import re
import string
from string import digits
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.layers import Input, LSTM, Embedding, Dense,Dropout,TimeDistributed
from keras.models import Model

def main(data_path):

    #Read the Files
    with open(os.path.join(data_path, 'movie_lines.txt'), encoding = 'utf-8', errors = 'ignore') as f:
      movieLines = f.read().split('\n')
    with open(os.path.join(data_path, 'movie_conversations.txt'), encoding = 'utf-8', errors = 'ignore') as f:
      movieConversations = f.read().split('\n')

#Read the Files
#movieLines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
#movieConversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

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
        C = c.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conv_ids.append(C.split(','))

    # Sort the sentences into questions (inputs) and answers (targets)
    questions = []
    answers = []
    for c in conv_ids:
        for i in range(len(c) - 1):
            questions.append(id2line[c[i]])
            answers.append(id2line[c[i+1]])

    #Remove Punctuations
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

        return txt

    #Clean the data
    cleanQuestions = []
    for q in questions:
        cleanQuestions.append(removePuncAndClean(q))

    cleanAnswers = []
    for a in answers:
        cleanAnswers.append(removePuncAndClean(a))

    # Remove questions and answers that are shorter than 1 word and longer than 25 words.
    smallQuestions = []
    smallAnswers = []
    i = 0
    for q in cleanQuestions:
        if 2 <= len(q.split()) <= 25:
            smallQuestions.append(q)
            smallAnswers.append(cleanAnswers[i])
        i += 1
    cleanQuestions = []
    cleanAnswers = []
    i = 0
    for a in smallAnswers:
        if 2 <= len(a.split()) <= 25:
            cleanAnswers.append(a)
            cleanQuestions.append(smallQuestions[i])
        i += 1

    cleanQuestions = list(cleanQuestions)
    cleanAnswers = list(cleanAnswers)

    inputQuestions = []
    targetQuestions = []

    #Set sample size
    SampleSize = 15000

    for x in cleanQuestions[:SampleSize]:
      inputQuestions.append(x)

    for x in cleanAnswers[:SampleSize]:
      targetQuestions.append(x)

    #Convert to dataframe
    dfData =  list(zip(inputQuestions, targetQuestions))
    data = pd.DataFrame(dfData, columns = ['input' , 'target'])

    # Add start and end tokens to target sequences
    data.target = data.target.apply(lambda x : 'START '+ x + ' END')
    print('---')
    print('Random Sample Data: ')
    print(data.sample(6))
    print('---')

    #Build Vocabulary
    #Vocabulary of Questions storing all the words in a set
    allInputWords=set()
    for i in data.input:
        for word in i.split():
            if word not in allInputWords:
                allInputWords.add(word)
    # Vocabulary of Answers  storing all the words in a set
    allTargetWords=set()
    for t in data.target:
        for word in t.split():
            if word not in allTargetWords:
                allTargetWords.add(word)

    # Max Length of questions 
    lenList=[]
    for l in data.input:
        lenList.append(len(l.split(' ')))
    maxInputLen = np.max(lenList)
    print("Maximum Question's Length: ", maxInputLen)

    # Max Length of Answers
    lenList=[]
    for l in data.target:
        lenList.append(len(l.split(' ')))
    maxTargetLen = np.max(lenList)
    print("Maximum Answer's Length: ", maxTargetLen)

    inpWords = sorted(list(allInputWords))
    tarWords = sorted(list(allTargetWords))

    #Storing the Vocabulary size for Encoder and Decoder
    numOfEncTokens = len(allInputWords)
    numOfDecTokens = len(allTargetWords)
    print("Encoder token size is {} and decoder token size is {}".format(numOfEncTokens,numOfDecTokens))

    #Zero Padding
    numOfDecTokens += 1 
    print("Decoder Token size after zero padding: ", numOfDecTokens)

    #Dictionary to index each word in Questions: key is index and value is word
    inputIdx2char = {}
    #Dictionary to get words given its index: key is word and value is index
    inputChar2Idx = {}

    for key, value in enumerate(inpWords):
        inputIdx2char[key] = value
        inputChar2Idx[value] = key

    #Dictionary to index each word in Answers: key is index and value is word
    outputIdx2char = {}
    #Dictionary to get words given its index: key is word and value is index
    outputChar2idx = {}
    for key,value in enumerate(tarWords):
        outputIdx2char[key] = value
        outputChar2idx[value] = key

    # Spliting our data into train and test
    from sklearn.model_selection import train_test_split
    X, y = data.input, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    print("Train data size is: ", X_train.shape)
    print("Test data size is : ", X_test.shape)

    #Generator function to train on batch to reduce computation, increase learning and model performance
    def generate_batch(X, y, batch_size = 128):
        while True:
            for j in range(0, len(X), batch_size):
                
                #encoder input data
                encoderInputData = np.zeros((batch_size, maxInputLen),dtype='float32')
                #decoder input data
                decoderInputData = np.zeros((batch_size, maxTargetLen),dtype='float32')
                
                #decoder target data
                decoderTargetData = np.zeros((batch_size, maxTargetLen, numOfDecTokens),dtype='float32')
                
                for i, (inputTxt, targetTxt) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for txt, word in enumerate(inputTxt.split()):
                        encoderInputData[i, txt] = inputChar2Idx[word] # encoder input seq
                        
                    for txt, word in enumerate(targetTxt.split()):
                        if txt<len(targetTxt.split())-1:
                            decoderInputData[i, txt] = outputChar2idx[word] # decoder input seq
                        if txt>0:
                            decoderTargetData[i, txt - 1, outputChar2idx[word]] = 1
                            
                yield([encoderInputData, decoderInputData], decoderTargetData)


    neuronDim = 256

    #Encoder
    encoderInputs = Input(shape=(None,))
    encoderEmbeddings =  Embedding(numOfEncTokens, neuronDim, mask_zero = True)(encoderInputs)
    encoderLstm = LSTM(neuronDim, return_state=True)
    encoderOutputs, state_h, state_c = encoderLstm(encoderEmbeddings)
    encoderStates = [state_h, state_c]


    #Decoder, with encoderStates as initial state
    decoderInputs = Input(shape=(None,))
    decoderEmbeddings = Embedding(numOfDecTokens, neuronDim, mask_zero = True)
    decEmb = decoderEmbeddings(decoderInputs)
    #To use the return states in inference.
    decoderLstm = LSTM(neuronDim, return_sequences=True, return_state=True)
    decoderOutputs, _, _ = decoderLstm(decEmb,initial_state=encoderStates)
    decoderDense = Dense(numOfDecTokens, activation='softmax')
    decoderOutputs = decoderDense(decoderOutputs)

    #Define the model
    model = Model([encoderInputs, decoderInputs], decoderOutputs)

    #Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    trainSamples = len(X_train)
    valSamples = len(X_test)
    batchSize = 64
    epochs = 50

    print('---')
    print('Printing Model Summary...')
    print('---')
    model.summary()

    ''' 
    #UNCOMMENT TO TRAIN THE EPOCHS

    for i in range(100):
      history=model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batchSize),
                        steps_per_epoch = trainSamples//batchSize,
                        epochs=1,
                        validation_data = generate_batch(X_test, y_test,    batch_size = batchSize),
                        validation_steps = valSamples//batchSize)

    model.save_weights('Enc_Dec_Weights_100_epochs.h5')
    '''

    print('Loading the Model weights...')

    model.load_weights('Enc_Dec_Weights_70_epochs.h5')


    #Encode the input sequence to get the "Context vectors"
    encoderModel = Model(encoderInputs, encoderStates)

    #Setting up the decoder
    #Tensors to hold the states of the previous time-step
    decoderStateInput_h = Input(shape=(neuronDim,))
    decoderStateInput_c = Input(shape=(neuronDim,))
    decoderStateInput = [decoderStateInput_h, decoderStateInput_c]

    #To retrieve the embeddings of the decoder sequence
    decoderEmbeddings2 = decoderEmbeddings (decoderInputs)

    #To predict the next word in the sequence, setting the initial states to the states from the previous time-step
    decoderOutputs2, state_h2, state_c2 = decoderLstm(decoderEmbeddings2, initial_state=decoderStateInput)
    decoderStates2 = [state_h2, state_c2]

    #A dense softmax layer to generate probability distribution over the answers vocabulary
    decoderOutputs2 = decoderDense(decoderOutputs2)

    #Final decoder model
    decoderModel = Model([decoderInputs] + decoderStateInput,
                        [decoderOutputs2] + decoderStates2)

    #Deoode Sequence function
    def decodeSequence(inputSequence):
        #Encode the input as state vectors
        statesValue = encoderModel.predict(inputSequence)

        #Generate empty target sequence of length 1
        targetSequence = np.zeros((1,1))

        #Fill the first character of target sequence with the start character
        targetSequence[0, 0] = outputChar2idx['START']

        #Sampling loop for a batch of sequences assuming batch of size 1
        toStop = False
        decodedSentence = ''
        while not toStop:
            outputTokens, h, c = decoderModel.predict([targetSequence] + statesValue)

            #Sample a token
            sampledTokenIdx = np.argmax(outputTokens[0, -1, :])
            sampledWord =outputIdx2char[sampledTokenIdx]
            decodedSentence += ' '+ sampledWord

            #Stop condition by either the max length or find stop character
            if (sampledWord == 'END' or
              len(decodedSentence) > 50):
                toStop = True

            #Updating the target sequence (of length 1)
            targetSequence = np.zeros((1,1))
            targetSequence[0, 0] = sampledTokenIdx

            #Updating the states
            statesValue = [h, c]

        return decodedSentence
    print('---')
    #Train generator
    trainGenerator = generate_batch(X_train[:5], y_train[:5], batch_size = 1)
    k=-1
    #Test Generator
    testGenerator = generate_batch(X_test[:5], y_test[:5], batch_size=1)
    m=-1
    n=-1

    #Print five example sentences
    print("FIVE EXAMPLES: TRAIN SENTENCE PREDICTIONS: ")
    print('---')
    for i in range(5):
        n+=1
        (inputSequence, actualOutput), _ = next(trainGenerator)
        decodedSentence = decodeSequence(inputSequence)
        print('Question           :', X_train[n:n+1].values[0])
        print('Actual response    :', y_train[n:n+1].values[0][6:-4])
        print('Predicted response :', decodedSentence[:-4])
        print('---')
        
    print("FIVE EXAMPLES: TEST SENTENCE PREDICTIONS: ")
    print('---')
    n=-1
    for i in range(5):
        n+=1
        (inputSequence, actualOutput), _ = next(testGenerator)
        decodedSentence = decodeSequence(inputSequence)
        print('Question           :', X_test[n:n+1].values[0])
        print('Actual response    :', y_test[n:n+1].values[0][6:-4])
        print('Predicted response :', decodedSentence[:-4])
        print('---')

    #Import libraries to calculate Bleu Score
    print('Importing libraries to calculate bleu score...')
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.translate.bleu_score import sentence_bleu
    c = SmoothingFunction()
    print('---')
    print("Calculating Bleu Score for Train data...")
    print('---')
    bleuScoresTrain = []
    for i in range(len(X_train[:5])):

        k+=1
        (inputSentence, actualOutput), _ = next(trainGenerator)
        decodedSentence = decodeSequence(inputSentence)

        actualOutput = y_train[k:k+1].values[0][6:-4]
        predictedOutput = decodedSentence[:-4]

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
        bleuScoresTrain.append(BLEUscore)

    print("Bleu Score for Train data: ", sum(bleuScoresTrain)/float(len(bleuScoresTrain)))
    print('---')
    print("Calculating Bleu Score for Test data...")

    bleuScoresTest = []
    for i in range(len(X_test[:5])):

        m+=1
        (inputSentence, actualOutput), _ = next(testGenerator)
        decodedSentence = decodeSequence(inputSentence)

        actualOutput = y_train[m:m+1].values[0][6:-4]
        predictedOutput = decodedSentence[:-4]

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

    print('---')
    print("Bleu Score for Test data: ", sum(bleuScoresTest)/float(len(bleuScoresTest)))
    print('---')
    print('END!')

#MAIN FUNCTION
if __name__ == "__main__":
    print('Scanning through the corpus...')
    data_path = sys.argv[1]
    main(data_path)
