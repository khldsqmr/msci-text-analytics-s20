import os
import sys
import random
from random import shuffle
from random import sample
def main():
#IMPORT and READ the text files
    print("Please change the current working directory to the address where the pos.txt file, neg.txt file and .py file is stored by using os.chdir() and check using os.getcwd()")
    pos = input("Enter the name of the positive file (pos.txt): ")
    with open(pos) as positive:
	    pos = positive.read()
    neg = input("Enter the name of the negative file (neg.txt): ")
    with open(neg) as negative:
	    neg = negative.read()
#MERGE THE STRINGS
    merge_data = pos+'\n'+neg
#SPLIT ALONG EACH LINE 
    merge_data = merge_data.split('\n')
#STORE THE LISTS IN A MAIN LIST
    merge_list = []
    for line in merge_data:
        merge_list.append(line.split())
#WITH STOPWORDS
    for i, line in enumerate(merge_list):
      for c in '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n':
        line = [word.replace(c,'') for word in line]
        merge_list[i] = line
#STORE THE MAIN LIST IN A VARIABLE (WITH STOPWORDS)
    merge_list_with_sw = merge_list.copy()
#OUTPUT THE CSV FILE (WITH STOPWORDS AS out_with_sw.csv)
    f = open('out_with_sw.csv','w')
    i = 0
    while (i<len(merge_list_with_sw)):   
        f.write(str(merge_list_with_sw[i])+"\n")
        i = i + 1
    f.close()
#TRAINING, TESTING AND VALIDATION DATA (WITH STOPWORDS)
    shuffle(merge_list_with_sw)
#OUTPUT THE CSV FILE (TRAINING DATA AS out_with_sw_train.csv)
    training_with_stopwords = merge_list_with_sw[:round(0.8*len(merge_list_with_sw))]
    f = open('out_with_sw_train.csv','w')
    i = 0
    while (i<len(training_with_stopwords)):   
        f.write(str(training_with_stopwords[i])+"\n")
        i = i + 1
    f.close()
#OUTPUT THE CSV FILE (TESTING DATA AS out_with_sw_test.csv)
    testing_with_stopwords = merge_list_with_sw[round(0.8*len(merge_list_with_sw)): round(0.9*len(merge_list_with_sw))]
    f = open('out_with_sw_test.csv','w')
    i = 0
    while (i<len(testing_with_stopwords)):   
        f.write(str(testing_with_stopwords[i])+"\n")
        i = i + 1
    f.close()
#OUTPUT THE CSV FILE (VALIDATION DATA AS out_with_sw_validation.csv)
    validation_with_stopwords = merge_list_with_sw[round(0.9*len(merge_list_with_sw)): len(merge_list_with_sw)]
    f = open('out_with_sw_validation.csv','w')
    i = 0
    while (i<len(validation_with_stopwords)):   
        f.write(str(validation_with_stopwords[i])+"\n")
        i = i + 1
    f.close()
#WITHOUT STOPWORDS
#LIST OF STOPWORDS
    stopwords = ["i", "the", "me", "my","myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
#REMOVE THE STOPWORDS
    for line in merge_list:
      for c in range(len(line)):
        for i,word in enumerate(line):
            if word.lower() in stopwords:
              line.remove(word)
#STORE THE MAIN LIST IN A VARIABLE (WITHOUT STOPWORDS)
    merge_list_without_sw = merge_list.copy()
#OUTPUT THE CSV FILE (WITHOUT STOPWORDS AS out_without_sw.csv)
    f = open('out_without_sw.csv','w')
    i = 0
    while (i<len(merge_list_without_sw)):   
        f.write(str(merge_list_without_sw[i])+"\n")
        i = i + 1
    f.close()
#TRAINING,TESTING AND VALIDATION DATA (WITHOUT STOPWORDS)
    shuffle(merge_list_without_sw)
#OUTPUT THE CSV FILE (TRAINING DATA AS out_without_sw_train.csv)
    training_without_stopwords = merge_list_without_sw[:round(0.8*len(merge_list_without_sw))]
    f = open('out_without_sw_train.csv','w')
    i = 0
    while (i<len(training_without_stopwords)):   
        f.write(str(training_without_stopwords[i])+"\n")
        i = i + 1
    f.close()
#OUTPUT THE CSV FILE (TESTING DATA AS out_without_sw_test.csv)
    testing_without_stopwords = merge_list_without_sw[round(0.8*len(merge_list_without_sw)):round(0.9*len(merge_list_without_sw))]
    f = open('out_without_sw_test.csv','w')
    i = 0
    while (i<len(testing_without_stopwords)):   
        f.write(str(testing_without_stopwords[i])+"\n")
        i = i + 1
    f.close()
#OUTPUT THE CSV FILE (VALIDATION DATA AS out_without_sw_validation.csv)
    validation_without_stopwords = merge_list_without_sw[round(0.9*len(merge_list_without_sw)):len(merge_list_without_sw)]
    f = open('out_without_sw_validation.csv','w')
    i = 0
    while (i<len(validation_without_stopwords)):   
        f.write(str(validation_without_stopwords[i])+"\n")
        i = i + 1
    f.close()
#MAIN FUNCTION
if __name__ == "__main__":
  main()