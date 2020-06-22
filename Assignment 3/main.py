import os
import sys
import random
from gensim.models import Word2Vec
def main(data_path):
    #IMPORT and READ the text files
    with open(os.path.join(data_path, 'pos.txt')) as f:
      pos_lines = f.readlines()
    with open(os.path.join(data_path, 'neg.txt')) as f:
      neg_lines = f.readlines()
    #MERGE THE STRINGS
    merge_data = pos_lines + neg_lines
    #REMOVE SPECIAL CHARACTERS
    for c in '!"#$%&()*+/:;<=>@[\\]^`,.-{|}~\t':
        merge_data = [word.lower().replace(c,'') for word in merge_data]

    merge_data = [line.strip().split() for line in merge_data]

    #CREATE WORD-TO-VECTOR MODEL
    w2v = Word2Vec(merge_data, size=100, window=5, min_count=1, workers=4)
    #SAVE MODEL
    w2v.save('./data/w2v.model')

    print('------------------------------------------')

    #PRINT TOP 20 WORDS SIMILAR TO "good"
    print("Top 20 words similar to 'good' are: ")
    for x in w2v.most_similar("good", topn=20):
      top_20_good = x[0]
      prob_top_20_good = x[1]
      print('%20s' % top_20_good, ',' , round(prob_top_20_good, 3))
    
    print('------------------------------------------')
    
    #PRINT TOP 20 WORDS SIMILAR TO "bad"
    print("Top 20 words similar to 'bad' are: ")
    for y in w2v.most_similar("bad", topn=20):
      top_20_bad = y[0]
      prob_top_20_bad = y[1]
      print('%20s' % top_20_bad, ',' , round(prob_top_20_bad, 3)) 
    
    print('------------------------------------------')

#MAIN FUNCTION
if __name__ == "__main__":
    print('Scanning through the corpus...')
    data_path = sys.argv[1]
    main(data_path)
