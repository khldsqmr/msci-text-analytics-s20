#IMPORT and READ the text files

#text = open(sys.argv[1])
import os
import sys
import random
from gensim.models import Word2Vec

def main(data_path):
#LOAD THE MODEL
  w2v = Word2Vec.load('./data/w2v.model')
  
  textfile = open(os.path.join(data_path)).read()
  textfile = textfile.split('\n')
  print('------------------------------------------')
#FIND TOP 20 SIMILAR WORDS FOR EACH WORD IN TEXT FILE
  for i in textfile:
    i = i.lower()
    print('Top 20 words similar to \"', i, '\" are: ')
    for x in w2v.most_similar(i, topn=20):
      top_20_words = x[0]
      prob_top_20_words = x[1]
      print('%20s' % top_20_words, ',' , round(prob_top_20_words, 3))
    print('------------------------------------------')

#MAIN FUNCTION
if __name__ == "__main__":
    print('Scanning through the Text file...')
    data_path = sys.argv[1]
    main(data_path)
