Assignment 2 
MSCI 641: Text Analytics 

Classification accuracy:
----------------------------------------------------------------------
Condition		        Text features		    Accuracy (test set)
----------------------------------------------------------------------
Without Stopwords	  unigrams		        0.8080
Without Stopwords	  bigrams			        0.7911
Without Stopwords	  unigrams+bigrams	  0.8264
With Stopwords		  unigrams		        0.8096
With Stopwords		  bigrams			        0.8213
With Stopwords		  unigrams+bigrams	  0.8320
----------------------------------------------------------------------

a. Which condition performed better: with or without stopwords? Write a briefparagraph (5-6 sentences) discussing why you think there is a difference in performance.

Answer: In this model, classfier with stopwords performed better than the one without the stopwords. This is because our corpus contains reviews which deal with the semantic meaning of the words. Although the stopwords act like noise for Count Vectorizer or TFIDF, they are important to understand the meaning of the reviews. Words such as "not", "against", "don't", etc get removed which changes the meaning of the sentence. For example, after removing the stopwords, review such as "My daughter did not admire the shampoo" becomes "daughter admire shampoo". This is actually a negative review, which is classfied as a positive review by the model. Thus, this decreases the model's accuracy as opposed to increasing it, resulting in better performance of the model with stopwords.

b. Which condition performed better: unigrams, bigrams or unigrams+bigrams? Briefly (in 5-6 sentences) discuss why you think there is a difference?

Answer: In this model, "unigrams+bigrams" performed better than the "unigrams" and "bigrams" conditions, because it captures multi-word expressions and considers word order dependence. Additionally, the bag of words model doesn’t account for potential misspellings or word derivations. In the review, "My daughter did not admire the shampoo", the "unigrams+bigrams" probability to predict the word based on its predecessor(s) is higher than the probability to predict a single word without a predecssor(s). The "unigrams+bigrams" combination captures phrases and increases the vocabulary of our grams. The features related to a specific review have more frequency than the individual unigrams and bigrams, resulting in improved classification and higher accuracy.
