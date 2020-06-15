import os
import sys
import pickle
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path

def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]

def load_data(data_dir):
    x_train_sw = read_csv(os.path.join(data_dir, 'out_with_sw_train.csv'))
    x_val_sw = read_csv(os.path.join(data_dir, 'out_with_sw_validation.csv'))
    x_test_sw = read_csv(os.path.join(data_dir, 'out_with_sw_test.csv'))

    x_train_no_sw = read_csv(os.path.join(data_dir, 'out_without_sw_train.csv'))
    x_val_no_sw = read_csv(os.path.join(data_dir, 'out_without_sw_validation.csv'))
    x_test_no_sw = read_csv(os.path.join(data_dir, 'out_without_sw_test.csv'))

    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train_sw)]
    y_val = labels[len(x_train_sw): len(x_train_sw)+len(x_val_sw)]
    y_test = labels[-len(x_test_sw):]
    return x_train_sw, x_val_sw, x_test_sw, x_train_no_sw, x_val_no_sw, x_test_no_sw, y_train, y_val, y_test

    #a. mnb_uni.pkl: Classifier forunigrams w/stopwords
    #b. mnb_bi.pkl: Classifier for bigrams w/stopwords
    #c. mnb_uni_bi.pkl: Classifier for unigrams+bigrams w/ stopwords
    #d. mnb_uni_ns.pkl: Classifier forunigrams w/ostopwords
    #e. mnb_bi_ns.pkl: Classifier for bigrams w/ostopwords
    #f. mnb_uni_bi_ns.pkl: Classifier for unigrams+bigrams w/ostopwords

def train_unigram(x_train, y_train):
    print('Calling Vectors...')
    #print('Calling uni-gram Vectorizer:')
    unigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    x_train_count = unigram_vectorizer.fit_transform(x_train)
    #print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    #print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    root = Path(".")
    my_path = root / "data" / "mnb_uni.pkl"
    with open(my_path,'wb') as file:
        pickle.dump(clf, file)
    return clf, unigram_vectorizer, tfidf_transformer

def train_bigram(x_train, y_train):
    #print('Calling bi-gram Vectorizer:')
    bigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
    x_train_count = bigram_vectorizer.fit_transform(x_train)
    #print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    #print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    root = Path(".")
    my_path = root / "data" / "mnb_bi.pkl"
    with open(my_path,'wb') as file:
        pickle.dump(clf, file)
    return clf, bigram_vectorizer, tfidf_transformer

def train_unibigram(x_train, y_train):
    #print('Calling uni-bi-gram Vectorizer:')
    unibigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    x_train_count = unibigram_vectorizer.fit_transform(x_train)
    #print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    #print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    root = Path(".")
    my_path = root / "data" / "mnb_uni_bi.pkl"
    with open(my_path,'wb') as file:
        pickle.dump(clf, file)
    return clf, unibigram_vectorizer, tfidf_transformer

#WITHOUT STOPWORDS

def train_unigram_no_sw(x_train, y_train):
    #print('Calling uni-gram Vectorizer:')
    unigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    x_train_count = unigram_vectorizer.fit_transform(x_train)
    #print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    #print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    root = Path(".")
    my_path = root / "data" / "mnb_uni_ns.pkl"
    with open(my_path,'wb') as file:
        pickle.dump(clf, file)
    return clf, unigram_vectorizer, tfidf_transformer

def train_bigram_no_sw(x_train, y_train):
    #print('Calling bi-gram Vectorizer:')
    bigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
    x_train_count = bigram_vectorizer.fit_transform(x_train)
    #print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    #print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    root = Path(".")
    my_path = root / "data" / "mnb_bi_ns.pkl"
    with open(my_path,'wb') as file:
        pickle.dump(clf, file)
    return clf, bigram_vectorizer, tfidf_transformer

def train_unibigram_no_sw(x_train, y_train):
    #print('Calling uni-bi-gram Vectorizer:')
    unibigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    x_train_count = unibigram_vectorizer.fit_transform(x_train)
    #print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    #print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    root = Path(".")
    my_path = root / "data" / "mnb_uni_bi_ns.pkl"
    with open(my_path,'wb') as file:
        pickle.dump(clf, file)
    return clf, unibigram_vectorizer, tfidf_transformer

#EVALUATE FUNCTION
def evaluate(x, y, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {
        'accuracy': accuracy_score(y, preds)
        }

#MAIN FUNCTION
def main(data_dir):
    """
    loads the dataset along with labels, trains a simple MNB classifier
    and returns test scores in a dictionary
    """
# LOAD DATA
    x_train_sw, x_val_sw, x_test_sw, x_train_no_sw, x_val_no_sw, x_test_no_sw, y_train, y_val, y_test = load_data(data_dir)

#TRAINING and GETTING VALUES FOR CLF, VECTORIZER, TFIDF TRANSFORMER
    uni_clf_sw, unigram_vectorizer_sw, uni_tfidf_transformer_sw = train_unigram(x_train_sw, y_train)
    bi_clf_sw, bigram_vectorizer_sw, bi_tfidf_transformer_sw = train_bigram(x_train_sw, y_train)
    uni_bi_clf_sw, unibigram_vectorizer_sw, uni_bi_tfidf_transformer_sw = train_unibigram(x_train_sw, y_train)
    uni_clf_no_sw, unigram_vectorizer_no_sw, uni_tfidf_transformer_no_sw = train_unigram_no_sw(x_train_no_sw, y_train)
    bi_clf_no_sw, bigram_vectorizer_no_sw, bi_tfidf_transformer_no_sw = train_bigram_no_sw(x_train_no_sw, y_train)
    uni_bi_clf_no_sw, unibigram_vectorizer_no_sw, uni_bi_tfidf_transformer_no_sw = train_unibigram_no_sw(x_train_no_sw, y_train)

#UNIGRAM (WITHout STOPWORDS)
    uni_wo_scores = {}
    # validate
    print('unigrams (Without Stopwords)')
    #print('Validating')
    #uni_wo_scores['val'] = evaluate(x_val_no_sw, y_val, uni_clf_no_sw, unigram_vectorizer_no_sw, uni_tfidf_transformer_no_sw)
    # test
    print('Testing')
    uni_wo_scores['test'] = evaluate(x_test_no_sw, y_test, uni_clf_no_sw, unigram_vectorizer_no_sw, uni_tfidf_transformer_no_sw)
    print(uni_wo_scores)
    print('-------------------------------------')

#BIGRAM (WITHOUT STOPWORDS)
    bi_wo_scores = {}
    # validate
    print('bigrams (Without Stopwords)')
    #print('Validating')
    #bi_wo_scores['val'] = evaluate(x_val_no_sw, y_val, bi_clf_no_sw, bigram_vectorizer_no_sw, bi_tfidf_transformer_no_sw)
    # test
    print('Testing')
    bi_wo_scores['test'] = evaluate(x_test_no_sw, y_test, bi_clf_no_sw, bigram_vectorizer_no_sw, bi_tfidf_transformer_no_sw)
    print(bi_wo_scores)
    print('-------------------------------------')

#UNIBIGRAM (WITHOUT STOPWORDS)
    uni_bi_wo_scores = {}
    # validate
    print('unigrams + bigrams (Without Stopwords)')
    #print('Validating')
    #uni_bi_wo_scores['val'] = evaluate(x_val_no_sw, y_val, uni_bi_clf_no_sw, unibigram_vectorizer_no_sw, uni_bi_tfidf_transformer_no_sw)
    # test
    print('Testing')
    uni_bi_wo_scores['test'] = evaluate(x_test_no_sw, y_test, uni_bi_clf_no_sw, unibigram_vectorizer_no_sw, uni_bi_tfidf_transformer_no_sw)
    print(uni_bi_wo_scores)
    print('-------------------------------------')

#UNIGRAM (WITH STOPWORDS)
    uni_scores = {}
    # validate
    print('unigrams (With Stopwords)')
    #print('Validating')
    #uni_scores['val'] = evaluate(x_val_sw, y_val, uni_clf_sw, unigram_vectorizer_sw, uni_tfidf_transformer_sw)
    # test
    print('Testing')
    uni_scores['test'] = evaluate(x_test_sw, y_test, uni_clf_sw, unigram_vectorizer_sw, uni_tfidf_transformer_sw)
    print(uni_scores)
    print('-------------------------------------')

#BIGRAM (WITH STOPWORDS)
    bi_scores = {}
    # validate
    print('bigrams (With Stopwords)')
    #print('Validating')
    #bi_scores['val'] = evaluate(x_val_sw, y_val, bi_clf_sw, bigram_vectorizer_sw, bi_tfidf_transformer_sw)
    # test
    print('Testing')
    bi_scores['test'] = evaluate(x_test_sw, y_test, bi_clf_sw, bigram_vectorizer_sw, bi_tfidf_transformer_sw)
    print(bi_scores)
    print('-------------------------------------')

#UNIBIGRAM (WITH STOPWORDS)
    uni_bi_scores = {}
    # validate
    print('unigrams + bigrams (With Stopwords)')
    #print('Validating')
    #uni_bi_scores['val'] = evaluate(x_val_sw, y_val, uni_bi_clf_sw, unibigram_vectorizer_sw, uni_bi_tfidf_transformer_sw)
    # test
    print('Testing')
    uni_bi_scores['test'] = evaluate(x_test_sw, y_test, uni_bi_clf_sw, unibigram_vectorizer_sw, uni_bi_tfidf_transformer_sw)
    print(uni_bi_scores)
    print('-------------------------------------')

if __name__ == '__main__':
    data_path = sys.argv[1]
    pprint(main(data_path))
