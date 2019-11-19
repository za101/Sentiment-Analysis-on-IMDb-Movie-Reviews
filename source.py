import pandas as pd
import numpy as np
import nltk
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords ):
    review_text = BeautifulSoup(review,features="lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


if __name__ == '__main__':
    train = pd.read_csv('TrainData.tsv', header=0, \
                    delimiter="\t", quoting=3)
    
    test = pd.read_csv('testData.tsv', header=0, delimiter="\t", \
                   quoting=3 )
    
    clean_train_reviews = []
    print "Removing stop words from movie reviews training DataSet of length ",(2*len(train["review"]))/3

    labels=[]
    for i in xrange( 0, (2*len(train["review"]))/3):
        clean_train_reviews.append(" ".join(review_to_wordlist(train["review"][i], True)))
        labels.append(train["sentiment"][i])

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5500)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    np.asarray(train_data_features)
    
    print "Training random forest"
    forest = RandomForestClassifier(n_estimators = 100)

    print "Training SVM"
    svclassifier = SVC(kernel='linear')

    svc=svclassifier.fit(train_data_features, labels)
    forest = forest.fit( train_data_features, labels)
    
    clean_test_reviews = []
    test1=[]
    for i in xrange((2*len(train["review"]))/3,len(train["review"])):
        clean_test_reviews.append(" ".join(review_to_wordlist(train["review"][i], True)))
        test1.append(train["id"][i])

    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    print "Predicting results of testing\n"
    result = forest.predict(test_data_features)
    result1 = svc.predict(test_data_features)


    output = pd.DataFrame( data={"id":test1, "sentiment":result} )
    output.to_csv('Final.csv', index=False, quoting=3)
    print "Wrote results to Final.csv"

    t = pd.read_csv('Final.csv', header=0, delimiter="\t", \
                   quoting=3 )
    ctnr_rf=0
    ctnr_svm=0
    j=0
    #print "train->",train["sentiment"] 
    #print "result->",result
    for i in xrange( 2*len(train["review"])/3,len(train["review"]) ):
        if (train["sentiment"][i] == result[j]):
            ctnr_rf+=1
        if (train["sentiment"][i] == result1[j]):
            ctnr_svm+=1
        j+=1

    total=len(train["review"]) - (2*len(train["review"]))/3
    
    print "matched (random forest) :",ctnr_rf
    print "matched (svm ):",ctnr_svm
    print  "total", total
    
    acc_rf=(ctnr_rf*1.0/total*1.0)*100.0
    acc_svm=(ctnr_svm*1.0/total*1.0)*100.0

    print "accuracy (random forest) = ",acc_rf 
    print "accuracy (SVM) = ",acc_svm 
    





