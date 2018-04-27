import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,f1_score,precision_recall_curve

train_toxic_text, val_toxic_text, test_toxic_text, train_non_toxic_text, val_non_toxic_text, test_non_toxic_text = pickle.load(open('../Data/train_test_whole_base_val.pkl','rb'))
X_train = train_toxic_text + train_non_toxic_text
X_val = val_toxic_text + val_non_toxic_text
X_test = test_toxic_text + test_non_toxic_text
y_train = np.array([1]*len(train_toxic_text) + [0]*len(train_non_toxic_text))
y_val = np.array([1]*len(val_toxic_text) + [0]*len(val_non_toxic_text))
y_test = np.array([1]*len(test_toxic_text) + [0]*len(test_non_toxic_text))
def clean(comment):
    comment = comment.lower()
    comment = re.sub("\\n"," ",comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    comment = re.sub("\[\[.*\]","",comment)
    comment = re.sub("(http://.*?\s)|(http://.*)",'',comment)
    comment = re.sub("\[\[User.*",'',comment)
    comment = re.sub('\.+', '. ', comment)
    return comment

X_train_sentences = [clean(comment) for comment in X_train]
X_val_sentences = [clean(comment) for comment in X_val]
X_test_sentences = [clean(comment) for comment in X_test]
word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=50000)

# word_vectorizer = CountVectorizer(
#     strip_accents='unicode',
#     analyzer='word',
#     token_pattern=r'\w{1,}',
#     stop_words='english',
#     ngram_range=(1, 1),
#     max_features=50000)

word_vectorizer.fit(X_train_sentences)
train_word_features_tfidf = word_vectorizer.transform(X_train_sentences)
val_word_features_tfidf = word_vectorizer.transform(X_val_sentences)
test_word_features_tfidf = word_vectorizer.transform(X_test_sentences)

print (test_word_features_tfidf.shape)
shuffle_array = np.arange(train_word_features_tfidf.shape[0])
np.random.shuffle(shuffle_array)
train_word_features_tfidf = train_word_features_tfidf[shuffle_array]
train_targets = y_train[shuffle_array]

train_features = train_word_features_tfidf
test_features = test_word_features_tfidf
def pr(y_i, y):
    p = train_features[y==y_i].sum(0)
    return (p+1) / (p.sum()+1)

y = y_train
r = np.log(pr(1,y) / pr(0,y))
x_nb = train_features.multiply(r)
classifier = SVC(probability=True)
classifier.fit(x_nb, train_targets)

x_test_nb = test_features.multiply(r)
pred_y = classifier.predict(x_test_nb)
prob_y = classifier.predict_proba(x_test_nb)
print (pred_y)

print (accuracy_score(y_test, pred_y))
print (precision_recall_fscore_support(y_test, pred_y))
print (roc_auc_score(y_test, prob_y[:,1]))
print (roc_auc_score(y_test, pred_y))
print (f1_score(y_test, pred_y))

precision, recall, _ = precision_recall_curve(y_test, prob_y[:,1], pos_label=1)
pickle.dump([precision, recall], open('results/nbsvm_tfidf','wb'),-1)
