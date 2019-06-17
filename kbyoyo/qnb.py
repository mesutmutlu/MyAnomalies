import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
import scipy as sp
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def calc_score(y_pred, y_true):
    #print(y_pred.ndim, y_true.ndim)
    w_p_score = precision_score(y_pred, y_true, average="weighted")
    w_r_score = recall_score(y_pred, y_true, average="weighted")
    w_f1_score = f1_score(y_pred, y_true, average="weighted")

    return w_p_score, w_r_score, w_f1_score

def print_score(w_p_score, w_r_score, w_f1_score):

    if y_pred.ndim == 2:
        print("Weighted Precision score: {}".format(w_p_score))
        print("Weighted Recall score: {}".format(w_r_score))
        print("Weighted F1 score: {}".format(w_f1_score))
    else:
        raise AssertionError("Prediction dimension is bigger than2 or smaller than 1")

pd_tr_cust = pd.read_csv(r"C:\datasets\kbyoyo\qnb\Koc_Yaz_Okulu_Data_Train_Cust.txt", delimiter=";")
pd_tr_agent = pd.read_csv(r"C:\datasets\kbyoyo\qnb\Koc_Yaz_Okulu_Data_Train_Agent.txt", delimiter=";")
pd_tr_target = pd.read_csv(r"C:\datasets\kbyoyo\qnb\Koc_Yaz_Okulu_Data_Train_Target.txt", delimiter=";")
pd_ts_cust = pd.read_csv(r"C:\datasets\kbyoyo\qnb\Koc_Yaz_Okulu_Data_Test_Cust.txt", delimiter=";")
pd_ts_agent = pd.read_csv(r"C:\datasets\kbyoyo\qnb\Koc_Yaz_Okulu_Data_Test_Agent.txt", delimiter=";")

# print(pd_tr_cust.head())
# print(pd_tr_agent.head())
# print(pd_tr_target.head())
# print(pd_ts_cust.head())
# print(pd_ts_agent.head())

labels = pd_tr_target.drop("ID", axis=1).columns.values.tolist()

train = pd_tr_cust.set_index("ID").join(pd_tr_agent.set_index("ID"), how="inner").join(pd_tr_target.set_index("ID"), how="inner")
train["SPEECH"] = train["CUST_TXT"] + " " + train["AGENT_TXT"]
#print(x_train)
train.drop(["CUST_TXT", "AGENT_TXT"], inplace=True, axis=1)
X_train = train["SPEECH"].values
Y_train = train.drop("SPEECH", axis=1).values.tolist()

val = pd_ts_cust.set_index("ID").join(pd_ts_agent.set_index("ID"), how="inner")
val["SPEECH"] = val["CUST_TXT"] + " " + val["AGENT_TXT"]
val.drop(["CUST_TXT", "AGENT_TXT"], inplace=True, axis=1)
X_val = val["SPEECH"].values


vectorizer = TfidfVectorizer( analyzer="word")
X_train = vectorizer.fit_transform(X_train)
svd = TruncatedSVD(n_components=120, random_state=1337)
svd.fit(X_train)
print("explained_variance_ratio_.sum() ", str(svd.explained_variance_ratio_.sum()))
#print(svd.explained_variance_ratio_)
X_train = svd.transform(X_train)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=42, test_size=0.33, shuffle=True)

assert(np.array(y_train).shape[1] == len(labels))


sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=0.001)
lr = LogisticRegression(solver='lbfgs')
mn = MultinomialNB()
svc = LinearSVC()

l_clf = [lr, svc, sgd]


classifiers = [
        (SGDClassifier(), {"n_estimators": sp.stats.randint(25, 100),
                                 'learning_rate': sp.stats.uniform(0.0001, 1)}),
        (lgb.LGBMClassifier(), {'num_leaves': sp.stats.randint(25, 330),
                            'n_estimators': sp.stats.randint(25, 150),
                            # 'bagging_fraction': sp.stats.uniform(0.4, 0.9),
                            'learning_rate': sp.stats.uniform(0.001, 0.5),
                            # 'min_data': sp.stats.randint(50,700),
                            # 'is_unbalance': [True, False],
                            # 'max_bin': sp.stats.randint(3,25),
                            'boosting_type': ['gbdt', 'dart'],
                            # 'bagging_freq': sp.stats.randint(3,35),
                            'max_depth': sp.stats.randint(3, 30),
                            'min_split_gain': sp.stats.uniform(0.001, 0.5),
                            'objective': 'multiclass'})
        ]

if 1 == 1:
    for classifier in l_clf:
        print(datetime.now(), classifier.__class__.__name__, "multi-label")
        clf = OneVsRestClassifier(classifier)
        clf.fit(x_train, np.array(y_train))
        y_pred = clf.predict(x_test)
        args = calc_score(y_pred, np.array(y_test))
        print_score(*args)

if 1 == 1:
    for classifier in l_clf :
        print(datetime.now(), classifier.__class__.__name__, "single-label")
        i = 0
        n_labels = len(labels)
        y_pred = np.empty((n_labels, x_test.shape[0]))
        for category in labels:
            #print(datetime.now(),'... Processing {}'.format(category))
            # train the model using X_dtm & y
            classifier.fit(x_train, np.array(y_train)[:,i])
            # compute the testing scores
            y_pred[i] = classifier.predict(x_test)
            i += 1
            #print(i, n_labels)
            if i == n_labels:
                args = calc_score(y_pred.T, np.array(y_test))
                print_score(*args)
