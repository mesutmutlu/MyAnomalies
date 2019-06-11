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

def print_score(y_pred, y_true, clf):
    #print("Clf: ", clf.__class__.__name__)
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_true)))
    print("Hamming score: {}".format(hamming_score(y_pred, y_true)))
    print("Accuracy score: {}".format(accuracy_score(y_pred, y_true)))
    if y_pred.shape[1] > 1:
        print("Weighted Precision score: {}".format(precision_score(y_pred, y_true, average="weighted")))
        print("Weighted Recall score: {}".format(recall_score(y_pred, y_true, average="weighted")))
        print("Weighted F1 score: {}".format(f1_score(y_pred, y_true, average="weighted")))
        print("Micro Precision score: {}".format(precision_score(y_pred, y_true, average="micro")))
        print("Micro Recall score: {}".format(recall_score(y_pred, y_true, average="micro")))
        print("Micro F1 score: {}".format(f1_score(y_pred, y_true, average="micro")))
        print("Macro Precision score: {}".format(precision_score(y_pred, y_true, average="macro")))
        print("Macro Recall score: {}".format(recall_score(y_pred, y_true, average="macro")))
        print("Macro F1 score: {}".format(f1_score(y_pred, y_true, average="macro")))
    else:
        print("Precision score: {}".format(precision_score(y_pred, y_true)))
        print("Recall score: {}".format(recall_score(y_pred, y_true)))
        print("F1 score: {}".format(f1_score(y_pred, y_true)))
    print("---")

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


#nb_clf = MultinomialNB()
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=0.001)
lr = LogisticRegression(solver='lbfgs')
#mn = MultinomialNB()
svc = LinearSVC()

l_clf = [lr, svc, sgd]

for classifier in l_clf:
    print(datetime.now(), classifier.__class__.__name__, "multi-label")
    clf = OneVsRestClassifier(classifier)
    clf.fit(x_train, np.array(y_train))
    y_pred = clf.predict(x_test)
    print_score(y_pred, np.array(y_test), classifier)


for classifier in l_clf:
    i = 0
    for category in labels:
        print(datetime.now(), classifier.__class__.__name__,'... Processing {}'.format(category))
        # train the model using X_dtm & y
        classifier.fit(x_train, np.array(y_train)[:,i])
        # compute the testing scores
        y_pred = classifier.predict(x_test)
        print_score(y_pred, np.array(y_test), classifier)
        i += 1
