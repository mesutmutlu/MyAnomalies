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
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

sys.stdout.buffer.write(chr(9986).encode('utf8'))
print(pd.set_option('display.max_columns', 500))
pd.set_option('display.width', 1000)

print(pd.read_csv("val_scores.csv"))
print(pd.read_csv("test_scores.csv"))


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
n_comp = 70
svd = TruncatedSVD(n_components=n_comp, random_state=1337)
svd.fit(X_train)
print("explained_variance_ratio_.sum() ", str(svd.explained_variance_ratio_.sum()))
#print(svd.explained_variance_ratio_)
X_train = svd.transform(X_train)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=42, test_size=0.33, shuffle=True)

assert(np.array(y_train).shape[1] == len(labels))

pd_test_scores = pd.DataFrame(columns=["clf","estimator", "params", "precision", "recall", "f1"])
pd_val_scores = pd.DataFrame(columns=["clf","estimator", "params", "metric", "score"])

l_ovr_clf = [
        (SGDClassifier(), {"estimator__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                           "estimator__eta0":sp.stats.uniform(0.001, 0.5),
                           "estimator__early_stopping":[True],
                           "estimator__loss":["hinge","log", "perceptron"],
                           "estimator__penalty":["l2"],
                           "estimator__alpha":sp.stats.uniform(0.00001, 0.001),
                           "estimator__n_jobs":[1],
                           "estimator__max_iter":sp.stats.randint(500, 2000),
                           "estimator__tol":sp.stats.uniform(0.0001, 0.01)}),
        (LGBMClassifier(), {'estimator__num_leaves': sp.stats.randint(25, 330),
                            'estimator__n_estimators': sp.stats.randint(25, 150),
                            #'estimator__bagging_fraction': sp.stats.uniform(0.4, 0.9),
                            'estimator__learning_rate': sp.stats.uniform(0.001, 0.5),
                            # 'min_data': sp.stats.randint(50,700),
                            'estimator__is_unbalance': [True],
                            # 'max_bin': sp.stats.randint(3,25),
                            'estimator__boosting_type': ['gbdt', 'dart'],
                            # 'bagging_freq': sp.stats.randint(3,35),
                            'estimator__max_depth': sp.stats.randint(3, 30),
                            'estimator__min_split_gain': sp.stats.uniform(0.001, 0.5),
                            #'estimator__objective': 'binary',
                            "estimator__n_jobs": [1]}),
        (LogisticRegression(), {"estimator__penalty":["l2"],
                                "estimator__C":sp.stats.randint(1, 10),
                                "estimator__solver": ["lbfgs", "liblinear", "sag", "saga"],
                                "estimator__n_jobs": [1]}),
        (LinearSVC(),{"estimator__penalty":["l2"],
                      "estimator__C": sp.stats.randint(1, 10)})
        ]

dct = DecisionTreeClassifier()
knc = KNeighborsClassifier()
rfc = RandomForestClassifier()
mlp = MLPClassifier()

l_clf = [
        (DecisionTreeClassifier(), {"max_depth": sp.stats.randint(n_comp/4, n_comp*2/3),
                                 'min_samples_split': sp.stats.randint(20, 300),
                                  "min_samples_leaf": sp.stats.randint(2, 150)
                                    }),
        (KNeighborsClassifier(), {'n_neighbors': sp.stats.randint(3, 100),
                            'algorithm': ["auto","ball_tree","kd_tree","brute"],
                            'leaf_size': sp.stats.uniform(10, 100)}),
            (RandomForestClassifier(), {"n_estimators":sp.stats.randint(10, 50),
                                 "max_depth": sp.stats.randint(n_comp/4, n_comp*2/3),
                                 'min_samples_split': sp.stats.randint(20, 300),
                                  "min_samples_leaf": sp.stats.randint(2, 150)})

        ]



for metric in ["precision_weighted", "recall_weighted", "f1_weighted", "accuracy"]:

    if 1 == 1:
        for clf in l_ovr_clf:
            print(datetime.now(), clf[0].__class__.__name__, "multi-label")
            random_search = RandomizedSearchCV(OneVsRestClassifier(clf[0]), param_distributions=clf[1], verbose=0,cv=5,
                                               n_iter=2, scoring=metric, n_jobs=3)

            # clf = OneVsRestClassifier(clf)
            #         # clf.fit(x_train, np.array(y_train))
            random_search.fit(x_train, np.array(y_train))
            #print(random_search.cv_results_)
            pd_val_scores = pd_val_scores.append(
                pd.Series([clf[0].__class__.__name__, random_search.best_estimator_, random_search.best_params_,
                           metric, random_search.best_score_], index=pd_val_scores.columns),
                ignore_index=True)
            y_pred = random_search.predict(x_test)
            args = calc_score(y_pred, np.array(y_test))
            pd_test_scores = pd_test_scores.append(
                pd.Series([clf[0].__class__.__name__, random_search.best_estimator_, random_search.best_params_,
                           args[0], args[1], args[2]], index=pd_test_scores.columns),
                ignore_index=True)

            #print_score(*args)
    if 1 == 1:
        for clf in l_clf:
            print(datetime.now(), clf[0].__class__.__name__, "multi-label")
            random_search = RandomizedSearchCV(clf[0], param_distributions=clf[1], verbose=0, cv=5,
                                               n_iter=2, scoring=metric, n_jobs=3)
            random_search.fit(x_train, np.array(y_train))
            pd_val_scores = pd_val_scores.append(
                pd.Series([clf[0].__class__.__name__, random_search.best_estimator_, random_search.best_params_,
                           metric, random_search.best_score_], index=pd_val_scores.columns),
                ignore_index=True)
            #print(random_search.cv_results_)
            y_pred = random_search.predict(x_test)
            args = calc_score(y_pred, np.array(y_test))
            #print(clf[0].__class__.__name__, args[0], args[1], args[2])
            pd_test_scores = pd_test_scores.append(
                pd.Series([clf[0].__class__.__name__, random_search.best_estimator_, random_search.best_params_,
                           args[0], args[1], args[2]], index=pd_test_scores.columns),
                ignore_index=True)
        #print_score(*args)

print(pd_val_scores)
pd_val_scores.to_csv("val_scores.csv")
print(pd_test_scores)
pd_test_scores.to_csv("test_scores.csv")
