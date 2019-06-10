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

x_train = pd_tr_cust.set_index("ID").join(pd_tr_agent.set_index("ID"), how="inner").join(pd_tr_target.set_index("ID"), how="inner")
x_train["SPEECH"] = x_train["CUST_TXT"] + " " + x_train["AGENT_TXT"]
#print(x_train)
x_train.drop(["CUST_TXT", "AGENT_TXT"], inplace=True, axis=1)


x_test = pd_ts_cust.set_index("ID").join(pd_ts_agent.set_index("ID"), how="inner")
x_test["SPEECH"] = x_test["CUST_TXT"] + " " + x_test["AGENT_TXT"]

#print(x_test)

train, val = train_test_split(x_train, random_state=42, test_size=0.33, shuffle=True)
X_train = train["SPEECH"].values
Y_train = train.drop("SPEECH", axis=1).values.tolist()
X_val = val["SPEECH"].values
Y_val = val.drop("SPEECH", axis=1).values.tolist()

print("------X_train------")
print(X_train[:5])
print("------Y_train------")
print(Y_train[:5])

vectorizer = TfidfVectorizer(ngram_range=(2,5), analyzer="word")
X_train = vectorizer.fit_transform(X_train)
svd = TruncatedSVD(n_components=120, random_state=1337)
svd.fit(X_train)
print("explained_variance_ratio_.sum() ", str(svd.explained_variance_ratio_.sum()))
#print(svd.explained_variance_ratio_)
X_train = svd.transform(X_train)
print(X_train)
print(Y_train)






sys.exit()
# Define a pipeline combining a text feature extractor with multi lable classifier
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
for category in labels:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_val)
    print(prediction)
    print('Validation accuracy is {}'.format(accuracy_score(val[category], prediction)))



SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
for category in labels:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_val)
    print('Test accuracy is {}'.format(accuracy_score(val[category], prediction)))


LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
for category in labels:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    LogReg_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = LogReg_pipeline.predict(X_val)
    print('Test accuracy is {}'.format(accuracy_score(val[category], prediction)))
    print('Test precision is {}'.format(accuracy_score(val[category], prediction)))
