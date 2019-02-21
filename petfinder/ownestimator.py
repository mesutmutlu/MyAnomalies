from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import numpy as np
import joblib

class RatioOrdinalClassfier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, estimator=lgb.LGBMClassifier()):
        """
        Called when initializing the classifier
        """
        # Parameters should have same name as attributes
        self.estimator = estimator
        self.estimators_ = []

    def encode_classes(self, y, yi):
        if y[0] > yi:
            return 1
        else:
            return 0

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.sorted_classes_ = np.unique(y)
        #print(self.sorted_classes_)
        self.num_classes_ = len(self.sorted_classes_)
        self.num_instances_ = len(X)
        self.probas_ = np.zeros((self.num_instances_, 1))

        for yi in self.sorted_classes_[:-1]:
            #print(yi)
            yt = y.copy()
            yt[yt <= yi] = 0
            yt[yt > yi] = 1
            est = self.estimator
            est.fit(X, yt.ravel())
            filename = "ownestimatormodel_"+str(yi)+".sav"
            joblib.dump(est, filename)
            self.estimators_.append(est)
        print(self.estimators_)
        return self

    def predict_proba(self, X):
        i = 0
        for yi in self.sorted_classes_[:-1]:
            print("estimator",yi,"-------")
            print(self.estimators_[yi])
            filename = "ownestimatormodel_" + str(yi) + ".sav"
            est = joblib.load(filename)
            yt_proba = est.predict_proba(X)[:, 1:2]
            if i == 0:
                # print(self.probas_.shape)
                self.probas_ = yt_proba
                # print(self.probas_.shape)
            else:
                # print(yt_proba.shape, self.probas_.shape)
                self.probas_ = np.concatenate((self.probas_, yt_proba), axis=1)
                # print(yt_proba.shape, self.probas_.shape)
            print("--------", yi, "--------")
            print(yt_proba)
            i += 1

        print(self.probas_)
            # print(self.probas_.shape)
            # print(self.probas_)

    def fit_predict(self, X, y):
        """
                This should fit classifier. All the "work" should be done here.

                Note: assert is not a good choice here and you should rather
                use try/except blog with exceptions. This is just for short syntax.
                """
        self.sorted_classes_ = np.unique(y)
        # print(self.sorted_classes_)
        self.num_classes_ = len(self.sorted_classes_)
        self.num_instances_ = len(X)
        self.probas_ = np.zeros((self.num_instances_, 1))
        i=0
        for yi in self.sorted_classes_[:-1]:
            # print(yi)
            yt = y.copy()
            yt[yt <= yi] = 0
            yt[yt > yi] = 1
            self.estimator.fit(X, yt.ravel())
            print(self.estimator)
            yt_proba = self.estimator.predict_proba(X)[:, 1:2]
            if i == 0:
                # print(self.probas_.shape)
                self.probas_ = yt_proba
                # print(self.probas_.shape)
            else:
                # print(yt_proba.shape, self.probas_.shape)
                self.probas_ = np.concatenate((self.probas_, yt_proba), axis=1)
                # print(yt_proba.shape, self.probas_.shape)
            print("--------", yi, "--------")
            print(yt_proba)
            i += 1
        ypf = np.zeros((self.num_instances_, 1))
        try:
            getattr(self, "probas_")
            for i in range(self.num_classes_):
                if i == 0:
                    ypi = 1 - self.probas_[:, 0:1]
                elif 0 < i < self.num_classes_ - 1:
                    ypi = self.probas_[:, i - 1:i] - self.probas_[:, i:i + 1]
                elif i == self.num_classes_ - 1:
                    ypi = self.probas_[:, i - 1:i]
                if i == 0:
                    ypf = ypi
                else:
                    ypf = np.concatenate((ypf, ypi), axis=1)
            # print(ypf)
            return np.argmax(ypf, axis=1).reshape(len(ypf), 1)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")


    def predict(self, X):
        self.predict_proba(X)
        ypf = np.zeros((self.num_instances_,1))
        try:
            getattr(self, "probas_")
            for i in range(self.num_classes_):
                if i == 0:
                    ypi = 1 - self.probas_[:, 0:1]
                elif 0 < i < self.num_classes_-1:
                    ypi = self.probas_[:, i-1:i] - self.probas_[:, i:i+1]
                elif i == self.num_classes_-1:
                    ypi = self.probas_[:, i-1:i]
                if i == 0:
                    ypf = ypi
                else:
                    ypf = np.concatenate((ypf,ypi), axis=1)
            #print(ypf)
            return np.argmax(ypf, axis=1).reshape(len(ypf), 1)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
