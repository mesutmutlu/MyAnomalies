from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from functools import partial
from math import sqrt
from collections import Counter
import numpy as np
import scipy as sp
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import pandas as pd
from petfinder.preprocessing import prepare_data
from petfinder.get_explore import read_data, Columns, Paths
import random
import datetime

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):
        x = a - min_rating
        y = b - min_rating
        conf_mat[x[0]][y[0]] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        x = r - min_rating
        hist_ratings[x[0]] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p


def coefficients(self):
    return self.coef_['x']


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model', n_splits=5, n_repeats=2):
    kf = RepeatedStratifiedKFold(n_splits=n_splits, random_state=42, n_repeats = n_repeats)
    fold_splits = kf.split(train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], n_splits*n_repeats))
    all_coefficients = np.zeros((n_splits*n_repeats, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        #print('Started ' + label + ' fold ' + str(i) + '/'+str(n_splits*n_repeats))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i-1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            #print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = train.columns.values
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        i += 1
    #print('{} cv RMSE scores : {}'.format(label, cv_scores))
    #print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    #print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
    #print('{} cv QWK scores : {}'.format(label, qwk_scores))
    #print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    #print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / (n_splits*n_repeats)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}
    return results

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    #print('Prep LGB')
    #cols = Columns.ind_num_cat_columns.value
    #cols.remove("RescuerID")
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    #print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    #print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients)
    #print("Valid Counts = ", Counter(test_y))
    #print("Predicted Counts = ", Counter(pred_test_y_k))
    #print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k)
    #print("QWK = ", qwk)
    #print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(), coefficients, qwk


def by_regressor_rs(train, test, y_train, runALG, metric, name, cv, i, id_test):
    mqwk = 0
    j = 0
    for j in range(10):
        params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': random.randint(50, 200),
              'max_depth': random.randint(5, 25),
              'learning_rate': random.uniform(0.001, 0.05),
              'bagging_fraction': random.uniform(0.5, 1),
              'feature_fraction': random.uniform(0.5, 1),
              'min_split_gain': random.uniform(0.001, 0.05),
              'min_child_samples': random.randint(75, 200),
              'min_child_weight': random.uniform(0.001, 0.05),
              'verbosity': -1,
              'data_random_seed': 3,
              'early_stop': 100,
              'verbose_eval': False,
              'n_jobs':4,
              'lambda_l2': random.uniform(0.001, 0.1),
              'num_rounds': 10000}
        print("RS prms", params)
        results_t = run_cv_model(train, test, y_train, runALG, params, metric, name, cv, i)
        print("rs mean qwk scores", np.mean(results_t["qwk"]), 'rs mean rmse',np.mean(results_t["cv"]))
        if np.mean(results_t["qwk"]) > mqwk:
            results = results_t
            mqwk = np.mean(results_t["qwk"])
    optR = OptimizedRounder()
    coefficients_ = np.mean(results['coefficients'], axis=0)
    #print(coefficients_)
    train_predictions = [r[0] for r in results['train']]
    train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
    Counter(train_predictions)

    optR = OptimizedRounder()
    test_predictions = [r[0] for r in results['test']]
    test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
    Counter(test_predictions)

    pd.DataFrame(sk_cmatrix(y_train, train_predictions), index=list(range(5)), columns=list(range(5)))
    submission = pd.DataFrame({'PetID': id_test.PetID.values, 'AdoptionSpeed': test_predictions})
    return submission


def by_regressor(train, test, y_train, runALG, prms, metric, name, cv, i, id_test):
    results = run_cv_model(train, test, y_train, runALG, prms, metric, name, cv, i)
    print("mean qwk scores", np.mean(results["qwk"]), 'mean rmse', np.mean(results["cv"]))
    optR = OptimizedRounder()
    coefficients_ = np.mean(results['coefficients'], axis=0)
    #print(coefficients_)
    train_predictions = [r[0] for r in results['train']]
    train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
    Counter(train_predictions)

    optR = OptimizedRounder()
    test_predictions = [r[0] for r in results['test']]
    test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
    Counter(test_predictions)

    pd.DataFrame(sk_cmatrix(y_train, train_predictions), index=list(range(5)), columns=list(range(5)))
    submission = pd.DataFrame({'PetID': id_test.PetID.values, 'AdoptionSpeed': test_predictions})
    return submission

if __name__ == "__main__":

    train, test = read_data()
    x_train, y_train, x_test, id_test = prepare_data(train, test)
    print(x_train.columns.values)
    if 1 == 1:
        params = {'application': 'regression',
                  'boosting': 'gbdt',
                  'metric': 'rmse',
                  'num_leaves': 80,
                  'max_depth': 9,
                  'learning_rate': 0.01,
                  'bagging_fraction': 0.85,
                  'feature_fraction': 0.8,
                  'min_split_gain': 0.01,
                  'min_child_samples': 150,
                  'min_child_weight': 0.1,
                  'verbosity': -1,
                  'data_random_seed': 3,
                  'early_stop': 100,
                  'verbose_eval': False,
                  'n_jobs': 4,
                  # 'lambda_l2': 0.05,
                  'num_rounds': 10000,
                  #'categorical_feature':Columns.ind_num_cat_columns.value
                  }
        # print("with all columns", datetime.datetime.now())
        # x_train_a = x_train
        # x_test_a = x_test
        # submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test)
        #
        # cols = Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
        # for c in cols:
        #     print("without" + c, datetime.datetime.now())
        #     x_train_a = x_train.drop([c], axis=1)
        #     x_test_a = x_test.drop([c], axis=1)
        #     submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test)

        print("without img_num_cols_1", datetime.datetime.now())
        x_train_a = x_train.drop(Columns.img_num_cols_1.value, axis=1)
        x_test_a = x_test.drop(Columns.img_num_cols_1.value, axis=1)
        submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test)

        print("without img_num_cols_2", datetime.datetime.now())
        x_train_a = x_train.drop(Columns.img_num_cols_2.value, axis=1)
        x_test_a = x_test.drop(Columns.img_num_cols_2.value, axis=1)
        submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test)

        print("without img_num_cols_3", datetime.datetime.now())
        x_train_a = x_train.drop(Columns.img_num_cols_3.value, axis=1)
        x_test_a = x_test.drop(Columns.img_num_cols_3.value, axis=1)
        submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test)

        print("without iann_cols", datetime.datetime.now())
        x_train_a = x_train.drop(Columns.iann_cols.value, axis=1)
        x_test_a = x_test.drop(Columns.iann_cols.value, axis=1)
        submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test)
        print("ended", datetime.datetime.now())