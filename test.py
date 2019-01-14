from scipy.optimize import minimize, rosen, rosen_der

from sklearn.feature_extraction import stop_words


from nltk.corpus import stopwords


if __name__ == "__main__":

    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    x1 = [1, 1, 1, 1, 1]

    r = rosen(x0)
    r1 = rosen(x1)

    res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
    print(r1)
    print(r, res.x)

    print(res)

    print(stopwords.words('chiniese'))