from mord import OrdinalRidge
from petfinder.get_explore import read_data
from petfinder.preprocessing import prepare_data
import pandas as pd

if __name__ == "__main__":
    train, test = read_data()
    x_train, y_train, x_test, id_test = prepare_data(train, test)
    clf = OrdinalRidge()

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    print(clf.score(x_train, y_train))
    print(pd.DataFrame({'PetID': id_test.PetID.values.ravel(), 'AdoptionSpeed': pred}))