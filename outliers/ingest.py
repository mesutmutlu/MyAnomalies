from outliers.feature_format import featureFormat, targetFeatureSplit
import pickle
from numpy import genfromtxt
import numpy

#features_list = ['poi','salary','expenses', 'long_term_incentive', 'director_fees', 'restricted_stock_deferred','other','from_poi_to_this_person', 'from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
#with open("C:/Users/dtmemutlu/PycharmProjects/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
#    data_dict = pickle.load(data_file)

    #featureFormat(data_dict, features_list)

if __name__ == "__main__":
    my_data = numpy.fromfile('weblog.csv', sep=",")
    import pandas as pd
    df = pd.read_csv('weblog.csv', sep=',', header='infer' )
#    df._reindex_multi()
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(df.describe(include='all'))
    df.set_index(['Time','IP','URL'], inplace=True)
    print(df)
