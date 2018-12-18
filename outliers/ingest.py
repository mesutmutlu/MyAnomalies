from outliers.feature_format import featureFormat, targetFeatureSplit
import pickle
from numpy import genfromtxt
import numpy as np
import datetime
import time
import re
import sys
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#features_list = ['poi','salary','expenses', 'long_term_incentive', 'director_fees', 'restricted_stock_deferred','other','from_poi_to_this_person', 'from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
#with open("C:/Users/dtmemutlu/PycharmProjects/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
#    data_dict = pickle.load(data_file)

    #featureFormat(data_dict, features_list)

def df_dropindex(df, arr):
    #arr is the tuples list of index name and its level like
    #[('chmod', 0),(4, 1)]
    for t in arr:
        try:
            df.drop(labels=t[0], level=t[1], inplace=True, axis='index')
        except:
            pass
    return df

def pd_dropcol():
    pass

def checkstrtimeformat(str, format):
    try:
        validtime = datetime.strptime(str, format)
        return True
    except ValueError:
        return False

def checkregex(str, pattern):
    p = re.compile(pattern)
    if p.match(str):
        return True
    else:
        return False

def checkurl(str):
    if str.startswith('GET') or str.startswith('POST') or str.startswith('HEAD'):
        return True
    else:
        return False

def ajustTimeMinutes(tm, minutes):

    tm = tm - timedelta(minutes=tm.minute % minutes,
                                     seconds=tm.second,
                                     microseconds=tm.microsecond)
    return tm

def clean_web_server_log():
    my_data = np.fromfile('weblog.csv', sep=",")
    df = pd.read_csv('weblog.csv', sep=',', header='infer')

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # print(df.describe(include='all'))
    df.set_index(['Time', 'IP', 'URL'], inplace=True)
    # print(df.index.levels[0])
    # print(df.index.levels)
    # print(df.drop(labels='[Fri', level=1, axis='index'))
    # sys.exit()
    i = 0
    for levels in df.index.levels:
        if levels.name == 'Time':
            format = '[%d/%b/%Y:%H:%M:%S'
            for time in levels:
                if not checkstrtimeformat(time, format):
                    # print(time)
                    df = df_dropindex(df, [(time, 0)])

        elif levels.name == 'IP':
            pattern = '[%d/%b/%Y:%H:%M:%S'
            for IP in levels:
                if not checkregex(IP,
                                  '^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$'):
                    df = df_dropindex(df, [(IP, 1)])
            pass
        elif levels.name == 'URL':
            for URL in levels:
                if not checkurl(URL):
                    df = df_dropindex(df, [(URL, 2)])
    # print(df)
    df.reset_index(inplace=True)
    #df['Time'].values = checkstrtimeformat(df['Time'].values, format)
    df['Time'] = df.apply(lambda x: (datetime.strptime(x.Time, format)), axis=1)
    df['AdjustedInMinutes'] = df.apply(lambda x: (ajustTimeMinutes(x.Time, 10)), axis=1)
    df['AdjustedInHour'] = df.apply(lambda x: (ajustTimeMinutes(x.Time, 60)), axis=1)
    df['AdjustedInDay'] = df.apply(lambda x: (datetime.strptime(x.Time.strftime('%d/%m/%Y'),'%d/%m/%Y')), axis=1)
    #df.apply(lambda x: (x.Value + (x.Value * 0.006)) if x.Order == 'SELL' else (x.Value - (x.Value * 0.006)), axis=1)
    url = df['URL'].str.split(" ", expand=True )
    df['URL'] = url[1].str.split("?", expand=True)[0]
    df['Method'] = url[0]
    df['HTTP'] = url[2]
    os.remove('weblogcl.csv')
    df.to_csv('weblogcl.csv', sep=',')
    return df

def feature_engineering(dh):
    df_staus = df.groupby(['AdjustedInDay', 'URL', 'Staus']).agg({'Staus': {'urlrescnt': 'size'}})
    # df = df.groupby(['AdjustedInDay', 'URL'])['Staus'].agg({'Staus': {'urlcnt': 'size'}})
    df_staus.columns = df_staus.columns.droplevel(0)
    df_staus.reset_index(inplace=True)
    df_staus.set_index(['AdjustedInDay', 'URL'])
    # print(df_staus)
    df_url = df.groupby(['AdjustedInDay', 'URL']).agg({'URL': {'urlcallcnt': 'size'}})
    df_url.columns = df_url.columns.droplevel(0)
    df_url.reset_index(inplace=True)
    df_url.set_index(['AdjustedInDay', 'URL'])
    # print(df_url)
    # df2 = df.reset_index(inplace=True)
    # print(len(df_url))
    # print(len(df_staus))
    dh = df_url.merge(df_staus, on=['AdjustedInDay', 'URL'], how='inner')
    dh = dh.sort_values(by='AdjustedInDay')
    # print(len(dh))
    # print (dh[dh['URL']=='/home.php'])
    #dh.plot(kind='line', x='AdjustedInDay', y='urlcallcnt')
    # print(dh.AdjustedInDay.unique())
    # print(dh.Staus.unique())
    dh = dh.pivot_table(values='urlrescnt', index=['AdjustedInDay', 'URL', 'urlcallcnt'], columns='Staus', fill_value=0)
    dh.reset_index(inplace=True)
    return dh


if __name__ == "__main__":

    df = clean_web_server_log()
    dh = feature_engineering(df)
    print(dh)
    print(dh.URL.nunique())
    dh.plot(kind='line', x='AdjustedInDay' ,y=['urlcallcnt'],secondary_y=['206'])
    plt.show()


