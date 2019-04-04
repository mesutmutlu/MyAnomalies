import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import math
import psutil
import os
from datetime import datetime
base_path = r"C:\datasets\earthquake"
from multiprocessing import Pool, Manager
import time
train_path = os.path.join(base_path, "train/train.csv")
test_file_path = os.path.join(base_path, "test/")
df_list = []
rows = 10_000
w_sizes = [10, 100, 1000]

def seg_rolling_features(seg_id, seg, w_sizes):
    # print("start", seg_id, datetime.now())
    X_df = pd.DataFrame(index=[seg_id], dtype=np.float64)

    X_df.loc[seg_id, 'mean'] = seg.mean()
    X_df.loc[seg_id, 'std'] = seg.std()
    X_df.loc[seg_id, 'max'] = seg.max()
    X_df.loc[seg_id, 'min'] = seg.min()
    X_df.loc[seg_id, 'var'] = seg.var()
    X_df.loc[seg_id, 'sum'] = seg.sum()

    X_df.loc[seg_id, "quantile_0.2_mean"] = seg.quantile(0.2, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.4_mean_"] = seg.quantile(0.4, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.6_mean"] = seg.quantile(0.6, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.8_mean"] = seg.quantile(0.8, interpolation="linear")

    for window in w_sizes:
        seg_roll_std = seg.rolling(window).std()
        seg_roll_mean = seg.rolling(window).mean()
        seg_roll_var = seg.rolling(window).var()
        seg_roll_min = seg.rolling(window).min()
        seg_roll_max = seg.rolling(window).max()
        seg_roll_sum = seg.rolling(window).sum()

        seg_emw_std = seg.ewm(span=window).std()
        seg_emw_mean = seg.ewm(span=window).mean()
        seg_emw_var = seg.ewm(span=window).var()

        X_df.loc[seg_id, "rolling_min_min_" + str(window)] = seg_roll_min.min()
        X_df.loc[seg_id, "rolling_min_sum_" + str(window)] = seg_roll_min.sum()
        X_df.loc[seg_id, "rolling_min_maseg_" + str(window)] = seg_roll_min.max()
        X_df.loc[seg_id, "rolling_min_std_" + str(window)] = seg_roll_min.std()
        X_df.loc[seg_id, "rolling_min_var_" + str(window)] = seg_roll_min.var()
        X_df.loc[seg_id, "rolling_min_mean_" + str(window)] = seg_roll_min.mean()
        X_df.loc[seg_id, "rolling_min_quan02_" + str(window)] = seg_roll_min.quantile(0.2)
        X_df.loc[seg_id, "rolling_min_quan04_" + str(window)] = seg_roll_min.quantile(0.4)
        X_df.loc[seg_id, "rolling_min_quan06_" + str(window)] = seg_roll_min.quantile(0.6)
        X_df.loc[seg_id, "rolling_min_quan08_" + str(window)] = seg_roll_min.quantile(0.8)

        X_df.loc[seg_id, "rolling_maseg_min_" + str(window)] = seg_roll_max.min()
        X_df.loc[seg_id, "rolling_maseg_sum_" + str(window)] = seg_roll_max.sum()
        X_df.loc[seg_id, "rolling_maseg_maseg_" + str(window)] = seg_roll_max.max()
        X_df.loc[seg_id, "rolling_maseg_std_" + str(window)] = seg_roll_max.std()
        X_df.loc[seg_id, "rolling_maseg_var_" + str(window)] = seg_roll_max.var()
        X_df.loc[seg_id, "rolling_maseg_mean_" + str(window)] = seg_roll_max.mean()
        X_df.loc[seg_id, "rolling_maseg_quan02_" + str(window)] = seg_roll_max.quantile(0.2)
        X_df.loc[seg_id, "rolling_maseg_quan04_" + str(window)] = seg_roll_max.quantile(0.4)
        X_df.loc[seg_id, "rolling_maseg_quan06_" + str(window)] = seg_roll_max.quantile(0.6)
        X_df.loc[seg_id, "rolling_maseg_quan08_" + str(window)] = seg_roll_max.quantile(0.8)

        X_df.loc[seg_id, "rolling_std_min_" + str(window)] = seg_roll_std.min()
        X_df.loc[seg_id, "rolling_std_sum_" + str(window)] = seg_roll_std.sum()
        X_df.loc[seg_id, "rolling_std_maseg_" + str(window)] = seg_roll_std.max()
        X_df.loc[seg_id, "rolling_std_std_" + str(window)] = seg_roll_std.std()
        X_df.loc[seg_id, "rolling_std_var_" + str(window)] = seg_roll_std.var()
        X_df.loc[seg_id, "rolling_std_mean_" + str(window)] = seg_roll_std.mean()
        X_df.loc[seg_id, "rolling_std_quan02_" + str(window)] = seg_roll_std.quantile(0.2)
        X_df.loc[seg_id, "rolling_std_quan04_" + str(window)] = seg_roll_std.quantile(0.4)
        X_df.loc[seg_id, "rolling_std_quan06_" + str(window)] = seg_roll_std.quantile(0.6)
        X_df.loc[seg_id, "rolling_std_quan08_" + str(window)] = seg_roll_std.quantile(0.8)

        X_df.loc[seg_id, "rolling_sum_min_" + str(window)] = seg_roll_sum.min()
        X_df.loc[seg_id, "rolling_sum_sum_" + str(window)] = seg_roll_sum.sum()
        X_df.loc[seg_id, "rolling_sum_maseg_" + str(window)] = seg_roll_sum.max()
        X_df.loc[seg_id, "rolling_sum_sum_" + str(window)] = seg_roll_sum.std()
        X_df.loc[seg_id, "rolling_sum_var_" + str(window)] = seg_roll_sum.var()
        X_df.loc[seg_id, "rolling_sum_mean_" + str(window)] = seg_roll_sum.mean()
        X_df.loc[seg_id, "rolling_sum_quan02_" + str(window)] = seg_roll_sum.quantile(0.2)
        X_df.loc[seg_id, "rolling_sum_quan04_" + str(window)] = seg_roll_sum.quantile(0.4)
        X_df.loc[seg_id, "rolling_sum_quan06_" + str(window)] = seg_roll_sum.quantile(0.6)
        X_df.loc[seg_id, "rolling_sum_quan08_" + str(window)] = seg_roll_sum.quantile(0.8)

        X_df.loc[seg_id, "rolling_var_min_" + str(window)] = seg_roll_var.min()
        X_df.loc[seg_id, "rolling_var_sum_" + str(window)] = seg_roll_var.sum()
        X_df.loc[seg_id, "rolling_var_maseg_" + str(window)] = seg_roll_var.max()
        X_df.loc[seg_id, "rolling_var_std_" + str(window)] = seg_roll_var.std()
        X_df.loc[seg_id, "rolling_var_var_" + str(window)] = seg_roll_var.var()
        X_df.loc[seg_id, "rolling_var_mean_" + str(window)] = seg_roll_var.mean()
        X_df.loc[seg_id, "rolling_var_quan02_" + str(window)] = seg_roll_var.quantile(0.2)
        X_df.loc[seg_id, "rolling_var_quan04_" + str(window)] = seg_roll_var.quantile(0.4)
        X_df.loc[seg_id, "rolling_var_quan06_" + str(window)] = seg_roll_var.quantile(0.6)
        X_df.loc[seg_id, "rolling_var_quan08_" + str(window)] = seg_roll_var.quantile(0.8)

        X_df.loc[seg_id, "rolling_mean_min_" + str(window)] = seg_roll_mean.min()
        X_df.loc[seg_id, "rolling_mean_sum_" + str(window)] = seg_roll_mean.sum()
        X_df.loc[seg_id, "rolling_mean_maseg_" + str(window)] = seg_roll_mean.max()
        X_df.loc[seg_id, "rolling_mean_std_" + str(window)] = seg_roll_mean.std()
        X_df.loc[seg_id, "rolling_mean_var_" + str(window)] = seg_roll_mean.var()
        X_df.loc[seg_id, "rolling_mean_mean_" + str(window)] = seg_roll_mean.mean()
        X_df.loc[seg_id, "rolling_mean_quan02_" + str(window)] = seg_roll_mean.quantile(0.2)
        X_df.loc[seg_id, "rolling_mean_quan04_" + str(window)] = seg_roll_mean.quantile(0.4)
        X_df.loc[seg_id, "rolling_mean_quan06_" + str(window)] = seg_roll_mean.quantile(0.6)
        X_df.loc[seg_id, "rolling_mean_quan08_" + str(window)] = seg_roll_mean.quantile(0.8)

        X_df.loc[seg_id, "ewm_mean_min_" + str(window)] = seg_emw_mean.min()
        X_df.loc[seg_id, "ewm_mean_sum_" + str(window)] = seg_emw_mean.sum()
        X_df.loc[seg_id, "ewm_mean_maseg_" + str(window)] = seg_emw_mean.max()
        X_df.loc[seg_id, "ewm_mean_std_" + str(window)] = seg_emw_mean.std()
        X_df.loc[seg_id, "ewm_mean_var_" + str(window)] = seg_emw_mean.var()
        X_df.loc[seg_id, "ewm_mean_mean_" + str(window)] = seg_emw_mean.mean()
        X_df.loc[seg_id, "ewm_mean_quan02_" + str(window)] = seg_emw_mean.quantile(0.2)
        X_df.loc[seg_id, "ewm_mean_quan04_" + str(window)] = seg_emw_mean.quantile(0.4)
        X_df.loc[seg_id, "ewm_mean_quan06_" + str(window)] = seg_emw_mean.quantile(0.6)
        X_df.loc[seg_id, "ewm_mean_quan08_" + str(window)] = seg_emw_mean.quantile(0.8)

        X_df.loc[seg_id, "ewm_std_min_" + str(window)] = seg_emw_std.min()
        X_df.loc[seg_id, "ewm_std_sum_" + str(window)] = seg_emw_std.sum()
        X_df.loc[seg_id, "ewm_std_maseg_" + str(window)] = seg_emw_std.max()
        X_df.loc[seg_id, "ewm_std_std_" + str(window)] = seg_emw_std.std()
        X_df.loc[seg_id, "ewm_std_var_" + str(window)] = seg_emw_std.var()
        X_df.loc[seg_id, "ewm_std_mean_" + str(window)] = seg_emw_std.mean()
        X_df.loc[seg_id, "ewm_std_quan02_" + str(window)] = seg_emw_std.quantile(0.2)
        X_df.loc[seg_id, "ewm_std_quan04_" + str(window)] = seg_emw_std.quantile(0.4)
        X_df.loc[seg_id, "ewm_std_quan06_" + str(window)] = seg_emw_std.quantile(0.6)
        X_df.loc[seg_id, "ewm_std_quan08_" + str(window)] = seg_emw_std.quantile(0.8)

        X_df.loc[seg_id, "ewm_var_min_" + str(window)] = seg_emw_var.min()
        X_df.loc[seg_id, "ewm_var_sum_" + str(window)] = seg_emw_var.sum()
        X_df.loc[seg_id, "ewm_var_maseg_" + str(window)] = seg_emw_var.max()
        X_df.loc[seg_id, "ewm_var_std_" + str(window)] = seg_emw_var.std()
        X_df.loc[seg_id, "ewm_var_var_" + str(window)] = seg_emw_var.var()
        X_df.loc[seg_id, "ewm_var_mean_" + str(window)] = seg_emw_var.mean()
        X_df.loc[seg_id, "ewm_var_quan02_" + str(window)] = seg_emw_var.quantile(0.2)
        X_df.loc[seg_id, "ewm_var_quan04_" + str(window)] = seg_emw_var.quantile(0.4)
        X_df.loc[seg_id, "ewm_var_quan06_" + str(window)] = seg_emw_var.quantile(0.6)
        X_df.loc[seg_id, "ewm_var_quan08_" + str(window)] = seg_emw_var.quantile(0.8)
    return X_df

def collect_result(result):
    df_list.append(result)

def calc_train_features(train, df_list):
    print("Start date", datetime.now())
    segments = int(np.floor(train.shape[0] / rows))

    y_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
    # X_train = pd.DataFrame(index=range(segments), dtype=np.float64)
    X_train = None

    p = Pool(4)

    # for seg_id in tqdm_notebook(range(segments)):
    i = 1
    for seg_id in range(segments):
        if i< 6:
            seg = train.iloc[seg_id * rows:seg_id * rows + rows]

            y_train.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

            # x_seg = pd.Series(seg['acoustic_data'].values)
            # X_train = seg_rolling_features(seg_id, seg['acoustic_data'], [10, 100, 1000], X_train)
            p.apply_async(seg_rolling_features, [seg_id, seg['acoustic_data'], w_sizes], callback=collect_result)
            i += 1
        else:
            break
    p.close()
    p.join()

    for df in df_list:
        if X_train is None:
            X_train = df
        else:
            X_train = pd.concat([X_train, df])

    print("End date", datetime.now())

    print("X_train shape:", X_train.shape)
    print(X_train.head())
    X_train.to_csv("./output/xtrain.csv", index=False)
    print("y_train shape:", y_train.shape)
    print(y_train.head())
    y_train.to_csv("./output/ytrain.csv", index=False)

    return X_train, y_train

def calc_test_features(df_list):
    print("Start date", datetime.now())

    submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'), index_col='seg_id')
    # X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
    X_test = None
    p = Pool(2)

    m = Manager()
    q = m.Queue()
    # for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
    i = 0
    for i, seg_id in enumerate(submission.index):

        if i < 6:
            seg = pd.read_csv(test_file_path + seg_id + '.csv')['acoustic_data'].values
            x_seg = pd.Series(seg)
            # X_test = seg_rolling_features(seg_id, x_seg, [10, 100, 1000], X_test)
            res = p.apply_async(seg_rolling_features, [seg_id, x_seg, w_sizes], callback=collect_result)
            print('I am on number ' + str(i + 1) + ' of ' + str(len(submission.index)))
            i += 1
        else:
            break
    if 1 == 1:
        while True:
            if res.ready():
                break
            else:
                size = q.qsize()
                print(size)
                time.sleep(2)

    p.close()
    p.join()
    print(df_list)
    for df in df_list:
        if X_test is None:
            X_test = df
        else:
            X_test = pd.concat([X_test, df])

    print("End date", datetime.now())

    print("X_test shape:", X_test.shape)
    print(X_test.head())
    X_test.to_csv("./output/xtest.csv", index=False)
    return X_test



if __name__ == "__main__":

    train = pd.read_csv(os.path.join(base_path, "train/train.csv.010"), header=None)

    train.columns = ["acoustic_data", "time_to_failure"]

    calc_test_features(df_list)
