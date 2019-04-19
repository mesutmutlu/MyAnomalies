import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import math
import psutil
import os
import time
from datetime import datetime
base_path = r"C:\datasets\earthquake"
from multiprocessing import Pool, Manager
import time
train_path = os.path.join(base_path, "train/train.csv")
test_file_path = os.path.join(base_path, "test/")
rows = 10_000
w_sizes = [10, 100, 1000, 3000, 5000]
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from obspy.signal.trigger import classic_sta_lta
df_list = []

def seg_rolling_features(seg_id, seg, w_sizes):
    print("start", seg_id, datetime.now())
    X_df = pd.DataFrame(index=[seg_id], dtype=np.float64)

    X_df.loc[seg_id, 'mean'] = seg.mean()
    X_df.loc[seg_id, 'std'] = seg.std()
    X_df.loc[seg_id, 'max'] = seg.max()
    X_df.loc[seg_id, 'min'] = seg.min()
    X_df.loc[seg_id, 'var'] = seg.var()
    X_df.loc[seg_id, 'sum'] = seg.sum()

    for d in [1, 5, 50, 100, 500, 1000, 2000, 5000]:
        npdiff = np.diff(seg, n=d)
        X_df.loc[seg_id, 'npdiff_mean_' + str(d)] = npdiff.mean()
        X_df.loc[seg_id, 'npdiff_std_' + str(d)] = npdiff.std()
        X_df.loc[seg_id, 'npdiff_max_' + str(d)] = npdiff.max()
        X_df.loc[seg_id, 'npdiff_min_' + str(d)] = npdiff.min()
        X_df.loc[seg_id, 'npdiff_var_' + str(d)] = npdiff.var()
        X_df.loc[seg_id, 'npdiff_sum_' + str(d)] = npdiff.sum()
        X_df.loc[seg_id, 'npdiff_mad_' + str(d)] = pd.Series(npdiff).mad()

        if d != 1:
            pct = seg.pct_change(periods=d)
            X_df.loc[seg_id, 'pct_change_mean_' + str(d)] = pct.mean()
            X_df.loc[seg_id, 'pct_change_std_' + str(d)] = pct.std()
            X_df.loc[seg_id, 'pct_change_max_' + str(d)] = pct.max()
            X_df.loc[seg_id, 'pct_change_min_' + str(d)] = pct.min()
            X_df.loc[seg_id, 'pct_change_var_' + str(d)] = pct.var()
            X_df.loc[seg_id, 'pct_change_sum_' + str(d)] = pct.sum()
            X_df.loc[seg_id, 'pct_change_mad_' + str(d)] = pct.mad()

            sdiff = seg.diff(periods=d)
            X_df.loc[seg_id, 'sdiff_mean_' + str(d)] = sdiff.mean()
            X_df.loc[seg_id, 'sdiff_std_' + str(d)] = sdiff.std()
            X_df.loc[seg_id, 'sdiff_max_' + str(d)] = sdiff.max()
            X_df.loc[seg_id, 'sdiff_min_' + str(d)] = sdiff.min()
            X_df.loc[seg_id, 'sdiff_var_' + str(d)] = sdiff.var()
            X_df.loc[seg_id, 'sdiff_sum_' + str(d)] = sdiff.sum()
            X_df.loc[seg_id, 'sdiff_mad_' + str(d)] = sdiff.mad()

    X_df.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(seg) / seg[:-1]))[0])
    X_df.loc[seg_id, 'abs_max'] = np.abs(seg).max()
    X_df.loc[seg_id, 'abs_min'] = np.abs(seg).min()
    X_df.loc[seg_id, 'max_to_min'] = seg.max() / np.abs(seg.min())
    X_df.loc[seg_id, 'max_to_min_diff'] = seg.max() - np.abs(seg.min())
    X_df.loc[seg_id, 'count_big'] = len(seg[np.abs(seg) > 500])

    X_df.loc[seg_id, "quantile_0.001"] = seg.quantile(0.001, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.01"] = seg.quantile(0.01, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.05"] = seg.quantile(0.05, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.1"] = seg.quantile(0.1, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.2"] = seg.quantile(0.2, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.4"] = seg.quantile(0.4, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.6"] = seg.quantile(0.6, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.8"] = seg.quantile(0.8, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.9"] = seg.quantile(0.9, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.95"] = seg.quantile(0.95, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.99"] = seg.quantile(0.99, interpolation="linear")
    X_df.loc[seg_id, "quantile_0.999"] = seg.quantile(0.999, interpolation="linear")

    X_df.loc[seg_id, 'abs_mean'] = np.abs(seg).mean()
    X_df.loc[seg_id, 'abs_std'] = np.abs(seg).std()

    X_df.loc[seg_id, 'mad'] = seg.mad()
    X_df.loc[seg_id, 'kurt'] = seg.kurtosis()
    X_df.loc[seg_id, 'skew'] = seg.skew()

    X_df.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(seg)).mean()
    X_df.loc[seg_id, 'Hann_window_mean'] = (convolve(seg, hann(150), mode='same') / sum(hann(150))).mean()

    X_df.loc[seg_id, 'count_big'] = len(seg[np.abs(seg) > 500])

    for sta_lta in [(500, 10000), (500, 20000), (3333, 6666), (3333, 9999), (5000, 50000), (5000, 100000),
                    (10000, 25000), (10000, 100000)]:
        if (len(seg) > sta_lta[0]) & (len(seg) > sta_lta[1]):

            stalta = classic_sta_lta(seg, sta_lta[0], sta_lta[1])
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_mean'] = stalta.mean()
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_std'] = stalta.std()
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_max'] = stalta.max()
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_min'] = stalta.min()
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_var'] = stalta.var()
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_sum'] = stalta.sum()
            X_df.loc[seg_id, 'classic_sta_' + str(sta_lta[0]) + '_lta_' + str(sta_lta[1]) + '_mad'] = pd.Series(stalta).mad()

            conv = np.convolve(seg[-sta_lta[0]:], seg[-sta_lta[1]:])
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_sum'] = conv.sum()
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_std'] = conv.std()
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_max'] = conv.max()
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_min'] = conv.min()
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_var'] = conv.var()
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_sum'] = conv.sum()
            X_df.loc[seg_id, 'convolve_' + str(sta_lta[0]) + '_' + str(sta_lta[1]) + '_mad'] = pd.Series(conv).mad()

    for slice in [50, 100, 1000, 10000, 25000, 50000]:
        X_df.loc[seg_id, 'var_first_' + str(slice)] = seg[:slice].var()
        X_df.loc[seg_id, 'var_last_' + str(slice)] = seg[-1 * slice:].var()
        X_df.loc[seg_id, 'std_first_' + str(slice)] = seg[:slice].std()
        X_df.loc[seg_id, 'std_last_' + str(slice)] = seg[-1 * slice:].std()
        X_df.loc[seg_id, 'mean_first_' + str(slice)] = seg[:slice].mean()
        X_df.loc[seg_id, 'mean_last_' + str(slice)] = seg[-1 * slice:].mean()
        X_df.loc[seg_id, 'min_first_' + str(slice)] = seg[:slice].min()
        X_df.loc[seg_id, 'min_last_' + str(slice)] = seg[-1 * slice:].min()
        X_df.loc[seg_id, 'max_first_' + str(slice)] = seg[:slice].max()
        X_df.loc[seg_id, 'max_last_' + str(slice)] = seg[-1 * slice:].max()
        X_df.loc[seg_id, 'mean_change_rate_first_' + str(slice)] = np.mean(
            np.nonzero((np.diff(seg[:slice]) / seg[:slice][:-1]))[0])
        X_df.loc[seg_id, 'mean_change_rate_last_' + str(slice)] = np.mean(
            np.nonzero((np.diff(seg[-1 * slice:]) / seg[-1 * slice:][:-1]))[0])

    quantile_list = [0.001, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.999]
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
        X_df.loc[seg_id, "rolling_min_max_" + str(window)] = seg_roll_min.max()
        X_df.loc[seg_id, "rolling_min_std_" + str(window)] = seg_roll_min.std()
        X_df.loc[seg_id, "rolling_min_var_" + str(window)] = seg_roll_min.var()
        X_df.loc[seg_id, "rolling_min_mean_" + str(window)] = seg_roll_min.mean()
        X_df.loc[seg_id, "rolling_min_mad_" + str(window)] = seg_roll_min.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "rolling_min_quan" + str(q) + "_" + str(window)] = seg_roll_min.quantile(q)

        X_df.loc[seg_id, "rolling_max_min_" + str(window)] = seg_roll_max.min()
        X_df.loc[seg_id, "rolling_max_sum_" + str(window)] = seg_roll_max.sum()
        X_df.loc[seg_id, "rolling_max_max_" + str(window)] = seg_roll_max.max()
        X_df.loc[seg_id, "rolling_max_std_" + str(window)] = seg_roll_max.std()
        X_df.loc[seg_id, "rolling_max_var_" + str(window)] = seg_roll_max.var()
        X_df.loc[seg_id, "rolling_max_mean_" + str(window)] = seg_roll_max.mean()
        X_df.loc[seg_id, "rolling_max_mad_" + str(window)] = seg_roll_max.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "rolling_max_quan" + str(q) + "_" + str(window)] = seg_roll_max.quantile(q)

        X_df.loc[seg_id, "rolling_std_min_" + str(window)] = seg_roll_std.min()
        X_df.loc[seg_id, "rolling_std_sum_" + str(window)] = seg_roll_std.sum()
        X_df.loc[seg_id, "rolling_std_max_" + str(window)] = seg_roll_std.max()
        X_df.loc[seg_id, "rolling_std_std_" + str(window)] = seg_roll_std.std()
        X_df.loc[seg_id, "rolling_std_var_" + str(window)] = seg_roll_std.var()
        X_df.loc[seg_id, "rolling_std_mean_" + str(window)] = seg_roll_std.mean()
        X_df.loc[seg_id, "rolling_std_mad_" + str(window)] = seg_roll_std.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "rolling_std_quan" + str(q) + "_" + str(window)] = seg_roll_std.quantile(q)

        X_df.loc[seg_id, "rolling_sum_min_" + str(window)] = seg_roll_sum.min()
        X_df.loc[seg_id, "rolling_sum_sum_" + str(window)] = seg_roll_sum.sum()
        X_df.loc[seg_id, "rolling_sum_max_" + str(window)] = seg_roll_sum.max()
        X_df.loc[seg_id, "rolling_sum_sum_" + str(window)] = seg_roll_sum.std()
        X_df.loc[seg_id, "rolling_sum_var_" + str(window)] = seg_roll_sum.var()
        X_df.loc[seg_id, "rolling_sum_mean_" + str(window)] = seg_roll_sum.mean()
        X_df.loc[seg_id, "rolling_sum_mad_" + str(window)] = seg_roll_sum.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "rolling_sum_quan" + str(q) + "_" + str(window)] = seg_roll_sum.quantile(q)

        X_df.loc[seg_id, "rolling_var_min_" + str(window)] = seg_roll_var.min()
        X_df.loc[seg_id, "rolling_var_sum_" + str(window)] = seg_roll_var.sum()
        X_df.loc[seg_id, "rolling_var_max_" + str(window)] = seg_roll_var.max()
        X_df.loc[seg_id, "rolling_var_std_" + str(window)] = seg_roll_var.std()
        X_df.loc[seg_id, "rolling_var_var_" + str(window)] = seg_roll_var.var()
        X_df.loc[seg_id, "rolling_var_mean_" + str(window)] = seg_roll_var.mean()
        X_df.loc[seg_id, "rolling_var_mad_" + str(window)] = seg_roll_var.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "rolling_var_quan" + str(q) + "_" + str(window)] = seg_roll_var.quantile(q)

        X_df.loc[seg_id, "rolling_mean_min_" + str(window)] = seg_roll_mean.min()
        X_df.loc[seg_id, "rolling_mean_sum_" + str(window)] = seg_roll_mean.sum()
        X_df.loc[seg_id, "rolling_mean_max_" + str(window)] = seg_roll_mean.max()
        X_df.loc[seg_id, "rolling_mean_std_" + str(window)] = seg_roll_mean.std()
        X_df.loc[seg_id, "rolling_mean_var_" + str(window)] = seg_roll_mean.var()
        X_df.loc[seg_id, "rolling_mean_mean_" + str(window)] = seg_roll_mean.mean()
        X_df.loc[seg_id, "rolling_mean_mad_" + str(window)] = seg_roll_var.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "rolling_mean_quan" + str(q) + "_" + str(window)] = seg_roll_mean.quantile(q)

        X_df.loc[seg_id, "ewm_mean_min_" + str(window)] = seg_emw_mean.min()
        X_df.loc[seg_id, "ewm_mean_sum_" + str(window)] = seg_emw_mean.sum()
        X_df.loc[seg_id, "ewm_mean_max_" + str(window)] = seg_emw_mean.max()
        X_df.loc[seg_id, "ewm_mean_std_" + str(window)] = seg_emw_mean.std()
        X_df.loc[seg_id, "ewm_mean_var_" + str(window)] = seg_emw_mean.var()
        X_df.loc[seg_id, "ewm_mean_mean_" + str(window)] = seg_emw_mean.mean()
        X_df.loc[seg_id, "ewm_mean_mad_" + str(window)] = seg_emw_mean.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "ewm_mean_quan" + str(q) + "_" + str(window)] = seg_emw_mean.quantile(q)

        X_df.loc[seg_id, "ewm_std_min_" + str(window)] = seg_emw_std.min()
        X_df.loc[seg_id, "ewm_std_sum_" + str(window)] = seg_emw_std.sum()
        X_df.loc[seg_id, "ewm_std_max_" + str(window)] = seg_emw_std.max()
        X_df.loc[seg_id, "ewm_std_std_" + str(window)] = seg_emw_std.std()
        X_df.loc[seg_id, "ewm_std_var_" + str(window)] = seg_emw_std.var()
        X_df.loc[seg_id, "ewm_std_mean_" + str(window)] = seg_emw_std.mean()
        X_df.loc[seg_id, "ewm_std_mad_" + str(window)] = seg_emw_std.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "ewm_std_quan" + str(q) + "_" + str(window)] = seg_emw_std.quantile(q)

        X_df.loc[seg_id, "ewm_var_min_" + str(window)] = seg_emw_var.min()
        X_df.loc[seg_id, "ewm_var_sum_" + str(window)] = seg_emw_var.sum()
        X_df.loc[seg_id, "ewm_var_max_" + str(window)] = seg_emw_var.max()
        X_df.loc[seg_id, "ewm_var_std_" + str(window)] = seg_emw_var.std()
        X_df.loc[seg_id, "ewm_var_var_" + str(window)] = seg_emw_var.var()
        X_df.loc[seg_id, "ewm_var_mean_" + str(window)] = seg_emw_var.mean()
        X_df.loc[seg_id, "ewm_var_mad_" + str(window)] = seg_emw_var.mad()
        for q in quantile_list:
            X_df.loc[seg_id, "ewm_var_quan" + str(q) + "_" + str(window)] = seg_emw_var.quantile(q)

    return X_df

def collect_result(result):
    df_list.append(result)

def calc_train_features(train):
    print("Start date", datetime.now())

    segments = int(np.floor(train.shape[0] / rows))
    y_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
    X_train = None

    p = Pool(4)

    i = 1
    for seg_id in range(segments):
        if i< 20:
            seg = train.iloc[seg_id * rows:seg_id * rows + rows]
            y_train.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            p.apply_async(seg_rolling_features, [seg_id, seg['acoustic_data'], w_sizes], callback=collect_result)
            i += 1
        else:
            break
    print(datetime.now(), "pool closed")
    p.close()
    p.join()
    print(datetime.now(), "pool finished")
    print(df_list)
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

def calc_train_features_nomp(train):
    print("Start date", datetime.now())

    segments = int(np.floor(train.shape[0] / rows))
    y_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
    X_train = None

    i = 1
    for seg_id in range(segments):
        if i< 20:
            seg = train.iloc[seg_id * rows:seg_id * rows + rows]
            y_train.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            df = seg_rolling_features(seg_id, seg['acoustic_data'], w_sizes)
            i += 1
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

def calc_test_features():
    print("Start date", datetime.now())

    submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'), index_col='seg_id')
    # X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
    X_test = None
    p = Pool(4)
    # for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
    i = 0
    for i, seg_id in enumerate(submission.index):

        if i < 20:
            seg = pd.read_csv(test_file_path + seg_id + '.csv')['acoustic_data'].values
            x_seg = pd.Series(seg)
            print(x_seg.shape)
            # X_test = seg_rolling_features(seg_id, x_seg, [10, 100, 1000], X_test)
            p.apply_async(seg_rolling_features, args=[seg_id, x_seg, w_sizes], callback=collect_result)
            print('I am on number ' + str(i + 1) + ' of ' + str(len(submission.index)))
            i += 1
        else:
            break
    print(datetime.now(), "pool closing")
    p.close()

    p.join()

    print(datetime.now(), "pool finished")
    time.sleep(20)
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

def get_features():
    X_train = pd.read_csv(r"C:\datasets\earthquake\lanl earthquake prep data\xtrain.csv")
    #print(X_train.head())
    #X_train.drop("Unnamed: 0", inplace=True, axis=1)
    print(X_train.head())
    y_train = pd.read_csv(r"C:\datasets\earthquake\lanl earthquake prep data\ytrain.csv")
    #y_train.drop("Unnamed: 0", inplace=True, axis=1)
    print(y_train.head())

    X_test = pd.read_csv(r"C:\datasets\earthquake\lanl earthquake prep data\xtest.csv")
    # X_test.drop("Unnamed: 0", inplace=True, axis=1)
    print(X_test.head())

    for c in (X_train.columns.values.tolist()):
        mean = X_train[c].mean()
        X_train[c] = X_train[c] - mean
        X_test[c] = X_test[c] - mean
        std = X_train[c].std()
        X_train[c] = X_train[c] / std
        X_test[c] = X_test[c] / std

    return X_train, y_train, X_test

if __name__ == "__main__":

    path = r"C:\datasets\earthquake\lanl earthquake prep data"
    X_train = pd.read_csv(os.path.join(path, "xtrain.csv"))
    print(X_train.head())
    X_test = pd.read_csv(os.path.join(path, "xtest.csv")).drop("Unnamed: 0", axis=1)
    print(X_test.head())

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(method="ffill", inplace=True, axis=1)
    X_train= X_train.astype(np.float32)

    print((np.isfinite(X_train.values)))

    print(X_train.loc[:, X_train.dtypes != np.float32])

    sys.exit()

    for c in (X_train.columns.values.tolist()):
        X_train[c] = X_train[c].replace([np.inf, -np.inf], np.nan)
        X_train[c].fillna(method="ffill", inplace=True)
    print(np.any(np.isnan(X_train.values)))
    print(np.isnan(X_train.values))
    print(np.all(np.isfinite(X_train.values)))
    for c in (X_test.columns.values.tolist()):
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)
        X_test[c].fillna(X_test[c].mean(), inplace=True, axis=1)
    print(np.any(np.isnan(X_test.values)))
    print(np.all(np.isfinite(X_test.values)))


    sys.exit()
    train.columns = ["acoustic_data", "time_to_failure"]
    #calc_test_features()
    #calc_train_features(train)
    calc_train_features_nomp(train)
