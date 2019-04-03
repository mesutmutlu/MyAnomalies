import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import math
import psutil
import os
from datetime import datetime
base_path = r"C:\datasets\earthquake"
import multiprocessing
train_path = os.path.join(base_path, "train/train.csv")
test_file_path = os.path.join(base_path, "test")


def seg_rolling_features(seg, seg_id , w_sizes):
    print("start", seg_id, datetime.now())
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
        seg_roll_min = seg.rolling(window).min()
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
    print("end", seg_id, datetime.now())
    return X_df



if __name__ == "__main__":

    train = pd.read_csv(os.path.join(base_path, "train/train.csv.010"), header=None)

    train.columns = ["acoustic_data", "time_to_failure"]

    rows = 100_000
    w_sizes = [10, 100, 1000]

    segments = int(np.floor(train.shape[0] / rows))
    print(segments)
    df_list= []


    def collect_result(result):
        df_list.append(result)


    p = multiprocessing.Pool(2)
    for seg_id in (range(segments)):
        if seg_id < 15:
            print(seg_id)
            seg = train.iloc[seg_id * rows:seg_id * rows + rows]

            p.apply_async(seg_rolling_features, [seg['acoustic_data'], seg_id, w_sizes], callback=collect_result)
    p.close()
    p.join()
    X_train = None
    #print(df_list)
    for x in df_list:
        #print(type(x))
        if X_train is None:
            X_train = x
        else:
            X_train = pd.concat([X_train,x])
    print(X_train)
    sys.exit()
