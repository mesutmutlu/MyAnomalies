import numpy as np
import pandas as pd
import os
from obspy.signal.trigger import z_detect, classic_sta_lta
import sys
from sklearn.model_selection import KFold

base_path = r"C:\datasets\earthquake"



train = pd.read_csv(os.path.join(base_path, "train/train.csv.010"), header=None)
train.columns= ["acoustic_data", "time_to_failure"]

splitter = KFold(n_splits=5, random_state=2652124)

splits = splitter.split(train["acoustic_data"], train["time_to_failure"])

for (x, y) in splits:
        print(x)
        print(y)

print(splits)

sys.exit()

#train.set_index("acoustic_data")
x = pd.DataFrame(columns=["t"], data=[np.inf, -np.inf, np.NaN])

print(x["t"].nunique())

#print(train.index.tolist())

sys.exit()
X_train = None



segments = int(np.floor(train.shape[0] / rows))


for sta_lta in [(500, 10000), (500, 20000), (3333, 6666), (3333, 9999), (5000, 50000), (5000, 100000), (10000, 25000), (10000, 100000)]:
        X_tr.loc[seg_id, 'classic_sta_'+str(sta_lta[0])+'_lta_'+str(sta_lta[1])+'_mean'] = classic_sta_lta(seg, sta_lta[0], sta_lta[1]).mean()
        X_tr.loc[seg_id, 'classic_sta_'+str(sta_lta[0])+'_lta_'+str(sta_lta[1])+'_std'] = classic_sta_lta(seg, sta_lta[0], sta_lta[1]).std()
