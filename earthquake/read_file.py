import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import math
import psutil
import os
from datetime import datetime
base_path = r"C:\datasets\earthquake"
train_path = os.path.join(base_path, "train/train.csv")
test_file_path = os.path.join(base_path, "test")


train = pd.read_csv(train_path)
sys.exit()

rows = 150_000

X_train = pd.DataFrame(dtype=np.float64)
y_train = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])

i = 1

pd_seg = pd.DataFrame(dtype=np.float64, columns=['acoustic_data', 'time_to_failure'])
seg_id = 1
n_line = 1
with open(train_path) as infile:
    for line in infile:
        if n_line % 150000 == 1:
            print(n_line, n_line % 150000, datetime.now())
            seg_id += 1
            pd_seg = None
            pd_seg = pd.DataFrame(dtype=np.float64, columns=['acoustic_data', 'time_to_failure'])

        accoustic_data = line.split(",")[0]
        time_to_failure = line.split(",")[1]
        pd_seg.append({'acoustic_data' : accoustic_data , 'time_to_failure' : time_to_failure}, ignore_index=True)

        n_line +=1
#train = pd.read_csv(train_path, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
# pandas doesn't show us all the decimals
print(train.shape)