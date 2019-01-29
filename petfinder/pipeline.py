import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2 as sk_chi, SelectKBest
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import entropy
import numpy as np
from collections import Counter
import math
from petfinder.get_explore import read_data
from petfinder.preprocessing import prepare_data
from petfinder.feature_engineering import finalize_data, add_features

if __name__ == "__main__":
    train, test = read_data()
    train, test = prepare_data(train, test)