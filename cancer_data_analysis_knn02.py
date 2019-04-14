import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot


raw_data = pd.read_csv("cancer.csv", delimiter=',')
print(raw_data.tail(10))