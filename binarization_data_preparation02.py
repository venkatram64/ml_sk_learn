from sklearn import preprocessing
import numpy as np

data = np.array([[2.2, 5.9, -1.0], [5.4,-3.2, -5.1], [-1.9, 4.2, 3.2]])

minmax_scalar = preprocessing.MinMaxScaler(feature_range=(0,1))
data_minmax = minmax_scalar.fit_transform(data)
print("MinMaxScalar applied on the data \n\n", data_minmax)
