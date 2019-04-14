from sklearn import preprocessing
import numpy as np

data = np.array([[2.2, 5.9, -1.0], [5.4,-3.2, -5.1], [-1.9, 4.2, 3.2]])

bindata = preprocessing.Binarizer(threshold=1.5).transform(data)

print("Binaized data \n\n", bindata)

print("Mean (before)=", data.mean(axis=0))
print("Standard Deviation (before)=", data.std(axis=0))

scaled_data = preprocessing.scale(data)

print("Mean (After)=", scaled_data.mean(axis=0))
print("Standard Deviation (After)=", scaled_data.std(axis=0))
