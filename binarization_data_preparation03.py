from sklearn import preprocessing
import numpy as np
#normalization
data = np.array([[2.2, 5.9, -1.0], [5.4,-3.2, -5.1], [-1.9, 4.2, 3.2]])

data_l1 = preprocessing.normalize(data, norm='l1')
data_l2 = preprocessing.normalize(data, norm='l2')

print("L1 normalization data \n\n", data_l1)
print("L2 normalization data \n\n", data_l2)
