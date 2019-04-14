from sklearn import preprocessing

labels = ['satosa', 'versicolor', 'virginica']

# X => feature, y => labels

encoder = preprocessing.LabelEncoder()
encoder.fit(labels)

for i, item in enumerate(encoder.classes_):
    print(item, "=>", i)