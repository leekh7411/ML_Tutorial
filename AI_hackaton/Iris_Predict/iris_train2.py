import pandas as pd
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split

# Read Data
csv = pd.read_csv('iris.csv')

# Get column data
csv_data = csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
csv_label = csv["Name"]

# Divide training data and testing data
train_data,test_data,train_label,test_label = train_test_split(csv_data,csv_label)

# train and predict
clf = svm.SVC()
clf.fit(train_data,train_label)
pre = clf.predict(test_data)

# Accuracy score
ac_score = metrics.accuracy_score(test_label,pre)
print("AC SCORE :",ac_score)