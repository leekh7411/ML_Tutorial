import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics,model_selection,svm
from sklearn.model_selection import train_test_split
from collections import defaultdict
# read mushroom.csv data using pandas
mr = pd.read_csv("mushroom.csv",header=None)

# Change data's alphabet to integer
label = []
data = []
attr_list = defaultdict(lambda : 0)

# If csv not include header then use like this
for row_idx, row in mr.iterrows():
    label.append(row.ix[0])
    row_data = []
    for v in row.ix[1:]:
        attr_list[v] += 1
        row_data.append(ord(v))
    data.append(row_data)

# Cross Validation
print("Cross-Validation!")
clf = RandomForestClassifier()
ac_scores = model_selection.cross_val_score(clf,data,label,cv=3)
print("each accuracy score\n",ac_scores)
print("mean of scores:",ac_scores.mean())

# Separate data set
train_data,test_data,train_label,test_label = train_test_split(data,label)

# Train data
clf = RandomForestClassifier()
clf.fit(train_data,train_label)

# predict
pred = clf.predict(test_data)

# test
ac_score = metrics.accuracy_score(pred,test_label)
print("accuracy score:",ac_score)