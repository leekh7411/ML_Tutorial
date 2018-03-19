import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics,model_selection,svm
from sklearn.grid_search import GridSearchCV
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

# Setting parameters for Greed-Search
params = [
    {"C":[1,10,100,1000],"kernel":["linear"]},
    {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]}
]

# Separate data set
train_data,test_data,train_label,test_label = train_test_split(data,label)

# Train data
clf = GridSearchCV(svm.SVC,params,n_jobs=-1)
clf.fit(train_data,train_label)

# predict
pred = clf.predict(test_data)

# test
ac_score = metrics.accuracy_score(pred,test_label)
print("accuracy score:",ac_score)