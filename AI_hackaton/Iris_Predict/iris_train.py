from sklearn import svm, metrics
import random, re

csv = []

with open('iris.csv','r',encoding='utf-8') as fp:
    for line in fp:
        line = line.strip() # line change remove
        cols = line.split(',')# delim = ','
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$',n) else n
        cols = list(map(fn,cols))
        csv.append(cols)

# remove file header
del csv[0]

# data shuffle
random.shuffle(csv)

# separate train set and test set
total_len = len(csv)
train_len = int(total_len*2/3)
train_data = []
train_label = []
test_data = []
test_label = []
for i in range(total_len):
    data = csv[i][0:4] # 0 ~ 3
    label = csv[i][4]  # 4
    if i < train_len :
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

# train and predict
clf = svm.SVC()
clf.fit(train_data,train_label)
pre = clf.predict(test_data)

# Accuracy score
ac_score = metrics.accuracy_score(test_label,pre)
print("AC SCORE:",ac_score)

