''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# init data
import random

# function for make label
def cal_bmi(h,w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5 : return "thin"
    if bmi < 25 : return "normal"
    return "fat"

# output file init
fp = open("bmi.csv","w",encoding="utf-8")
fp.write("height,weight,label\r\n")

# random data create
cnt = {"thin":0,"normal":0,"fat":0}
for i in range(20000):
    h = random.randint(140,200)
    w = random.randint(45,100)
    label = cal_bmi(h,w)
    cnt[label] += 1
    fp.write("{0},{1},{2}\r\n".format(h,w,label))
fp.close()
print("train data init finish!")
print(cnt)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# start train
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv("bmi.csv")
label = tbl["label"]
h = tbl["height"]
w = tbl["weight"]
wh = pd.concat([w,h],axis=1) # axis 1 means row-wise concatenate

# separate data
train_data,test_data,train_label,test_label = train_test_split(wh,label)

# Let's training
clf = svm.SVC()
clf.fit(train_data,train_label)

# predict
pred = clf.predict(test_data)

# test
ac_score = metrics.accuracy_score(pred,test_label)
cl_report = metrics.classification_report(pred,test_label)

print("accuracy score:",ac_score)
print("classification report\n",cl_report)


