from sklearn import svm,metrics

if __name__ == "__main__":
    # XOR Result data
    xor_data = [
        #P Q R
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,0]
    ]

    # init data set and label
    data = []
    label = []

    for row in xor_data:
        p = row[0]
        q = row[1]
        r = row[2]
        data.append([p,q])
        label.append(r)

    # Train data
    clf = svm.SVC()
    clf.fit(data,label)

    # Predict
    pre = clf.predict(data)
    print(" predict : ", pre)

    # Calculate percentage of correct answers
    ac_score = metrics.accuracy_score(label,pre)
    print(" per of correct : ", ac_score)


