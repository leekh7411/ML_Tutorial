import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True,dtype='float32')

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1],(4,1))


print('x', x_data)
print('y', y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))
#h = tf.matmul(W,X)
#hypothesis = tf.div(1.,1.+tf.exp(-h))

# Neural Net
W1 = tf.Variable(tf.random_uniform([2,2],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0))

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

#Our Hypothesis
L2 = tf.sigmoid(tf.matmul(X,W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2,W2) + b2)

cost = tf.reduce_mean(-Y*tf.log(hypothesis) - (1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in range(18000):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        if step % 20 == 0:
            print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}))

    #Test network
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5),Y)

    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5),correct_prediction, accuracy],feed_dict={X:x_data,Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data})*100)