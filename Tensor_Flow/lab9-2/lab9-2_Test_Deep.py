import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True,dtype='float32')

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1],(4,1))


print('x', x_data)
print('y', y_data)

X = tf.placeholder(tf.float32, name="X-Input")
Y = tf.placeholder(tf.float32, name="Y-Input")

#W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))
#h = tf.matmul(W,X)
#hypothesis = tf.div(1.,1.+tf.exp(-h))

# Neural Net
"""
        [This is 'Wide']
            .
            ..
            ...
            {W}
X --------->{W} ---> {W2} ---> {W3} ---> {W4} .... [This is 'Deep']
            {W}
            ...
            ..
            .

"""
W1 = tf.Variable(tf.random_uniform([2,5],-1.0,1.0), name="Weight1") # first '2' value is fixed, input data's degree is '2'
W2 = tf.Variable(tf.random_uniform([5,4],-1.0,1.0), name="Weight2") # first layer? has five weight
W3 = tf.Variable(tf.random_uniform([4,1],-1.0,1.0), name="Weight3") # seconde layer? has fout weight

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

#Our Hypothesis
with tf.name_scope("Layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X,W1) + b1)
with tf.name_scope("Layer3") as scope:
    L3 = tf.sigmoid(tf.matmul(L2,W2) + b2)
with tf.name_scope("Layer4") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3,W3) + b3)

with tf.name_scope("Cost") as scope:
    cost = tf.reduce_mean(-Y*tf.log(hypothesis) - (1-Y)*tf.log(1-hypothesis))
    cost_summ = tf.scalar_summary("cost",cost)

with tf.name_scope("Train") as scope:
    a = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

W1_hist = tf.histogram_summary("weights-1",W1)
W2_hist = tf.histogram_summary("weights-2",W2)
W3_hist = tf.histogram_summary("weights-3",W3)

B1_hist = tf.histogram_summary("biases-1",b1)
B2_hist = tf.histogram_summary("biases-2",b2)
B3_hist = tf.histogram_summary("biases-3",b3)

Y_hist = tf.histogram_summary("y",Y)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph_def)


    sess.run(init)

    for step in range(18000):
        # This Case tensorboard record data every step
        # summary , _ = sess.run([merged,train],feed_dict={X:x_data,Y:y_data})
        # writer.add_summary(summary,step)

        sess.run(train,feed_dict={X:x_data,Y:y_data})

        if step % 2000 == 0:
            summary = sess.run(merged,feed_dict={X:x_data, Y:y_data})
            writer.add_summary(summary,step)
            print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}))

    #Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5),Y)

    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5),correct_prediction, accuracy],feed_dict={X:x_data,Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data})*100)

