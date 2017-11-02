import tensorflow as tf
import matplotlib.pyplot as plt

X = [1.0,2.0,3.0]
Y = [1.0,2.0,3.0]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X,W)

# simplified cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

# before starting, initialize the variables. We will run this first
init = tf.initialize_all_variables()

# For Graph
W_val = []
cost_val = []

# launch the graph
sess = tf.Session()
sess.run(init)

for i in range(-30,50):
    print(i*0.1, sess.run(cost,feed_dict={W: i * 0.1}))
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost,feed_dict={W: i * 0.1}))

#Graph Display
plt.plot(W_val,cost_val,'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()

