import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

# set Varialble to TensorFlow's Variable
W = tf.Variable(tf.random_uniform([1],-1.0,1.0)) # -1.0 ~ 1.0 Random Value
W1 = tf.Variable(tf.random_uniform([1],-1.0,1.0)) # -1.0 ~ 1.0 Random Value

b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

# Using PlaceHolder --> Benefit  : Recylclable
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


hypothesis = W * X + b

# simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize the variables. We will run this first
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}), sess.run(W), (sess.run(b)))

print(sess.run(hypothesis,feed_dict={X:5}))
print(sess.run(hypothesis,feed_dict={X:2.5}))