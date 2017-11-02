import tensorflow as tf
sess = tf.Session()
hello = tf.constant('Hello, TensorFlow!')
print(sess.run(hello))
print(str(sess.run(hello),encoding="utf-8"))


a = tf.constant(1)
print(a)
with tf.Session() as sess:
    print(a.eval())

x = tf.constant(35,name='x')
y = tf.Variable(x+5,name='y')
model = tf.initialize_all_variables() # what...the...f..
with tf.Session() as sess:
    sess.run(model) # Global variables init start actually here
    print(sess.run(y)) # start y variable

x2 = tf.lin_space(1.0,-1.0,10)
print(x2)
g = tf.get_default_graph()
print([op.name for op in g.get_operations()])
sess = tf.Session()
print(sess.run(x2))
sess.close()