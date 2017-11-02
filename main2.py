import tensorflow as tf

a = tf.add(1,2,)
b = tf.mul(a,3)
c = tf.add(4,5,)
d = tf.mul(c,6,)
e = tf.mul(4,5,)
f = tf.div(c,6,)
g = tf.add(b,d)
h = tf.mul(g,f)
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("test1",sess.graph)
    print(sess.run(h))
    writer.close()
