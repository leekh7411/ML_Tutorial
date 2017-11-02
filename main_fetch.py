import tensorflow as tf

input1 = tf.constant(11.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2,input3)
mul = tf.mul(intermed,input1)
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("test2", sess.graph)
    result = sess.run([mul,intermed])
    print(result)
    writer.close()