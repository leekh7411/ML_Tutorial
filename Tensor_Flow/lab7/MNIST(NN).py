import tensorflow as tf
import Tensor_Flow.lab7.input_data as input_data


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else :
        stddev = tf.sqrt(3.0/(n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1

x = tf.placeholder("float",[None, 784])
y = tf.placeholder("float",[None, 10])
W1 = tf.get_variable("W1",shape=[784,500], initializer=xaver_init(784,500))
W2 = tf.get_variable("W2",shape=[500,256], initializer=xaver_init(500,256))
W3 = tf.get_variable("W3",shape=[256,10], initializer=xaver_init(256,10))

b1 = tf.Variable(tf.zeros([500]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(x,W1),b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))
activation = tf.add(tf.matmul(L2,W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
checkpoint_dir = "cps/"

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ('load learning')
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss

        # Display logs per epoch step

        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        if epoch % display_step == 0: # Softmax
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (sess.run(b3))

    print ("Optimization Finished!")

    # Test network
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    saver.save(sess, checkpoint_dir + 'network.ckpt')



