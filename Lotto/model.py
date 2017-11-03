import tensorflow as tf
import numpy as np

dis = .9
class CNN:
    def __init__(self,session,input_size,output_size,name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._filter_size = 3
        self._keep_prob = tf.placeholder(tf.float32)

        self._build_network()


    def _build_network(self,h_size=10,l_rate=1e-1):
        #with tf.variable_scope(self.net_name):
        self._X = tf.placeholder(tf.float32,[None,7,7,1],name="input_x")
        # First Layer Of Weight
        W1 = tf.get_variable("W1",shape=[3,3,1,32],
                             initializer=tf.contrib.layers.xavier_initializer())
        print(W1)

        layer1 = tf.nn.conv2d(self._X,W1,strides=[1,1,1,1],padding='SAME')
        layer1 = tf.nn.relu(layer1)
        print(layer1)

        layer1 = tf.nn.max_pool(layer1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
        print(layer1)

        layer1 = tf.nn.dropout(layer1,keep_prob=self._keep_prob)



        # Second Layer Of Weight
        W2 = tf.get_variable("W2",shape=[3,3,32,64],
                             initializer=tf.contrib.layers.xavier_initializer())
        print(W2)

        layer2 = tf.nn.conv2d(layer1,W2,strides=[1,1,1,1],padding='SAME')
        layer2 = tf.nn.relu(layer2)
        print(layer2)

        layer2 = tf.nn.max_pool(layer2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
        print(layer2)

        layer2 = tf.nn.dropout(layer2,keep_prob=self._keep_prob)



        W3 = tf.get_variable("W3",shape=[7*7*64,512],
                             initializer=tf.contrib.layers.xavier_initializer())
        print(W3)

        layer3 = tf.reshape(layer2,[-1,7*7*64])
        layer3 = tf.matmul(layer3,W3)
        layer3 = tf.nn.relu(layer3)
        print(layer3)

        layer3 = tf.nn.dropout(layer3,keep_prob=self._keep_prob)



        W4 = tf.get_variable("W4",shape=[512,45],
                             initializer=tf.contrib.layers.xavier_initializer())
        self._l_pred = tf.matmul(layer3,W4)
        print(W4)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(tf.float32,[None,self.output_size],name="output_y")
        # Loss Function
        #self._loss = tf.reduce_mean(tf.square(self._Y - self._l_pred))
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._l_pred,labels=self._Y))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self,x):
        #x = np.reshape(state,[None,7,7,1])
        return self.session.run(self._l_pred,feed_dict={self._X:x,self._keep_prob:0.7})

    def update(self,x,y):
        return self.session.run([self._loss,self._train],feed_dict={self._X:x,self._Y:y,self._keep_prob:0.7})


class RNN:
    def __init__(self, session, n_input,n_step, n_output, name="main"):
        self.session = session
        self._n_input = n_input
        self._n_output = n_output
        self._n_step = n_step
        self.net_name = name
        self._n_hidden = 256
        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        # with tf.variable_scope(self.net_name):
        self._X = tf.placeholder(tf.float32, [None, self._n_step, self._n_input], name="input_x")
        self._Y = tf.placeholder(tf.float32, [None, self._n_output],name="output_y")
        # First Layer Of Weight
        W = tf.get_variable("W", shape=[self._n_hidden,self._n_output],
                             initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable("b", shape=[self._n_output],
                            initializer=tf.contrib.layers.xavier_initializer())

        # init RNN cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden)

        # Simpled RNN using tensorflow's nn.dynamic_rnn()
        outputs, states = tf.nn.dynamic_rnn(cell,self._X,dtype=tf.float32)

        outputs = tf.transpose(outputs,[1,0,2])
        outputs = outputs[-1]
        self._model = tf.matmul(outputs,W) + b
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._model,labels=self._Y))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._cost)


    def predict(self, x):
        # x = np.reshape(state,[None,7,7,1])
        return self.session.run(self._model, feed_dict={self._X: x})

    def update(self, x, y):
        return self.session.run([self._cost, self._train], feed_dict={self._X: x, self._Y: y})



