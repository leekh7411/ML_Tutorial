import tensorflow as tf
import gym
import numpy as np
env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
episode_limit = 500
hidden_layer1_size = 1256
# Q-Network
# input and output size based on Environment(CartPole)
# input  ->  4 -> [a, b, c, d]
# output ->  2 -> 1 or 0
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = .1
# Qpred = X*W -> Predicted Q value
X = tf.placeholder(shape=[None,input_size],dtype=tf.float32) # state input
W = tf.Variable(tf.random_uniform([input_size,hidden_layer1_size],0,0.01)) # weight and init uniform random
W1 = tf.get_variable("W1",shape=[hidden_layer1_size,output_size],
                     initializer= tf.contrib.layers.xavier_initializer())
Qpred = tf.matmul(tf.matmul(X,W),W1)
# Output -> Y
Y = tf.placeholder(shape=[None,output_size],dtype=tf.float32)# None = 1 or otherwise
# Build tensorflow error value calculate structure -> loss
loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.AdamOptimizer  (learning_rate=learning_rate).minimize(loss)
# set Q-Learning related parameters
dis = .9
# Create Lists to Contain Total Rewards And Steps Per Episode
rList = []
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(episode_limit):
        s = env.reset()
        eps = 1./((i/10)+1)
        rAll = 0
        done = False
        local_loss = []
        step_count = 0
        # Training Q-Net
        while not done:
            env.render()
            step_count += 1
            # Reshape input state [ 1 X [input size] ] using numpy
            x = np.reshape(s,[1,input_size])
            # Get Q States when we go to LEFT or RIGHT cases
            Qs = sess.run(Qpred,feed_dict={X:x})
            # And Choice Random Action or Predicted Action by Q States
            if np.random.rand(1) < eps:
                # This is Random Action
                a = env.action_space.sample()
            else:
                # This is Predicted Action by Q States
                a = np.argmax(Qs)
            # Get Next State, Reward, Done flag by Env given action
            s1, reward,done,_ = env.step(a)

            if done:
                Qs[0,a] = -100 # Pole down -> Game Over
            else:
                # s1 is Next State(an output of env.step(a))
                # Reshape s1 -> [1x[input_size]] -> x1
                x1 = np.reshape(s1,[1,input_size])
                Qs1 = sess.run(Qpred,feed_dict={X:x1})
                Qs[0,a] = reward + dis * np.max(Qs1)

            sess.run(train,feed_dict={X:x,Y:Qs})
            s = s1

        rList.append(step_count)
        print("Episode : {} Steps : {}".format(i,step_count))

        # Check learning finish
        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break


    # Let's See Our Trained Q-Network
    observation = env.reset()
    reward_sum = 0
    while True:
        env.render()
        x = np.reshape(observation,[1,input_size])
        Qs = sess.run(Qpred,feed_dict={X:x})
        a = np.argmax(Qs)

        observation,reward,done,_ = env.step(a)
        reward_sum += reward

        if done:
            print("Total Score : {}".format(reward_sum))
            break






