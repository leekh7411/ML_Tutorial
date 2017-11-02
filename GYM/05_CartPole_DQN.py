import tensorflow as tf
import numpy as np
import random
import gym
from GYM.dqn import DQN
from collections import deque
from gym.envs.registration import register
# Environment Init
register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    reward_threshold=10000.0,
)
env = gym.make('CartPole-v2')
env.reset()

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
dis = 0.99
learning_rate = .07
REPLAY_MEMORY = 5000
episode_limit = 5000

def bot_play(mainDQN):
    # See our trained network in action
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total Score : {}".format(reward_sum))
            break


def simple_replay_train(DQN,train_batch):
    x_stack = np.empty(0).reshape(0,DQN.input_size)
    y_stack = np.empty(0).reshape(0,DQN.output_size)

    # Get Stored Information From The Buffer
    for state, action,reward,next_state, done in train_batch:
        Q = DQN.predict(state)
        # Check Terminal
        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + dis * np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])

    return DQN.update(x_stack,y_stack)

def main():
    max_episode = 5000
    # store the observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess,input_size,output_size)
        tf.global_variables_initializer().run()

        for episode in range(episode_limit):
            e = 1./((episode/10)+1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                env.render()
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done,_ = env.step(action)

                if done :
                    reward = -100 # big penalty

                # Save the experience to our buffer
                replay_buffer.append((state,action,reward,next_state,done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

                if step_count > 10000 :
                    break # Good Enough

            print("Episode = {} steps = {}".format(episode,step_count))
            if step_count > 10000:
                pass
                # break?

            if episode % 10 == 1:# train every 10 iteration
                # Get random batch of experience
                for _ in range(50):
                    # mini batch works better!
                    minibatch = random.sample(replay_buffer,10)
                    loss,_ = simple_replay_train(mainDQN,minibatch)
                print("Loss : ", loss)

        bot_play(mainDQN)

if __name__ == "__main__":
    main()

