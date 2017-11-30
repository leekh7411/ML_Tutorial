import Five_stones.game as game
import Five_stones.model as model
import tensorflow as tf
import numpy as np
import random
from collections import deque

env = game.TicTacToeSingle()
input_size  = env.Board.__len__()
output_size = env.Board.__len__()
dis = 0.9
learning_rate = .09
REPLAY_MEMORY = 5000
episode_limit = 1000

def get_copy_var_ops(*,dest_scope_name="target",src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name
    )
    dst_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name
    )
    for src_var,dst_var in zip(src_vars,dst_vars):
        op_holder.append(dst_var.assign(src_var.value()))
    return op_holder


def replay_train(mainDQN,targetDQN,train_batch):
    x_stack = np.empty(0).reshape(0,mainDQN.input_size)
    y_stack = np.empty(0).reshape(0,mainDQN.output_size)


    for state,action,reward,next_state,cur_player,done in train_batch:
        action = int(action)
        if action == -1:
            continue
        Q = mainDQN.predict(state)

        if done:
            Q[0,action-1] = reward
        else:
            Q[0,action-1] = reward + dis * np.max(targetDQN.predict(next_state))

        state = env.Board
        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return mainDQN.update(x_stack,y_stack)


def play_with_bot():
    with tf.Session() as sess:
        # Init Model
        mainDQN = model.DQN(sess, input_size, output_size, name="main")
        targetDQN = model.DQN(sess, input_size, output_size, name="target")

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        # Init Saver
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("There is not saved models")
            return -1

        state = env.__reset__()

        while not env.GameOver:
            if env.Turn == env.PlayerA:
                # Bot
                Q = mainDQN.predict(state)
                while True:
                    reshapeQ = Q.reshape((3, 3))
                    print(reshapeQ)
                    action = (np.argmax(Q))
                    action += 1 # index revision
                    print("ACTION(A) : ", action)
                    if env.__checkBoard__(action) :
                        break
                    else:
                        Q[0][np.argmax(Q)] = 0




            else:
                # Human
                action = input("Enter location 1~9: ")

            env.__step__(int(action))


        if env.Winner == env.Draw:
            print("Draw Game")
        elif env.Winner == env.PlayerA:
            print("Player A win!")
        elif not env.Winner == env.PlayerB:
            print("Player B win!")


def main():
    replay_buffer = deque()

    with tf.Session() as sess:
        # Init Model
        mainDQN = model.DQN(sess, input_size, output_size, name="main")
        targetDQN = model.DQN(sess, input_size, output_size, name="target")

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        # Init Saver
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        sess.run(copy_ops)

        for episode in range(episode_limit):
            e = 1./(1+(episode/10))
            done = False
            win_count = 0
            game_limit = 200
            reward = 0

            for game_count in range(game_limit):
                state = env.__reset__()

                while not env.GameOver:
                    if np.random.rand(1) < e:
                        action = random.randint(1,9)
                    else:
                        action = (np.argmax(mainDQN.predict(state)))

                    # Get new state and reward from environment
                    cur_state = np.copy(state)
                    #env.__printParmBoard__(cur_state,"Current Board state")

                    env.__step__(int(action))
                    next_state = np.copy(env.Board)
                    next_state = env.__getReverse__(next_state)
                    #env.__printParmBoard__(next_state,"Next Board state")

                    cur_player = env.Turn
                    done = env.GameOver

                    if env.GameOver :
                        if env.Winner == env.Turn:
                            reward += 2
                        else:
                            reward -= 4


                    replay_buffer.append((cur_state,action,reward,next_state,cur_player,done))

                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                    state = next_state


            print("Episode = {} , Reward {}".format(episode,reward))

            if episode % 10 == 1:
                for _ in range(50):
                    # mini batch works better!
                    minibatch = random.sample(replay_buffer, 10)
                    loss = replay_train(mainDQN,targetDQN, minibatch)

                print("Loss: ", loss)

        # 최적화가 끝난 뒤, 변수를 저장합니다.
        saver.save(sess, './model/dqn.ckpt')



if __name__ == "__main__" :
    main()
    #play_with_bot()

