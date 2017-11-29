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
episode_limit = 100

def replay_train(mainDQN,targetDQN,train_batch):
    x_stack = np.empty(0).reshape(0,mainDQN.input_size)
    y_stack = np.empty(0).reshape(0,mainDQN.output_size)


    for state,action,reward,next_state,next_player,done in train_batch:
        action = int(action)

        #print("ACTION : ", action)
        if action == -1:
            continue

        Q = mainDQN.predict(state)


        if done:
            Q[0,action] = reward

        else:
            Q[0,action] = reward + dis * np.max(targetDQN.predict(next_state))

        # np.zeros(3,3) -> np.zeros(9)
        state = env.Board

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return mainDQN.update(x_stack,y_stack)


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
                # User A
                Q = mainDQN.predict(state)
                while True:
                    reshapeQ = Q.reshape((3, 3))
                    print(reshapeQ)
                    action = (np.argmax(Q))
                    if env.__checkBoard__(action) :
                        break
                    else:
                        Q[0][np.argmax(Q)] = 0

                    print("ACTION(A) : ", action)


            else:
                # User B
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
        mainDQN_A = model.DQN(sess, input_size, output_size, name="mainA")
        targetDQN_A = model.DQN(sess, input_size, output_size, name="targetA")
        mainDQN_B = model.DQN(sess, input_size, output_size, name="mainB")
        targetDQN_B = model.DQN(sess, input_size, output_size, name="targetB")
        copy_ops_A = get_copy_var_ops(dest_scope_name="targetA", src_scope_name="mainA")
        copy_ops_B = get_copy_var_ops(dest_scope_name="targetB", src_scope_name="mainB")

        # Init Saver
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        sess.run(copy_ops_A)
        sess.run(copy_ops_B)

        for episode in range(episode_limit):
            e = 1./(1+(episode/10))
            done = False
            win_count = 0
            game_limit = 100
            reward_A = 0
            reward_B = 0
            for game_count in range(game_limit):
                state = env._reset()

                action = 0
                reward = 0
                done = False

                while not env._get_GameOver():
                    if np.random.rand(1) < e:
                        action = env._get_sample_action()
                    else:
                        if env._get_PlayerTurn():
                            # User A
                            action = env._get_location(np.argmax(mainDQN_A.predict(state)))
                            #print("ACTION(A) : ", action)
                        else :
                            # User B
                            action = env._get_location(np.argmax(mainDQN_B.predict(state)))
                            #print("ACTION(B) : ", action)

                    # Get new state and reward from environment
                    cur_state = np.copy(state)
                    next_state, next_player, reward, done = env._step(int(action))


                    if done :
                        if env._is_draw():
                            reward_A += -0
                            reward_B += -0
                        elif env._get_A() == reward:
                            reward_A += 5
                            reward_B -= 5
                        elif env._get_B() == reward:
                            reward_B += 5
                            reward_A -= 5


                    replay_buffer.append((cur_state,action,reward_A,reward_B,next_state,next_player,done))

                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                    state = next_state


            print("Episode = {} , A {} : B {}".format(episode,reward_A,reward_B))

            if episode % 10 == 1:
                for _ in range(50):
                    # mini batch works better!
                    lossA = 0
                    lossB = 0
                    minibatch = random.sample(replay_buffer, 10)
                    lossA,lossB= replay_train(mainDQN_A,mainDQN_B,targetDQN_A,targetDQN_B, minibatch)

                print("Loss A: ", lossA, " / Loss B: ", lossB)

        # 최적화가 끝난 뒤, 변수를 저장합니다.
        saver.save(sess, './model/dqn.ckpt')



if __name__ == "__main__" :
    #main()
    play_with_bot_A()

