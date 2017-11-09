import Five_stones.game as game
import Five_stones.model as model
import tensorflow as tf
import numpy as np
import random
from collections import deque

env = game.TicTacToe()
input_size  = env._get_observation_space()
output_size = env._get_action_spcae()
dis = 0.99
learning_rate = .07
REPLAY_MEMORY = 5000
episode_limit = 5000


def replay_train(mainDQN_A,mainDQN_B,targetDQN_A,targetDQN_B,train_batch):
    ax_stack = np.empty(0).reshape(0,mainDQN_A.input_size)
    ay_stack = np.empty(0).reshape(0,mainDQN_A.output_size)
    bx_stack = np.empty(0).reshape(0,mainDQN_B.input_size)
    by_stack = np.empty(0).reshape(0,mainDQN_B.output_size)

    for state,action,reward_A,reward_B,next_state,next_player,done in train_batch:
        action = int(action)
        print(state)
        print(next_state)
        if not next_player:
            QA = mainDQN_A.predict(state)
        else :
            QB = mainDQN_B.predict(state)

        if done:
            if not next_player:
                QA[0,action] = reward_A
            else:
                QB[0,action] = reward_B
        else:
            if not next_player:
                QA[0,action] = reward_A + dis * np.max(targetDQN_A.predict(next_state))
            else:
                QB[0,action] = reward_B + dis * np.max(targetDQN_B.predict(next_state))

        if not next_player:
            ax_stack = np.vstack([ax_stack, state])
            ay_stack = np.vstack([ay_stack, QA])
        else:
            bx_stack = np.vstack([bx_stack, state])
            by_stack = np.vstack([by_stack, QB])

    return mainDQN_A.update(ax_stack,ay_stack),mainDQN_B.update(bx_stack,by_stack)


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


def main():
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN_A = model.DQN(sess,input_size,output_size,name="mainA")
        targetDQN_A = model.DQN(sess,input_size,output_size,name="targetA")
        mainDQN_B = model.DQN(sess,input_size,output_size,name="mainB")
        targetDQN_B = model.DQN(sess,input_size,output_size,name="targetB")
        tf.global_variables_initializer().run()
        copy_ops_A = get_copy_var_ops(dest_scope_name="targetA",src_scope_name="mainA")
        copy_ops_B = get_copy_var_ops(dest_scope_name="targetB",src_scope_name="mainB")
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
                        else :
                            # User B
                            action = env._get_location(np.argmax(mainDQN_B.predict(state)))

                    # Get new state and reward from environment
                    cur_state = np.copy(state)
                    next_state, next_player, reward, done = env._step(int(action))


                    if done :
                        if env._is_draw():
                            reward_A += -0
                            reward_B += -0
                        elif env._get_A() == reward:
                            reward_A += 1
                            reward_B -= 0
                        elif env._get_B() == reward:
                            reward_B += 1
                            reward_A -= 0


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



if __name__ == "__main__" :
    main()

