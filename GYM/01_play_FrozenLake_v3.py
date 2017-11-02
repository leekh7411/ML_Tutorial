import gym
from gym.envs.registration import register
import sys
import termios
import tty
#----------------------------------------------------------------------------------------------------------------------#
class  _Getch:
    def __call__ (self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcgetattr(fd, termios.TCSADRAIN,old_settings)
        return ch

inkey = _Getch()
#----------------------------------------------------------------------------------------------------------------------#
# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
#----------------------------------------------------------------------------------------------------------------------#
# KEY MAPPING
arraw_keys={
    'U':UP,
    'D':DOWN,
    'R':RIGHT,
    'L':LEFT
}
#----------------------------------------------------------------------------------------------------------------------#
# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
)

env = gym.make('FrozenLake-v3')
env.render() # Show initial board

while True:
    # Choose an action from keyboard
    key = input()

    if key not in arraw_keys.keys():
        print("Game aborted!")
        break

    action = arraw_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ",state,"/Action: ",action,"/Reward: ",reward,"/Info: ",info)

    if done:
        print("Finished with reward: ",reward)
        break



#----------------------------------------------------------------------------------------------------------------------#
