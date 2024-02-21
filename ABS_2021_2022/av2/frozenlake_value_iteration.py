import gym
from mdp import value_iteration
import numpy as np
import time
import os

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()

    policy, value = value_iteration(env, discount_factor=0.9)
    print(policy)
    print(value)

    state = env.reset()
    env.render()
    done = False
    while not done:
        new_action = np.argmax(policy[state])
        state, reward, done, _ = env.step(new_action)
        print(reward)
        env.render()
        time.sleep(0.5)
        os.system('cls')
