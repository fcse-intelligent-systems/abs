import numpy as np
import gymnasium as gym
from mdp import value_iteration, policy_iteration

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='human')

    state, _ = env.reset()

    policy, _ = value_iteration(env,
                                env.action_space.n,
                                env.observation_space.n)

    terminated = False

    while not terminated:
        action = np.argmax(policy[state])
        state, reward, terminated, _, _ = env.step(action)
        env.render()

    print()
