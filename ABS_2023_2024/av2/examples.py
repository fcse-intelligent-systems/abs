import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='human')

    env.reset()

    env.render()

    #for i in range(10):
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        state, reward, terminated, _, _ = env.step(action)
        env.render()

    print()