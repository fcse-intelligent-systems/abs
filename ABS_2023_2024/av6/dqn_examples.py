import gymnasium as gym
from deep_q_learning import DQN
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error

def build_model(state_space_shape, num_actions):
    model = Sequential()
    model.add(Dense(16, input_shape=state_space_shape))
    model.add(Dense(16))
    model.add(Dense(num_actions, activation='lineart'))

    model.compile(SGD(0.001), mean_squared_error)

    return model



if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset()
    env.render()

    state_space_shape = env.observation_space.shape
    num_actions = env.action_space.n
    num_episodes = 100
    num_steps_per_episode = 5

    model = build_model(state_space_shape, num_actions)
    target_model = build_model(state_space_shape, num_actions)

    agent = DQN(state_space_shape, num_actions, model, target_model)

    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(num_steps_per_episode):
            action = agent.get_action(state, 0)
            new_state, reward, terminated, _, _ = env.step(action)
            agent.update_memory(state, action, reward, new_state, terminated)

        agent.train()

        if episode % 10 == 0:
            agent.update_target_model()
    print()