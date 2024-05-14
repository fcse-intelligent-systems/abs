import gymnasium as gym
import numpy as np
import time
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from deep_q_learning import DQN


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4, padding='same', activation='relu', input_shape=state_space_shape))
    model.add(Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(Adam(lr=learning_rate), loss=MeanSquaredError())
    return model


def preprocess_state(state):
    img = Image.fromarray(state)
    img = img.convert('L')
    grayscale_img = np.array(img, dtype=np.float)
    grayscale_img /= 255
    return grayscale_img


def preprocess_reward(reward):
    return np.clip(reward, -1000., 1000.)


if __name__ == '__main__':
    env = gym.make('ALE/MsPacman-v5')

    state_space_shape = env.observation_space.shape[:-1] + (1,)
    num_actions = env.action_space.n

    num_episodes = 500
    learning_rate = 0.01
    discount_factor = 1.0
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.1
    batch_size = 8
    memory_size = 5000

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    agent = DQN(state_space_shape, num_actions, model, target_model, learning_rate,
                discount_factor, batch_size, memory_size)

    total_rewards = []

    for episode in range(num_episodes):
        state = preprocess_state(env.reset())
        done = False
        rewards = 0
        while not done:
            action = agent.get_action(state, epsilon)
            new_state, reward, done, info = env.step(action)
            new_state = preprocess_state(new_state)
            reward = preprocess_reward(reward)
            agent.update_memory(state, action, reward, new_state, done)
            state = new_state
            rewards += reward
            if done:
                print(f'Episode: {episode}, reward: {rewards}')
                total_rewards.append(rewards)
                agent.train()
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
        if episode % 50 == 0:
            agent.update_target_model()
        if episode % 50 == 0:
            agent.save('mspacman', episode)

    agent.save('mspacman', num_episodes)

    print(np.mean(total_rewards))

    agent.load('mspacman', 500)

    done = False
    state = preprocess_state(env.reset())
    env.render()
    rewards = 0
    while not done:
        action = agent.get_action(state, min_epsilon)
        state, reward, done, info = env.step(action)
        state = preprocess_state(state)
        reward = preprocess_reward(reward)
        env.render()
        rewards += reward
        if done:
            time.sleep(2)
    print(rewards)
