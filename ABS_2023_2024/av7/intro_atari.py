import numpy as np
import gymnasium as gym
from PIL import Image
from deep_q_learning import DuelingDQN
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten


def preprocess_state(state):
    img = Image.fromarray(state)
    img2 = img.convert('L')

    img3 = np.array(img2, dtype=np.float)
    img3 /= 255


if __name__ == '__main__':
    env = gym.make('ALE/MsPacman-v5', render_mode='human')
    state, _ = env.reset()
    env.render()

    state, _, _, _, _ = env.step()
    processed_states = preprocess_state(state)

    layers = [Conv2D(32, activation='relu'),
              MaxPool2D(),
              Conv2D(16, activation='relu'),
              MaxPool2D(),
              Flatten()]

    agent = DuelingDQN(...)
    agent.build_model(layers)
    print()
