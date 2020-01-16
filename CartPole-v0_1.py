import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

best_length = 0
episode_lengths = []

best_weights = np.zeros(4)

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        counter = 0

        while not done:
            #env.render()
            counter += 1

            action = 1 if np.dot(observation, new_weights) > 0 else 0

            observation, reward, done, _ = env.step(action)

            if done:
                break;

        length.append(counter)
    average_length = float(sum(length) / len(length))

    if average_length > best_length:
        best_length = average_length
        best_weights = new_weights
    episode_lengths.append(average_length)

    if i % 10 == 0:
        print('The best length is ', best_length)

done = False
counter = 0
#env = wrappers.Monitor(env, 'MovieFiles', force=True)
observation = env.reset()

while not done:
    env.render()
    counter += 1

    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation, reward, done, _ = env.step(action)

    if done:
        break;

print('The best game lasted ', counter, ' moves.')
