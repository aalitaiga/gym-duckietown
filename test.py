from __future__ import division

import gym
import gym_duckietown

env = gym.make('DuckietownGrid-v0')

for i_episode in range(20):
    print(i_episode)
    observation = env.reset()
    for t in range(2001):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        # print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    if t == 2000:
        print("Episode finished after {} timesteps".format(t+1))