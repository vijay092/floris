import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from agent import Agent
from collections import deque
import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
atest = env.action_space.sample()
x_next,c,done,info = env.step(atest)
agent = Agent(state_size = len(x_next), action_size=len(atest),random_seed=1)

def ddpg(numEpisodes = 100, maxTime = 10000):
    scores_deque = deque(maxlen=100)
    total_scores = []
    for ep in range(numEpisodes):
        x = env.reset();
        scores = 0;
        env.render()
        for step in range(maxTime):
            a = agent.act(x.T);
            x_next,c,done,info = env.step(a)
            agent.step(x.T, a, c, x_next.T, done)
            scores += c
            x = x_next
            
            if np.any(done):                                  # exit loop if episode finished
                break
            
        scores_deque.append(np.mean(scores))
        total_scores.append(np.mean(scores))
ddpg()
env.close()