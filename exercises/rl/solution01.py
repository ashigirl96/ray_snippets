"""Solution of ray RL exercise 01"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


def sample_policy(state):
  return 0 if state[0] < 0 else 1


def rollout_policy(env, policy=sample_policy):
  """
  EXERCISE:
    Fill out this function by copying the 'random_rollout' function
    and then modifying it to choose the action using the policy.
  
  Args:
    env: Environment of OpenAI Gym
    policy: Agent's Policy

  Returns:
    cumulative_reward: Discounted episodic return
  """
  observ = env.reset()

  done = False
  cumulative_reward = 0

  while not done:
    # Choose a random action (either 0 or 1)
    action = policy(observ)
  
    # Take the action in the env.
    observ, reward, done, _ = env.step(action)
  
    # Update the cumulative reward.
    cumulative_reward += reward
  # Return the cumulative reward.
  return cumulative_reward


def random_rollout(env: gym.Env):
  observ = env.reset()
  
  done = False
  cumulative_reward = 0
  
  while not done:
    # Choose a random action (either 0 or 1)
    action = np.random.choice([0, 1])
    
    # Take the action in the env.
    observ, reward, done, _ = env.step(action)
    
    # Update the cumulative reward.
    cumulative_reward += reward
  # Return the cumulative reward.
  return cumulative_reward


def main():
  env = gym.make('CartPole-v0')
  reward = rollout_policy(env)
  print(reward)
  reward = rollout_policy(env)
  print(reward)


if __name__ == '__main__':
  main()