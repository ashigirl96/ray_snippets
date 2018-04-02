"""Solution of ray RL exercise 01"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import ray
import tensorflow as tf
from tensorflow import keras

ray.init()
ray.error_info()


class TwoLayerPolicy(keras.Model):
  def __init__(self, num_inputs, num_hiddens, num_outputs=1):
    super(TwoLayerPolicy, self).__init__(name='two_layer_policy')
    self.dense1 = keras.layers.Dense(num_hiddens, activation=tf.nn.relu)
    self.dense2 = keras.layers.Dense(num_outputs)
  
  def call(self, observ, training=None, mask=None):
    hidden = self.dense1(observ)
    output = self.dense2(hidden)
    assert output.shape.as_list()[1] == 1
    # return 0 if np.all(output.numpy() < 0) else 1
    return tf.cond(tf.reduce_all(output < 0),
                   lambda: tf.zeros_like(output),
                   lambda: tf.ones_like(output))


def rollout_policy(env, policy: keras.Model):
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
  observ = observ[np.newaxis, ...]
  
  done = False
  cumulative_reward = 0
  
  while not done:
    # Choose a random action (either 0 or 1)
    action = policy.predict(observ)
    action = int(action.squeeze())
    
    # Take the action in the env.
    observ, reward, done, _ = env.step(action)
    observ = observ[np.newaxis, ...]
    
    # Update the cumulative reward.
    cumulative_reward += reward
  # Return the cumulative reward.
  return cumulative_reward


@ray.remote
def evaluate_random_policy(num_rollouts):
  # Generate a random policy.
  policy = TwoLayerPolicy(4, 5)
  
  # Create an environment.
  env = gym.make('CartPole-v0')
  
  # Evaluate the same policy multiple times
  # and then take the average in order to evaluate the policy more accurately
  returns = [rollout_policy(env, policy) for _ in range(num_rollouts)]
  return np.mean(returns), np.max(returns)


def main():
  tf.enable_eager_execution()
  
  # Evaluate 100 randomaly generated policies.
  average_100_rewards, best_100_rewards = ray.get(evaluate_random_policy.remote(100))
  # Print the best score obtained.
  print("100 Policy:\n\tAverage return(total return): {0}\n\tBest return: {1}".format(
    average_100_rewards, best_100_rewards))
  
  # Evaluate 100 randomaly generated policies.
  average_1000_rewards, best_1000_rewards = ray.get(evaluate_random_policy.remote(1000))
  # Print the best score obtained.
  print("1000 Policy:\n\tAverage reward: {0}\n\tBest reward: {1}".format(
    average_1000_rewards, best_1000_rewards))


if __name__ == '__main__':
  main()