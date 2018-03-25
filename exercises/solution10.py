"""Solution of ray tutorial exercise 10
Pass Neural Net Weights Between Processes

GOAL: The goal of this exercise is to
      show how to send neural network weights between workers and the driver.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import tensorflow as tf


# プロセス間の重みをnumpy配列の辞書として（または平坦なnumpy配列として）配送することが最も効率的です。


@ray.remote
class SimpleModel(object):
  def __init__(self):
    x_data = tf.placeholder(tf.float32, shape=[100])
    y_data = tf.placeholder(tf.float32, shape=[100])
    
    w = tf.Variable(tf.random_uniform([1], -1., 1.))
    b = tf.Variable(tf.zeros([1]))
    y = w * x_data + b
    
    self.loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    grads = optimizer.compute_gradients(self.loss)
    self.train_op = optimizer.apply_gradients(grads)
    
    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.vars = ray.experimental.TensorFlowVariables(self.loss, self.sess)
    
    self.sess.run(init)
  
  def set_weights(self, weights):
    """Set the neural net weights.
    
    This method should assign the given weights to the neural net.
    """
    self.vars.set_flat(weights)
  
  def get_weights(self):
    """Get the neural net weights.
    
    This method should return the current neural net weights.
    """
    weights = self.vars.get_flat()
    return weights


def main(_):
  ray.init(redirect_output=True, num_cpus=4, num_gpus=2)
  
  actors = [SimpleModel.remote() for _ in range(4)]
  
  # EXERCISE: Get the neural weights from all of the actors.
  weights = []
  for i in range(4):
    weights.append(ray.get(actors[i].get_weights.remote()))
  
  # EXERCISE: Average all of the neural net weights.
  weights = np.mean(weights, axis=0)
  
  # EXERCISE: Set the average weights on the actors.
  for actor in actors:
    actor.set_weights.remote(weights)
  
  weights = ray.get([actor.get_weights.remote() for actor in actors])
  # VERIFY: Check that all of the actors have the same weights.
  for i, actor in enumerate(actors):
    np.testing.assert_equal(weights[i], weights[0])
  print('Success! The test passed.')
  
  print(weights[0])


if __name__ == '__main__':
  tf.app.run()