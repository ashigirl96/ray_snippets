"""Solution of ray tutorial exercise 04
Nested Parallelism.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray


@ray.remote
def compute_gradient(data):
  time.sleep(0.03)
  return 1


@ray.remote
def train_model(hyperparameters):
  """
  EXERCISE:
    Turn compute_gradient and train_model into
  remote functions so that they can be executed in parallel.

  Inside of train_model, do the calls to compute_gradient in parallel and
  fetch the results using ray.get.
  
  Returns:
    result that computed by ray.
  """
  result = 0
  for i in range(10):
    result += sum(
      ray.get([compute_gradient.remote(j) for j in range(2)]))
  return result


def main():
  ray.init(redirect_output=True, num_cpus=9)
  
  # Sleep a little to improve the accuracy of the timing measurements below.
  time.sleep(2.0)
  start_time = time.time()
  
  # Run some hyperparaameter experiments.
  results = []
  for hyperparameters in [{'learning_rate': 1e-1, 'batch_size': 100},
    {'learning_rate': 1e-2, 'batch_size': 100},
    {'learning_rate': 1e-3, 'batch_size': 100}]:
    results.append(train_model.remote(hyperparameters))
  
  end_time = time.time()
  duration = end_time - start_time
  
  results = ray.get(results)
  assert results == [20, 20, 20]
  assert duration < 0.35, ('The experiments ran in {} seconds. This is too '
                           'slow.'.format(duration))
  
  print('Success! The example took {} seconds.'.format(duration))


if __name__ == '__main__':
  main()