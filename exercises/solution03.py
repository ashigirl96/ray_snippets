"""Solution of ray tutorial exercise 03"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import ray


# This is a proxy for a function which generates some data.
@ray.remote
def create_data(i):
  time.sleep(0.3)
  return i * np.ones(10000)


# This is a proxy for an expensive aggregation step (which is also
# commutative and associative so it can be used in a tree-reduce).
@ray.remote
def aggregate_data(x, y):
  time.sleep(0.3)
  return x * y


def main():
  ray.init(redirect_output=True, num_cpus=8)
  
  # Sleep a little to improve the accuracy of the timing measurements below.
  time.sleep(2.0)
  start_time = time.time()
  
  # EXERCISE: Here we generate some data. Do this part in parallel.
  vectors = [create_data.remote(i + 1) for i in range(8)]
  print(vectors)
  
  result = aggregate_data.remote(vectors[0], vectors[1])
  result = aggregate_data.remote(result, vectors[2])
  result = aggregate_data.remote(result, vectors[3])
  result = aggregate_data.remote(result, vectors[4])
  result = aggregate_data.remote(result, vectors[5])
  result = aggregate_data.remote(result, vectors[6])
  result = aggregate_data.remote(result, vectors[7])
  result1 = ray.get(result)
  
  end_time = time.time()
  duration = end_time - start_time
  print('Success! The example took {} seconds.'.format(duration))
  
  time.sleep(2.0)
  start_time = time.time()
  
  while len(vectors) > 1:
    vectors.append(aggregate_data.remote(vectors.pop(0), vectors.pop(0)))
  result2 = ray.get(vectors)
  
  np.testing.assert_equal(result1, result2[0], err_msg='must res1 equal res2')
  
  end_time = time.time()
  duration = end_time - start_time
  
  # assert duration < 0.3 + 0.9 + 0.3, ('FAILURE: The data generation and '
  #                                     'aggregation took {} seconds. This is '
  #                                     'too slow'.format(duration))
  # assert duration > 0.3 + 0.9, ('FAILURE: The data generation and '
  #                               'aggregation took {} seconds. This is '
  #                               'too fast'.format(duration))
  
  print('Success! The example took {} seconds.'.format(duration))


if __name__ == '__main__':
  main()