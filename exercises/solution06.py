"""Solution of ray tutorial exercise 06
Process Tasks in Order of Completion

GOAL: The goal of this exercise is to show how to use
      ray.wait to process tasks in the order that they finish.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import ray

ray.init(redirect_output=True, num_cpus=5)


@ray.remote
def f():
  time.sleep(np.random.uniform(0, 5))
  return time.time()


def main():
  time.sleep(2.0)
  start_time = time.time()
  
  result_ids = [f.remote() for _ in range(10)]
  
  # Get the results.
  results = []
  for _ in range(10):
    result, result_ids = ray.wait(result_ids, num_returns=1)
    result = ray.get(result[0])
    results.append(result)
    print('Processing result which finished after {} '
          'seconds.'.format(result - start_time))
  
  end_time = time.time()
  duration = end_time - start_time
  
  # Verify.
  assert results == sorted(results), ('The results were not processed in the '
                                      'order that they finished.')
  print('Success! The example took {} seconds.'.format(duration))


if __name__ == '__main__':
  main()