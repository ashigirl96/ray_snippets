from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import ray


@ray.remote
def get_batches(i):
  np.random.seed(5 + i)
  x = np.random.uniform(0, 4)
  time.sleep(x)
  return i, time.time()


@ray.remote
def batch_train(res, batch_size=3):
  time.sleep(batch_size)
  return res


def main():
  ray.init(redirect_output=True, num_cpus=6)
  
  # Sleep a little to improve the accuracy of the timing measurements below.
  time.sleep(2.0)
  start_time = time.time()
  
  result_ids = [get_batches.remote(i) for i in range(6)]
  for i in range(2):
    batches, result_ids = ray.wait(result_ids, num_returns=3)
    res = batch_train.remote(batches, 3)
    print(ray.get(res))
    
    batch_time = time.time()
    batch_duration = batch_time - start_time
    print(batch_duration)
  
  # Sleep a little to improve the accuracy of the timing measurements below.
  time.sleep(2.0)
  start_time = time.time()
  
  result_ids = [get_batches.remote(i) for i in range(6)]
  batches = ray.get(result_ids)
  res = batch_train.remote(batches, batch_size=6)
  res = ray.get(res)
  end_time = time.time()
  print(end_time - start_time)


if __name__ == '__main__':
  main()