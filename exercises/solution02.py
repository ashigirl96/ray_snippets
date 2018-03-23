"""Solution of ray tutorial exercise 02"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import ray


@ray.remote
def _load_data(filename):
  time.sleep(0.1)
  return np.ones((1000, 100))


@ray.remote
def _normalize_data(data):
  time.sleep(0.1)
  return data - np.mean(data, axis=0)


@ray.remote
def _extract_features(normalized_data):
  time.sleep(0.1)
  return np.hstack([normalized_data, normalized_data ** 2])


@ray.remote
def _compute_loss(features):
  num_data, dim = features.shape
  time.sleep(0.1)
  return np.sum((np.dot(features, np.ones(dim)) - np.ones(num_data)) ** 2)


def main():
  ray.init('192.168.12.3:6349', redirect_output=True)
  
  time.sleep(2.)
  start_time = time.time()
  
  losses = []
  filenames = ['file1', 'file2', 'file3', 'file4']
  for filename in filenames:
    data = _load_data.remote(filename)
    normalized_data = _normalize_data.remote(data)
    features = _extract_features.remote(normalized_data)
    loss = _compute_loss.remote(features)
    losses.append(loss)
  
  loss = sum(ray.get(loss) for loss in losses)
  
  end_time = time.time()
  duration = end_time - start_time
  
  assert loss == 4000
  assert duration < 0.8, ('The loop took {} seconds. This is too slow.'.format(duration))
  assert duration > 0.4, ('The loop took {} seconds. This is too fast.'.format(duration))
  
  print('Success! The example took {} seconds.'.format(duration))
  print("Check asserts/solution02.png")


if __name__ == '__main__':
  main()