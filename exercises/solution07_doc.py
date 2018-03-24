"""Solution of ray tutorial exercise 07
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray

ray.init(redirect_output=True, num_cpus=10)


@ray.remote
class Counter(object):
  def __init__(self):
    self.value = 0
  
  def __getattr__(self, name):
    return getattr(self, name)
  
  def get_value(self):
    return self.value
  
  def increment(self):
    self.value += 1
    time.sleep(2.)
    return self.value


def measure_duration(func):
  time.sleep(2.0)
  start_time = time.time()
  
  result = func()
  
  end_time = time.time()
  duration = end_time - start_time
  print(duration, '[s]')
  
  return result


def main():
  counters = [Counter.remote() for _ in range(10)]
  
  results = measure_duration(
    lambda: ray.get([c.increment.remote() for c in counters]))
  print(results)
  
  results = measure_duration(
    lambda: ray.get([counters[0].increment.remote() for _ in range(4)]))
  print(results)
  
  results = measure_duration(
    lambda: ray.get([c.get_value.remote() for c in counters]))
  print(results)


if __name__ == '__main__':
  main()