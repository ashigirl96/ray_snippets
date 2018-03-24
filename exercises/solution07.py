"""Solution of ray tutorial exercise 07
Introducing Actors.

GOAL: The goal of this exercise is to show how to create an actor and how to call actor methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray

ray.init(redirect_output=True, num_cpus=4)


@ray.remote
class Foo(object):
  
  def __init__(self):
    self.counter = 0
  
  def reset(self):
    self.counter = 0
  
  def increment(self):
    time.sleep(0.5)
    self.counter += 1
    return self.counter


def _func():
  f1 = Foo.remote()
  f2 = Foo.remote()
  
  f1.reset.remote()
  f2.reset.remote()
  
  results = []
  for _ in range(5):
    results.append(f1.increment.remote())
    results.append(f2.increment.remote())
  
  return ray.get(results)


def main():
  time.sleep(2.0)
  start_time = time.time()
  
  results = _func()
  
  end_time = time.time()
  duration = end_time - start_time
  
  assert results == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
  assert duration < 3, ('The experiments ran in {} seconds. This is too '
                        'slow.'.format(duration))
  assert duration > 2.5, ('The experiments ran in {} seconds. This is too '
                          'fast.'.format(duration))
  print('Success! The example took {} seconds.'.format(duration))


if __name__ == '__main__':
  main()