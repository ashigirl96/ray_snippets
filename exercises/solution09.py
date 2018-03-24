"""Solution of ray tutorial exercise 09
Using the GPU API

GOAL: The goal of this exercise is to show how to use GPUs
      with remote functions and actors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray

ray.init(redirect_output=True, num_cpus=4, num_gpus=2)


@ray.remote(num_gpus=2)
def f():
  time.sleep(0.5)
  return ray.get_gpu_ids()


@ray.remote(num_gpus=2)
class Actor(object):
  def __init__(self):
    pass
  
  def get_gpu_ids(self):
    return ray.get_gpu_ids()


def main():
  time.sleep(2.0)
  start_time = time.time()
  
  gpu_ids = ray.get([f.remote() for _ in range(3)])
  
  end_time = time.time()
  
  for i in range(len(gpu_ids)):
    assert len(gpu_ids[i]) == 2
  assert end_time - start_time > 1
  
  actor = Actor.remote()
  gpu_ids = ray.get(actor.get_gpu_ids.remote())
  assert len(gpu_ids) == 2


if __name__ == '__main__':
  main()