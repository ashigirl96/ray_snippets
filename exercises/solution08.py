"""Solution of ray tutorial exercise 08
Speed up Serialization

GOAL: The goal of this exercise is to illustrate how to speed up serialization by using ray.put.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray

ray.init(redirect_output=True, num_cpus=4)

neural_net_weights = {'variable{}'.format(i): np.random.normal(size=1_000_000)
  for i in range(50)}

