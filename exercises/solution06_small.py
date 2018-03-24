from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import numpy as np

ray.init(redirect_output=True)

@ray.remote
def f():
  return np.random.uniform(0, 1)


# Start 5 tasks.
remaining_ids = [f.remote() for i in range(5)]
print("Fistrt", remaining_ids)
# Whenever one task finishes, start a new one.
for _ in range(10):
  ready_ids, remaining_ids = ray.wait(remaining_ids)
  # Get the available object and do something with it.
  print(ray.get(ready_ids))
  # Start a new task.
  remaining_ids.append(f.remote())
