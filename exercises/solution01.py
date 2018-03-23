"""Solution of ray tutorial exercise 01"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray


@ray.remote
def slow_function(i):
	"""This function is a proxy for a more interesting and computationally
	intensive function.
	"""
	time.sleep(1)
	return i


def measure_duration():
	"""Sleep a little to improve the accuracy of the timing measurements below.
	We do this because workers may still be starting up in the background.
	"""
	time.sleep(2.0)
	start_time = time.time()
	
	results = ray.get([slow_function.remote(i) for i in range(4)])
	
	end_time = time.time()
	duration = end_time - start_time
	return results, duration


def main():
	ray.init('192.168.12.3:6349', redirect_output=True)
	
	# EXERCISE:
	print(slow_function.remote(0))
	print(ray.get(slow_function.remote(0)))
	
	# EXERCISE:
	results, duration = measure_duration()
	assert results == [0, 1, 2, 3], 'Did you remember to call ray.get?'
  assert duration < 1.1, ('The loop took {} seconds. This is too slow.'.format(duration))
  assert duration > 1, ('The loop took {} seconds. This is too fast.'.format(duration))
	print('Success! The example took {} seconds.'.format(duration))


# EXERCISE:
# Look assets/solution01.png


if __name__ == '__main__':
	main()