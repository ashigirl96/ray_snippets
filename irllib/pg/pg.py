"""Simple policy gradient agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from ray.rllib.agent import Agent
from ray.rllib.optimizers import LocalSyncOptimizer

def default_config():
  num_workers = 4
  batch_size = 512
  gamma = 0.99
  horizon = 500
  lr = 0.0004
  optimizer = {}
  model = {'fcnet_hiddens': [128, 128]}
  return locals()

class PGAgent(Agent):
  """Simple policy gradient agent.
  
  This is an example agent to show how to implement algorithms in RLlib.
  In most cases, you will probably want to use the PPO agent instread.
  """
  _agent_name = 'PG'
  _default_config = default_config()
  
  def _init(self):
    pass
    
