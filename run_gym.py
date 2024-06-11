from Environment import TurnBasedFacilityPlacementEnv
from FacilityPlacementTask import FacilityPlacementTask
from SpatialObject import SpatialObject
from ray.rllib.utils.filter import MeanStdFilter
from ConstraintTemplates import *
from ConstraintType import *
from gym import spaces
import pandas as pd
import json
import os
from complex_input_net import ComplexInputNetworkADSK

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

checkpoint_path = '/Users/wangyi/Workspace/ray_results/facility_placement/full_scale_samples/facility-placement-ppo-train/PPO_FACILITY_PlACEMENT_7d736_00000_0_2023-03-29_12-54-18/checkpoint_000270'
taskset_path = 'tasksets/full_scale_samples'

# Config is an object instead of a dictionary since Ray version >= 1.13.
config = PPOConfig()

config.framework(
     framework="torch",
)

## Point the PPO to our new environment class.
config.environment(
     env=TurnBasedFacilityPlacementEnv,
     env_config={'tasks_folder': taskset_path},
)


config.model["fcnet_hiddens"]=[512, 512]

config.in_evaluation = True

ppo = config.build()

ppo.config['in_evaluation'] = True

ppo.restore(checkpoint_path)

model = ppo.get_policy().model

#print(model)


env = TurnBasedFacilityPlacementEnv({'tasks_folder': taskset_path})

num_tries = 1000
success_count = 0
satisfaction_count = 0

for i in range(num_tries):
    print('episode ' + str(i))
    state = env.reset()
    done = False
    cumulative_reward = 0
    #env.render(True, 1)
    while not done:
        action = ppo.compute_action(state)
        state, reward, done, _ = env.step(action)

        print('action:', action)
        #env.render(True)
        if reward >= 1.0:
            success_count += 1
            break
        cumulative_reward += reward

        print('reward:', reward)

    satisfaction_count += env.fpTask.compute_sat_percentage()
    print('cumulated reward:', cumulative_reward)
    print('--------')
    #env.render(True, 1)
    print(str(success_count) + ' success out of ' + str(i+1) + ' tries')
    print('Average satisfaction percentage:', satisfaction_count / float(i+1))

print(str(success_count) + ' success out of ' + str(num_tries) + ' tries')