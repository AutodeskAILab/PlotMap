from Environment import TurnBasedFacilityPlacementEnv
from FacilityPlacementTask import FacilityPlacementTask
from SpatialObject import SpatialObject
from gym import spaces
import json
import os
from datetime import datetime
from ConstraintType import *
from termcolor import colored
import sys

from ray.rllib.algorithms.ppo import PPOConfig

def build_model(model_path, taskset_path):
    config = PPOConfig()

    config.framework(
         framework="torch",
    )

    config.environment(
         env=TurnBasedFacilityPlacementEnv,
         env_config={'tasks_folder': taskset_path},
    )


    config.model["fcnet_hiddens"]=[1024, 512, 512]
    config.in_evaluation = True

    ppo = config.build()
    ppo.config['in_evaluation'] = True
    ppo.restore(checkpoint_path)

    return ppo


def run_facility_placement_agent(model_path, taskset_path, json_path = None):

    ppo = build_model(model_path, taskset_path)
    env = TurnBasedFacilityPlacementEnv({'tasks_folder': taskset_path})

    score = -1.0
    step_count = 0
    state = env.reset()
    done = False

    step_count = 0
    outjson = {'task_id': env.fpTask.Task_id.split('/')[-1],
               'scale': env.fpTask.Map_scale,
               'rollout_time': str(datetime.now()),
               'timesteps': []}

    facility_positions = [{'facility_id': obj.Id, 'location': obj.Polygon[0]} for obj in env.fpTask.Facillities]
    outjson['timesteps'].append({'step': 0, 'facility_positions': facility_positions})

    env.render(True, 1)
    while not done:
        action = ppo.compute_action(state)
        state, reward, done, _ = env.step(action)
        env.render(True)

        step_count += 1
        facility_positions = [{'facility_id': obj.Id, 'location': obj.Polygon[0]} for obj in env.fpTask.Facillities]
        outjson['timesteps'].append({'step': step_count, 'facility_positions': facility_positions})
        if reward >= 1.0:
            break

    score = env.fpTask.evaluate() 
    print(colored('score:' + str(score), 'green'))

    if json_path != None:
        json.dump(outjson, open(json_path, 'w'), indent=2)

    return outjson

if __name__== '__main__':
    if len(sys.argv) < 3:
        print('Usage: run_agent.py path_to_checkpont path_to_task')
        exit(0)
    checkpoint_path = sys.argv[1]
    taskset_path = sys.argv[2]
    run_facility_placement_agent(checkpoint_path, 
        taskset_path, 'facility_positions.json')