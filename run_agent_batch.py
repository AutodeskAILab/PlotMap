from Environment import TurnBasedFacilityPlacementEnv
from FacilityPlacementTask import FacilityPlacementTask
from SpatialObject import SpatialObject
from gym import spaces
import json
import os
from datetime import datetime
from ConstraintType import *
from termcolor import colored

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
    #while score < 0.0:
    while score < 0.99:
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
            #action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            env.render(True)

            step_count += 1
            facility_positions = [{'facility_id': obj.Id, 'location': obj.Polygon[0]} for obj in env.fpTask.Facillities]
            outjson['timesteps'].append({'step': step_count, 'facility_positions': facility_positions})
            if reward >= 1.0:
                break

        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_0']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_1']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_2']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_3']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_4']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_5']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_6']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_7']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_8']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideOCEAN'], ['obj_9']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_0']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_1']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_2']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_3']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_4']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_5']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_6']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_7']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_8']))
        env.fpTask.Constraints.append((ConstraintType.constraint_library['OutsideDEEPOCEAN'], ['obj_9']))
        score = env.fpTask.evaluate() 
        print(colored('score:' + str(score), 'green'))

    if json_path != None:
        print('success!')
        json.dump(outjson, open(json_path, 'w'), indent=2)

    return outjson

if __name__== '__main__':
    checkpoint_path = '/Users/wangyi/Workspace/ray_results/facility_placement/50_task_constraint_idx_obs_one_hot_1024_512_/facility_placement_ppo/PPO_FACILITY_PlACEMENT_9d43a_00000_0_2023-05-08_09-08-48/checkpoint_000133'
    taskset_path = 'tasksets/debug'
    run_facility_placement_agent(checkpoint_path, 
        taskset_path, 'facility_positions.json')