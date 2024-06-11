
from Environment import TurnBasedFacilityPlacementEnv
from FacilityPlacementTask import FacilityPlacementTask
from SpatialObject import SpatialObject
import json
import os
import numpy as np

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

out_directory = 'fake_trajectories_from_z3'
taskset_path = 'tasksets/generated_tasks_10_terrain_10_constraints'
trajectory_folder = 'tasksets/generated_tasks_10_terrain_10_constraints/z3_result'

if __name__ == "__main__":
    batch_builder = SampleBatchBuilder() 
    env = TurnBasedFacilityPlacementEnv({'tasks_folder': taskset_path})

    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    for filename in os.listdir(trajectory_folder):
        trajectory_file = os.path.join(trajectory_folder, filename)
        if not filename.endswith('.json') or not os.path.isfile(trajectory_file):
            continue

        trajectory = json.load(open(trajectory_file, 'r'))
        recorded_actions = trajectory['trajectory']
        task_file = trajectory['task']
        init_locs = trajectory['init_locs_for_trajectory']
        writer = JsonWriter(os.path.join('fake_trajectories_from_z3', task_file.rstrip('.json')))

        obs = env.reset_with_task_and_init_locs(os.path.join(taskset_path, task_file), init_locs)
        prev_action = (0.0, 0.0)
        prev_reward = 0
        terminated = False
        t = 0
        while not terminated:
            action = recorded_actions[t]
            new_obs, rew, terminated, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=0,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0, 
                action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                truncateds=False,
                infos=info,
                new_obs=prep.transform(new_obs),
            )
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
            env.render(True, 1)
        writer.write(batch_builder.build_and_reset())
