# PlotMap: Automated Layout Design for Building Game Worlds
![teaser](https://github.com/AutodeskAILab/PlotMap/assets/11589314/d12fe0ed-8414-47df-936a-58c8c12ba01a)

## Environment Setup
Dependent on the following libraries:
- Pytorch 2.0.x
- Ray(RLLib) 2.0.1
- Shapely 1.8.4
- Gym 0.23.1
- If there are other missing libraries, they can be simply installed with `pip install`

## RL Single Rollout

Use the following command to execute a single rollout

```
run_agent.py path_to_checkpont path_to_task
```
If successful, a json file namned `acility_positions.json` will be created in the current work directory with rollout information.

## RL Model Train
Task sets can be downloaded from https://github.com/AutodeskAILab/PlotMap-Dataset

Modify `facility_placement_ppo.yaml` as needed, then execute
```
python train.py -f facility_placement_ppo.yaml
```

## RL inference
After setting the path to the task set and the model to test with in line 22-23 in `run_gym.py`, execute
```
python run_gym.py
```
1000 rollouts will be performed and success rate will be reported after loading the task set (which may take some time). 

## CMA-ES solving/inference
Commands for running CMA-ES solving on example task
```
python fixed_solve.py --task_file='tasksets/task_1.json' --task_name='test_task' --output_file='results.json' --max_fevals=500 --fixed_indices=[0,1,2,3] --fixed_values=[35,95,6,15]
```
