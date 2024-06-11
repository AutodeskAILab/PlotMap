import gym
from gym import spaces
from gym.spaces import Dict
import numpy as np
import itertools
from shapely.geometry import Point, Polygon
import cv2
import pandas as pd
import random
import os
from PIL import Image
from img2vec_pytorch import Img2Vec
from sentence_transformers import SentenceTransformer
from ColorScheme import *
from ConstraintTemplates import TERRAIN_TYPES
from termcolor import colored

from ConstraintType import *
from FacilityPlacementTask import *
from LayoutDesignTaskJsonParser import *
from FacilityPlacementTaskRenderer import FacilityPlacementTaskRenderer

img2vec = Img2Vec()
CONSTRAINT_EMBEDDING_SIZE = 384
MAX_NUM_ARGS = 3
ARG_MASK = '#'

CONSTRAINTS = PopulateAllInstantiations(
	ConstraintType.constraint_library.keys(),
	TERRAIN_TYPES, ['obj_'+ str(i) for i in range(10)])

#print('constraint instantiations:', constraints)
#print('number of constraint instantiations:', len(constraints))

class TurnBasedFacilityPlacementEnv(gym.Env):

	def __init__(self, config):
		super().__init__()

		self.tasks_folder = config['tasks_folder']

		self.num_facilities = config.get("num_facilities", 10)
		self.num_terrain_tags = config.get("num_terrain_tags", len(TERRAIN_TYPES) + len(AUX_TERRAIN_TYPES))
		self.num_facility_tags = config.get("num_facility_tags", 10)
		self.max_dist = config.get("max_dist", 0.1)
		self.max_steps = config.get("max_steps", 20 * self.num_facilities)
		self.obs_map_shape = config.get("obs_map_shape", (42, 42))
		self.vision_only = config.get("vision_only", False)
		self.fixed_terrain_vision = config.get("fixed_terrain_vision", True)
		self.rwd_full_sat_weight = config.get("rwd_full_sat_weight", 0.8)
		self.max_num_constraints =  config.get("max_num_constraints", 10)
		
		self.reset_with_init_loc = config.get("init_locs", None)
		self.reset_with_fixed_task = config.get("fixed_task", None)
		self.facility_locking_states = config.get("facility_locking_states", [0] * self.num_facilities)

		self.obs_terrain_cache = None
		self.dimension = len(self.obs_map_shape)
		self.facility_tags = [[]]

		self.constraint_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

		# Action space
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dimension, ), dtype=np.float32)

		print('number of constraint types:', len(ConstraintType.constraint_list))

		# Observation space
		# Currently only supports single tag for facilities
		self.obs_facility_shape = (self.num_facilities, self.dimension + 2)
		self.obs_facility_flattened_shape = self.num_facilities * (self.dimension + 2)
		#self.obs_constraint_shape = (len(ConstraintType.constraint_list) + MAX_NUM_ARGS * self.num_facilities, self.max_num_constraints)
		self.obs_constraint_flattend_shape = len(CONSTRAINTS)
		if self.vision_only:
			self.obs_vision_shape = tuple(list(self.obs_map_shape) + [self.num_terrain_tags + self.num_facility_tags ])
			self.obs_vision_shape_self = tuple(list(self.obs_map_shape) + [self.num_terrain_tags + self.num_facility_tags + 1])
			#self.observation_space = spaces.Box(low = -1, high = 1, shape = self.obs_vision_shape_self, dtype=int)
			self.obs_vision_shape = (512, )
			self.observation_space = spaces.Tuple((spaces.Box(low = 0.0, high = 10.0, shape = self.obs_vision_shape, dtype=np.float64),
										spaces.Box(low = -1, high = 100, shape = self.obs_constraint_shape, dtype=np.int32)))
		else:
			self.obs_vision_shape = (512, )
			self.observation_space = spaces.Tuple((spaces.Box(low = 0.0, high = 10.0, shape = self.obs_vision_shape, dtype=np.float64),
						     spaces.Box(low = -1.0, high = float(self.num_facility_tags), shape = (self.obs_facility_flattened_shape, ), dtype=np.float64),
								 spaces.Box(low = 0, high = 1, shape = (self.obs_constraint_flattend_shape, ), dtype=np.int32)))
		self.reset()

	def step(self, action):
		self.move_curr_agent(action)
		self.next_step()

		obs = self._get_obs(self.currAgent)
		reward = self._get_reward()
		done = self._get_done()

		if done:
			print('reward:' + str(reward))

		info = {}

		return obs, reward, done, info

	def reset(self):
		# Seeding
		np.random.seed(seed=None)

		# Initialize
		self.step_count = 0
		self.currAgent = 0
		self.fpTask = None

		self._load_task()
	
		print(colored('loaded task: {}'.format(self.fpTask.Task_id), 'green'))

		self.obs_terrain_cache = None
		self.obs_constraint_cache = None

		self.renderer = FacilityPlacementTaskRenderer(self.fpTask)

		self._initialize_facility_locs()

		# Return initial observation
		obs = self._get_obs(self.currAgent)
		return obs
	
	def _load_task(self):
		if self.reset_with_fixed_task != None:
			try:
				task_name = self.reset_with_fixed_task.rstrip('.json')
				f = os.path.join(self.tasks_folder, self.reset_with_fixed_task)
				self.fpTask = FacilityPlacementTask.load_from_json(json.load(open(f, 'r')), task_name)
			except Exception:
				print('Failed to load specified task. Loading a random task.')
		
		if self.fpTask == None:
			# Load a random task
			filenames = os.listdir(self.tasks_folder)
			task_name = None
			while task_name is None or not task_name.endswith('.json'): 
				task_name = random.choice(filenames)
			f = os.path.join(self.tasks_folder, task_name)
			self.fpTask = FacilityPlacementTask.load_from_json(json.load(open(f, 'r')), f)

	def _initialize_facility_locs(self):
		if self.reset_with_init_loc != None:
			# Set facilities to be at the specified location
			for i in range(len(self.fpTask.Facillities)):
				self.fpTask.Facillities[i].Polygon = [[self.reset_with_init_loc[i][0], self.reset_with_init_loc[i][1]]]
		else:
			# Set facilities to be at random location
			for facility in self.fpTask.Facillities:
				facility.Polygon = [list(np.random.rand(self.dimension) * self.fpTask.Map_scale)]


	def _get_obs(self, facility_id):
		local_fov = self.sample_FOV(self.fpTask.Facillities[facility_id])

		return local_fov

	def _get_done(self):
		if self.step_count >= self.max_steps:
			return True
		else:
			return False

	def _get_reward(self):
		# reward is shared among all agents
		sat_val = self.fpTask.evaluate()
		if sat_val == 1.0:
			return 1.0
		else:
			return sat_val - 1.0
		# full_sat_rwd = 0.0
		# if sat_val >= 1.0:
		# 	full_sat_rwd = 1.0

		# return sat_val * (1 - self.rwd_full_sat_weight) + full_sat_rwd * self.rwd_full_sat_weight

	def render(self, visualize = False, waitKey = 1):
		if visualize:
			# OpenCV visualization
			self.renderer.render_task(waitKey)
			self.renderer.render_obs(self.obs_cache, waitKey)

		# print positions of all agents
		print('step count: ' + str(self.step_count))
		print('current agent: ' + self.fpTask.Facillities[self.currAgent].Id)
		for facility_id in range(len(self.fpTask.Facillities)):
			print('facility ' + self.fpTask.Facillities[facility_id].Id, self.fpTask.Facillities[facility_id].Polygon[0])

	def next_step(self):
		self.step_count += 1
		self.currAgent = self.step_count % self.num_facilities

	def move_curr_agent(self, action):
		if self.facility_locking_states[self.currAgent] == 1:
			return
		prev_pos = np.asarray(self.fpTask.Facillities[self.currAgent].Polygon[0])
		new_pos = prev_pos + np.asarray(action)  * self.fpTask.Map_scale * self.max_dist
		# Make sure facilities stay inside canvas
		new_pos = [min(max(0.0, new_pos[i]), self.fpTask.Map_scale[i]) for i in range(len(new_pos))]
		self.fpTask.Facillities[self.currAgent].Polygon = [new_pos]

	def create_terrain_vision(self):
		terrain_view = np.zeros(list(self.obs_map_shape) + [3], dtype=np.int32)
		for terrain_obj in self.fpTask.Terrain_objects:
			for tag in terrain_obj.Tags:
				if tag not in TERRAIN_TYPES:
					continue
				idx = self.fpTask.Terrain_tags.index(tag)
				for poly in terrain_obj.Polygon:
					pts = np.array([np.asarray(p) / np.asarray(self.fpTask.Map_scale) * np.asarray(self.obs_map_shape) for p in poly], np.int32)
					tmp = Polygon(pts)
					tmp = tmp.simplify(0)
					xx, yy = tmp.exterior.coords.xy
					pts = np.array([[int(xx[i]), int(yy[i])] for i in range(len(xx))])
					#print('pts:', pts)
					terrain_view = cv2.fillPoly(terrain_view, [pts], [x * 255 for x in TerrainColors[idx]])

		return terrain_view

	def create_constraint_obs(self):
		constraint_obs = np.full((self.obs_constraint_flattend_shape, ), 0)
		for constraint in self.fpTask.Constraints:
			constraintStr = GetConstraintStr(constraint)
			constraint_obs[CONSTRAINTS.index(constraintStr)] = 1
			#print('constriant ' + str(CONSTRAINTS.index(constraintStr)) + ' added')
		#print('constraint obs: ', constraint_obs)

		return constraint_obs

	def create_facility_vision(self, img, radius = 2):
		for facility in self.fpTask.Facillities:
			for tag in facility.Tags:
				idx = self.fpTask.Facility_tags.index(tag)
				position = np.asarray(facility.Polygon[0]) / np.asarray(self.fpTask.Map_scale) * np.asarray(self.obs_map_shape)
				position = position.astype(np.int32)
				img = cv2.circle(img, position, radius=radius, color=FacilityColor, thickness=-1)

	def create_vision_self_indicator(self, img, subject_facility, radius = 3):
		position = np.asarray(subject_facility.Polygon[0]) / np.asarray(self.fpTask.Map_scale) * np.asarray(self.obs_map_shape)
		position = position.astype(np.int32)
		img = cv2.circle(img, position, radius=radius, color=SelfIndicatorColor, thickness=-1)

	def create_facility_matrix(self, subject_facility):
		obs_facility = np.full(self.obs_facility_shape, -1.0)
		# Information about self
		for i in range(self.dimension):
			obs_facility[0][i] = float(subject_facility.Polygon[0][i]) / float(self.fpTask.Map_scale[i])
		# Self indicator
		obs_facility[0][self.dimension] = 1.0
		assert(subject_facility.Tags[0] in self.fpTask.Facility_tags) 
		obs_facility[0][self.dimension + 1] = float(self.fpTask.Facility_tags.index(subject_facility.Tags[0]))
		# Information about other facilities
		facility_cnt = 1
		for f in self.fpTask.Facillities:
			if f == subject_facility:
				continue
			for i in range(self.dimension):
				obs_facility[facility_cnt][i] = float(f.Polygon[0][i]) / float(self.fpTask.Map_scale[i])
			obs_facility[facility_cnt][self.dimension] = 0.0
			# currently only care about the first facility tag
			assert(f.Tags[0] in self.fpTask.Facility_tags) 
			obs_facility[facility_cnt][self.dimension + 1] = float(self.fpTask.Facility_tags.index(f.Tags[0]))
			facility_cnt += 1

		return obs_facility

	def sample_FOV(self, subject_facility):
		position = subject_facility.Polygon[0]

		if self.obs_terrain_cache is None or self.fixed_terrain_vision == False:
			self.obs_terrain_cache = self.create_terrain_vision()

		if self.obs_constraint_cache is None:
			self.obs_constraint_cache = self.create_constraint_obs()

		if self.vision_only:
			self.obs_cache = np.array(self.obs_terrain_cache, copy=True)
			self.create_facility_vision(self.obs_cache)
			self.create_vision_self_indicator(self.obs_cache, subject_facility)
			obs_img = Image.fromarray(self.obs_cache.astype('uint8'), 'RGB')
			obs_vec = img2vec.get_vec(obs_img)
			return (obs_vec, self.obs_constraint_cache)
		else:
			facility_matrix = self.create_facility_matrix(subject_facility)
			self.obs_cache = self.obs_terrain_cache
			terrain_img = Image.fromarray(self.obs_cache.astype('uint8'), 'RGB')
			terrain_vec = img2vec.get_vec(terrain_img)
			return (terrain_vec, facility_matrix.flatten(), self.obs_constraint_cache)
			#return (terrain_vec, facility_matrix.flatten())
'''
legacy code, to be removed 
'''
# def createTaskLibrary(task_folder):
# 	print('creating task library....')
# 	task_cache = []
# 	for filename in os.listdir(task_folder):
# 		f = os.path.join(task_folder, filename)
# 		if filename.endswith('.json') and os.path.isfile(f):
# 			print('adding task ' + f)
# 			task_cache.append(
# 				FacilityPlacementTask.load_from_json(
# 					json.load(open(f, 'r'))))
# 	print('finished creating task library.')

# 	return task_cache

# def env_creator(env_config):
	
# 	tasks = createTaskLibrary(env_config['tasks_folder'])
	
# 	env = TurnBasedFacilityPlacementEnv({'tasks': tasks})

#     return env

def env_creator(env_config):
	
	env = TurnBasedFacilityPlacementEnv({'tasks_folder': env_config['tasks_folder']})

	return env

if __name__ == '__main__':

	init_locs = [
    [
      45.601869489933904,
      -4.3790577010150533e-47
    ],
    [
      100.0,
      100.0
    ],
    [
      25.41098882779479,
      100.0
    ],
    [
      0.0,
      8.306197734518788
    ],
    [
      79.88606895833237,
      0.0
    ],
    [
      100.0,
      0.0
    ],
    [
      100.0,
      -1.0795210693868056e-78
    ],
    [
      0.0,
      10.590971442398315
    ],
    [
      99.22560239678754,
      37.55961306654223
    ],
    [
      58.059893032670864,
      58.0576677584049
    ]
  ]

	locking_state = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
	tasks_folder = 'tasksets/generated_tasks_10_terrain_15_constraints/'
	env = TurnBasedFacilityPlacementEnv({'tasks_folder': tasks_folder,
									     'fixed_task': 'task_5.json',
										 'init_locs': init_locs,
										 })

	sample = env.observation_space.sample()
	#print('observation shape:', sample[0].shape, sample[1].shape)

	#print('Sample observation:', sample)
	#terrain_obs, facility_obs = sample
	print('Observation shape:', sample[0].shape, sample[1].shape, sample[2].shape)
	init_obs = env.reset()

	print('initial observation:', init_obs)
	print('observation shape:', init_obs[0].shape, init_obs[1].shape, init_obs[2].shape)
	#print('observation dtype:', init_obs[0].dtype, init_obs[1].dtype)
	env.render(True, 0)

	actions = [0.1, 0.1]
	state, reward, done, info = env.step(actions)
	print('reward:', reward)
	#print('observation:', state)
	env.render(True, 0)

	actions = [0.1, 0.1]
	state, reward, done, info = env.step(actions)
	print('reward:', reward)
	#print('observation:', state)
	env.render(True, 0)

	actions = [0.1, 0.1]
	state, reward, done, info = env.step(actions)
	print('reward:', reward)
	#print('observation:', state)
	env.render(True, 0)

	actions = [0.1, 0.1]
	state, reward, done, info = env.step(actions)
	print('reward:', reward)
	#print('observation:', state)
	env.render(True, 0)

	actions = [0.1, 0.1]
	state, reward, done, info = env.step(actions)
	print('reward:', reward)
	#print('observation:', state)
	env.render(True, 0)

	#actions = [-0.3, -0.2]
	#state, reward, done, info = env.step(actions)
	#print('reward:', reward)
	#print('observation:', state)
	#env.render(True, 0)

	#for i in range(10):
	#	print('step ' + str(i))
	#	state, reward, done, info = env.step(action
