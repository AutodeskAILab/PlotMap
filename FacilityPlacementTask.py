from LayoutDesignTaskJsonParser import *
from ConstraintTemplates import *
from ConstraintType import *
import json
from shapely.geometry import Point, Polygon
import shapely
from itertools import product
import random
from z3 import *

def merge_terrain_object(terrain_objects):
	tag_separator = ','

	merged_terrain_objects = []
	tag_dict = {}

	for obj in terrain_objects:
		tag_str = tag_separator.join(obj.Tags)
		if tag_str not in tag_dict:
			tag_dict[tag_str] = [obj.Polygon]
		else:
			tag_dict[tag_str].append(obj.Polygon)

	cnt = 1
	for tag_str in tag_dict:
		new_id = 'merged_' + str(cnt)
		tags = []
		if tag_str != '':
			tags = tag_str.split(tag_separator)
		new_obj = SpatialObject(new_id, tag_dict[tag_str], tags, tag_str)
		merged_terrain_objects.append(new_obj)
		cnt += 1

	#print('merged_terrain_objects:', len(merged_terrain_objects))
	return merged_terrain_objects


class FacilityPlacementTask():

	def __init__(self, task_id, facility_tags, terrain_tags,
				 map_scale, facillities, terrain_objects, constraints, constraintText,
				 facility_radius = 0.03):

		self.Facility_tags = facility_tags
		self.Terrain_tags = terrain_tags
		self.Map_scale = map_scale
		self.Facillities = facillities
		self.Terrain_objects = merge_terrain_object(terrain_objects)
		self.Constraints = constraints
		self.ConstraintText = constraintText
		self.Facility_radius = facility_radius
		self.Self_radius = 0.1
		self.Terrain_cache = TerrainCache(self.Terrain_objects)
		self.Task_id = task_id

	def evaluate(self):
		values = []
		for constraint, args in self.Constraints:
			values.append(constraint.sat_eval(args, self.Facillities, self.Terrain_cache))
			#print(constraint.generate_utterance(args)[0], args, constraint.sat_eval(args, self.Facillities, self.Terrain_cache))
		return TRUTH_VALUE_AGGREGATION(values)
	
	def evaluate_fitness(self, x, y):
		"""
		A fitness function to evaluate (x, y) coordinates.
		Each x, y pair corresponds to a facility.
		"""
		if len(x) != len(self.Facillities) or len(y) != len(self.Facillities):
			raise ValueError("Size of x and y must match number of facilities.")

		for idx, f in enumerate(self.Facillities):
			f.Polygon = [[x[idx], y[idx]]]

		values = [constraint.sat_eval(args, self.Facillities, self.Terrain_cache) for constraint, args in self.Constraints]
		
		return TRUTH_VALUE_AGGREGATION(values)
	

	def solve_with_z3(self):
		s = Solver()
		
		# Construct variable map		
		var_map = {}
		for f in self.Facillities:
			var_map[f] = [None, None]
			var_map[f][0] = Real('x_' + f.Id)
			var_map[f][1] = Real('y_' + f.Id)
			s.add(var_map[f][0] >= 0.0)
			s.add(var_map[f][1] >= 0.0)
			s.add(var_map[f][0] <= self.Map_scale[0])
			s.add(var_map[f][1] <= self.Map_scale[1])

		for constraint, args in self.Constraints:
			s.add(constraint.get_z3_constraint(args, self.Facillities, self.Terrain_cache, var_map))

		r = s.check()
		if r != sat:
			return None
			#raise Exception('Unsatisfiable or unable to solve (' + str(r) + ')')
		
		# Record result
		m = s.model()
		for f in self.Facillities:
			x = m.evaluate(var_map[f][0]).as_fraction()
			x_float = float(x.numerator) / float(x.denominator)
			y = m.evaluate(var_map[f][1]).as_fraction()
			y_float = float(y.numerator) / float(y.denominator)
			f.Polygon = [[x_float, y_float]]
			
		return s.model()
		
	def compute_sat_percentage(self):
		values = []
		for constraint, args in self.Constraints:
			values.append(constraint.sat_eval(args, self.Facillities, self.Terrain_cache))

		return sum(values) / float(len(values))

	def generate_facts(self, constraint_range = [], truth_th = 0.99):
		# Todo: Replace the code with a call to ConstraintType.PopulateAllInstantiations()
		def getArgSet(arg_type):
			if arg_type == SpatialObjectType.TERRAIN:
				return self.Terrain_tags
			elif arg_type == SpatialObjectType.FACILITY:
				return self.Facility_tags
			else:
				raise Exception("Invalid argument type for spatial relation")

		if len(constraint_range) <= 0:
			constraint_range = ConstraintType.constraint_library

		statements = []
		for ct_name in constraint_range:
			ct = ConstraintType.constraint_library[ct_name]
			arg_types = ct.arg_types[len(ct.built_in_args):]
			arg_sets = []
			if len(arg_types) == 0:
				arg_sets = [[]]
			elif len(arg_types) == 1:
				arg_sets = [[e] for e in getArgSet(arg_types[0])]
			else:
				arg_sets = product(*[getArgSet(at) for at in arg_types])
			# Todo: three arguments
			for arg_set in arg_sets:
				# Temporarily removing self relations
				if len(arg_set) == 2 and arg_set[0] == arg_set[1]:
					continue
				statements.append((ct, arg_set))

		#print('statement count:', statements)
		facts = []
		for constraint, args in statements:
			truth = constraint.sat_eval(args, self.Facillities, self.Terrain_cache)
			if truth >= truth_th:
				facts.append((constraint, args, random.sample(constraint.generate_utterance(args), 1)))

		return facts

	def get_terrain_tags_of_point(self, point):
		tags = set([])
		for obj in self.Terrain_objects:
			if obj.PreparedShapelyPolygon == None:
				obj.PreparedShapelyPolygon = shapely.prepared.prep(obj.ShapelyPolygon)
			if obj.PreparedShapelyPolygon.contains(Point(point)):
				tags = tags.union(set(obj.Tags))

		tags = [self.Terrain_tags.index(tag) for tag in tags]

		return tags

	def get_self_indicator_of_point(self, point, self_location):
		dist = Point(point).distance(Point(self_location))
		if dist <= self.Self_radius * self.Map_scale[0]:
			return 1
		else:
			return 0

	def get_facility_tags_of_point(self, point):
		tags = set([])
		for obj in self.Facillities:
			dist = Point(point).distance(Point(obj.Polygon[0]))
			if dist <= self.Facility_radius * self.Map_scale[0]:
				tags = tags.union(set(obj.Tags))

		tags = [self.Facility_tags.index(tag) for tag in tags]
		return list(tags)

	@staticmethod
	def load_from_json(iptJson, task_id):
		try:
			map_scale = iptJson['scale']
			facility_tags = iptJson['plotFacilityTags']
			terrain_tags = TERRAIN_TYPES + AUX_TERRAIN_TYPES
			terrain_objects, _ = LoadTerrainObjects(iptJson)
			facilities, _ = LoadObjectsToPlace(iptJson)
			constraints, constraintsText = LoadConstraints(iptJson)
		except Exception:
			print("Loading facility placement task from json failed.")
			raise

		newTask = FacilityPlacementTask(task_id, facility_tags, terrain_tags, 
			map_scale, facilities, terrain_objects, constraints, constraintsText)

		return newTask

if __name__ == '__main__':
	from FacilityPlacementTaskRenderer import FacilityPlacementTaskRenderer
	task = FacilityPlacementTask.load_from_json(json.load(open('tasksets/generated_tasks_10_terrain_10_constraints/task_1.json', 'r')), "task_1")
	print('sat value: ', task.evaluate())

	for f in task.Facillities:
		f.Polygon = [[0., 0.]]

	renderer = FacilityPlacementTaskRenderer(task)
	renderer.render_task(0)

