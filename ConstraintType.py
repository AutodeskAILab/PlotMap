from SpatialObjectType import SpatialObjectType
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import shapely
from itertools import product

NEGATED_PREFIX = 'Not_'
UTTERANCE_FOLDER = 'utterances'

def GetConstraintStr(constraint):
	(ct, arg_set) = constraint
	argStr = ','.join(arg_set)
	return ct.name + '(' + argStr + ')'

def PopulateAllInstantiations(constraint_types, terrain_types, facilities):
	instantiations = []
	def getArgSet(arg_type):
			if arg_type == SpatialObjectType.TERRAIN:
				return terrain_types
			elif arg_type == SpatialObjectType.FACILITY:
				return facilities
			else:
				raise Exception("Invalid argument type for spatial relation")

	constraint_range = constraint_types

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

	return [GetConstraintStr(constraint) for constraint in statements]

class TerrainCache():

	def __init__(self, terrain_objects):
		self.terrain_objects = terrain_objects
		self.tag_terrain_map = {}
		self.tag_terrain_map_prepared = {}

	def get_combined_terrain_polygon(self, terrain_tag):
		if terrain_tag not in self.tag_terrain_map:
			list_of_terrain_polygons = [obj.ShapelyPolygon for obj in self.terrain_objects if terrain_tag in obj.Tags]
			combined_terrain_polygon = MultiPolygon([])
			for poly in list_of_terrain_polygons:
				combined_terrain_polygon = unary_union([combined_terrain_polygon, poly])
			self.tag_terrain_map[terrain_tag] = combined_terrain_polygon
			self.tag_terrain_map_prepared[terrain_tag] = shapely.prepared.prep(combined_terrain_polygon)

		return (self.tag_terrain_map[terrain_tag], self.tag_terrain_map_prepared[terrain_tag])

class ConstraintType():

	constraint_library = {}
	constraint_list = []

	def __init__(self, name, arg_types, eval_func, get_z3_constraint_func, utter_file, built_in_args = [], creating_negated_version = False):
		self.name = name
		self.arg_types = arg_types
		self.eval_func = eval_func
		self.get_z3_constraint_func = get_z3_constraint_func
		self.utter_file = UTTERANCE_FOLDER + '/' + utter_file
		self.built_in_args = built_in_args
		ConstraintType.constraint_library[name] = self
		ConstraintType.constraint_list.append(self)
		if creating_negated_version:
			# Creating a negated version
			negated_eval_func = lambda args: 1. - eval_func(args)
			ConstraintType.constraint_library[NEGATED_PREFIX + name] = ConstraintType(NEGATED_PREFIX + self.name, self.arg_types, negated_eval_func, False)

	def generate_utterance(self, args):

		args = self.built_in_args + list(args)
	
		assert(len(args) == len(self.arg_types))
		utterIpt = open(self.utter_file, 'r')
		lines = utterIpt.readlines()
		utterances = []
		for line in lines:
			utter = line
			for i in range(len(args)):
				utter = utter.replace('#' + str(i+1), args[i])
			utterances.append(utter.rstrip('\n'))

		return utterances

	def sat_eval(self, args, facilities, terrain_cache):

		args = self.built_in_args + list(args)

		assert(len(args) == len(self.arg_types))

		instantiated_args = []
		for i in range(len(args)):
			if self.arg_types[i] == SpatialObjectType.TERRAIN:
				instantiated_arg = terrain_cache.get_combined_terrain_polygon(args[i])
			elif self.arg_types[i] == SpatialObjectType.FACILITY:
				instantiated_arg = [obj for obj in facilities if args[i] in obj.Tags]
			else:
				raise ValueError
			instantiated_args.append(instantiated_arg)


		return self.eval_func(instantiated_args)
	
	def get_z3_constraint(self, args, facilities, terrain_cache, var_map):
		args = self.built_in_args + list(args)

		assert(len(args) == len(self.arg_types))

		instantiated_args = []
		for i in range(len(args)):
			if self.arg_types[i] == SpatialObjectType.TERRAIN:
				instantiated_arg = terrain_cache.get_combined_terrain_polygon(args[i])
			elif self.arg_types[i] == SpatialObjectType.FACILITY:
				instantiated_arg = [obj for obj in facilities if args[i] in obj.Tags]
			else:
				raise ValueError
			instantiated_args.append(instantiated_arg)


		return self.get_z3_constraint_func(instantiated_args, var_map)