from ConstraintType import ConstraintType
from SpatialObjectType import SpatialObjectType
from shapely import geometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiPoint, mapping
from shapely.ops import unary_union, nearest_points
import numpy as np
from z3 import *
import sys

TRUTH_VALUE_AGGREGATION = lambda vs: np.mean(vs)
EXIST_AGGREGATION = lambda vs: np.max(vs)
FORALL_AGGREGATION = lambda vs: np.max(vs)
DIST_NORMALIZING_FACTOR = 150.
COORD_NORMALIZING_FACTOR = 100.
CLOSE_TO_TH = 0.05
AWAY_FROM_TH = 0.2
FACILITY_SIZE = 0.15
VISIBLE_DIST = 0.20
LOCATION_SIZE_BUFFER = 0.1

TERRAIN_TYPES = ['OCEAN', 'LAKE', 'COAST', 'MOUNTAIN', 'FOREST', 'HILLS', 'WOODED_HILLS', 'PLAINS', 'DEEPOCEAN']
AUX_TERRAIN_TYPES = ['VISION_BLOCKING']

def GetLinesFromPolygon(poly):
	polygons = []
	lines = []
	if (isinstance(poly, MultiPolygon)):
		#print('it is a multipolygon')
		for p in poly:
			polygons.append(p)
	else:
		polygons.append(poly)

	for p in polygons:
		#print('poly:', p)
		xx, yy = p.exterior.coords.xy
		for i in range(len(xx) - 1):
			lines.append([[xx[i], yy[i]], [xx[i+1], yy[i+1]]])

	return lines

def GetSATConstraintsForVisibleFromSomeFacility(args, var_map):
	assert len(args) == 3

	terrain_polygon = args[0]
	lines = GetLinesFromPolygon(terrain_polygon[0])

	cst = False
	for f1 in args[1]:
		for f2 in args[2]:
			# See if the line segment formed by f1 and f2 has any intersection with any line segment on the polygon
			sub_cst = True
			for line in lines:
				slope_f = (var_map[f2][1] - var_map[f1][1]) / (var_map[f2][0] - var_map[f1][0])
				slope_tr = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])

				inter_x = (line[0][1] - var_map[f1][1] + slope_f * var_map[f1][0]- slope_tr * line[0][0]) / (slope_f - slope_tr)
				#inter_y = var_map[f1][1] + slope_f * (inter_x - var_map[f1][0])
				
				# No intersection either because the two lines are parallel, or the intersection between two lines are not on either lines
				no_intersect = Or(
					slope_f == slope_tr, 
					Or(And(inter_x >= var_map[f2][0], inter_x >= var_map[f1][0]),
					And(inter_x <= var_map[f1][0], inter_x <= var_map[f2][0])),
					And(var_map[f2][0] == var_map[f1][0], Or(And(var_map[f2][0] >= line[0][0], var_map[f2][0] >= line[1][0]), And(var_map[f2][0] <= line[1][0], var_map[f2][0] <= line[0][0])))
					)
				sub_cst = And(sub_cst, no_intersect)
			cst = Or(cst, sub_cst)

	return cst

def EvaluateVisibleFromSomeFacilitiy(args):

	assert len(args) == 3

	terrain_polygon = args[0]

	results = []
	for f1 in args[1]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[2]:
			pt2 = Point(f2.Polygon[0])
			line = LineString([[pt1.x, pt1.y], [pt2.x, pt2.y]])
			# Check if the line intersects with any vision blocking terrain types
			if line.length / DIST_NORMALIZING_FACTOR > VISIBLE_DIST or terrain_polygon[1].intersects(line):
				sub_results.append(0.0)
			else:
				sub_results.append(1.0)
			results.append(EXIST_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

VisibleFromSomeFacilitiyPred = ConstraintType("VisibleFromSome",
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateVisibleFromSomeFacilitiy,
				       			   GetSATConstraintsForVisibleFromSomeFacility,
	                               "utterances_VisibleFromSome.txt",
	                               built_in_args = ['VISION_BLOCKING'])


def GetSATConstraintsForToTheEastOfSomeFacility(args, var_map):
	assert len(args) == 2

	cst = True
	for f1 in args[0]:
		for f2 in args[1]:
			cst = And(cst, var_map[f1][0] > var_map[f2][0] + LOCATION_SIZE_BUFFER * DIST_NORMALIZING_FACTOR)

	return cst

def EvaluateToTheEastOfSomeFacility(args):

	assert len(args) == 2

	results = []
	for f1 in args[0]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[1]:
			pt2 = Point(f2.Polygon[0])
			if pt1.x > pt2.x:
				sub_results.append(1.0)
			else:
				sub_results.append(0.0)
			results.append(EXIST_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

ToTheEastOfSomeFacilityPred = ConstraintType("ToTheEastOfSome",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateToTheEastOfSomeFacility,
				       			   GetSATConstraintsForToTheEastOfSomeFacility,
	                               "utterances_ToTheEastOfSome.txt",
	                               built_in_args = [])

def GetSATConstraintsForToTheWestOfSomeFacility(args, var_map):
	assert len(args) == 2

	cst = True
	for f1 in args[0]:
		for f2 in args[1]:
			cst = And(cst, var_map[f1][0] + LOCATION_SIZE_BUFFER * DIST_NORMALIZING_FACTOR < var_map[f2][0])

	return cst


def EvaluateToTheWestOfSomeFacility(args):

	assert len(args) == 2

	results = []
	for f1 in args[0]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[1]:
			pt2 = Point(f2.Polygon[0])
			if pt1.x < pt2.x:
				sub_results.append(1.0)
			else:
				sub_results.append(0.0)
			results.append(EXIST_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

ToTheWestOfSomeFacilityPred = ConstraintType("ToTheWestOfSome",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateToTheWestOfSomeFacility,
				       			   GetSATConstraintsForToTheWestOfSomeFacility,
	                               "utterances_ToTheWestOfSome.txt",
	                               built_in_args = [])

def GetSATConstraintsForToTheSouthOfSomeFacility(args, var_map):
	assert len(args) == 2

	cst = True
	for f1 in args[0]:
		for f2 in args[1]:
			cst = And(cst, var_map[f1][1] > var_map[f2][1]  + LOCATION_SIZE_BUFFER * DIST_NORMALIZING_FACTOR)

	return cst

def EvaluateToTheSouthOfSomeFacility(args):

	assert len(args) == 2

	results = []
	for f1 in args[0]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[1]:
			pt2 = Point(f2.Polygon[0])
			if pt1.y > pt2.y:
				sub_results.append(1.0)
			else:
				sub_results.append(0.0)
			results.append(EXIST_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

ToTheSouthOfSomeFacilityPred = ConstraintType("ToTheSouthOfSome",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateToTheSouthOfSomeFacility,
				       			   GetSATConstraintsForToTheSouthOfSomeFacility,
	                               "utterances_ToTheSouthOfSome.txt",
	                               built_in_args = [])


def GetSATConstraintsForToTheNorthOfSomeFacility(args, var_map):
	assert len(args) == 2

	cst = True
	for f1 in args[0]:
		for f2 in args[1]:
			cst = And(cst, var_map[f1][1] + LOCATION_SIZE_BUFFER * DIST_NORMALIZING_FACTOR < var_map[f2][1])

	return cst

def EvaluateToTheNorthOfSomeFacility(args):

	assert len(args) == 2

	results = []
	for f1 in args[0]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[1]:
			pt2 = Point(f2.Polygon[0])
			if pt1.y < pt2.y:
				sub_results.append(1.0)
			else:
				sub_results.append(0.0)
			results.append(EXIST_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

ToTheNorthOfSomeFacilityPred = ConstraintType("ToTheNorthOfSome",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateToTheNorthOfSomeFacility,
				       			   GetSATConstraintsForToTheNorthOfSomeFacility,
	                               "utterances_ToTheNorthOfSome.txt",
	                               built_in_args = [])

def GetSATConstraintsForOnEast(args, var_map):
	assert len(args) == 1

	cst = True
	for f in args[0]:
		cst = And(cst, var_map[f][0] / COORD_NORMALIZING_FACTOR > 0.5 + LOCATION_SIZE_BUFFER)

	return cst

def EvaluateOnEast(args):

	assert len(args) == 1

	results = []
	for f in args[0]:
		pt = Point(f.Polygon[0])

		if pt.x / COORD_NORMALIZING_FACTOR > 0.5:
			results.append(1.0)
		else: 
			results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

OnEastPred = ConstraintType("OnEast",
	                               [SpatialObjectType.FACILITY],
	                               EvaluateOnEast,
				       			   GetSATConstraintsForOnEast,
	                               "utterances_OnEast.txt",
	                               built_in_args = [])

def GetSATConstraintsForOnWest(args, var_map):
	assert len(args) == 1

	cst = True
	for f in args[0]:
		cst = And(cst, var_map[f][0] / COORD_NORMALIZING_FACTOR + LOCATION_SIZE_BUFFER < 0.5 )

	return cst

def EvaluateOnWest(args):

	assert len(args) == 1

	results = []
	for f in args[0]:
		pt = Point(f.Polygon[0])

		if pt.x / COORD_NORMALIZING_FACTOR < 0.5:
			results.append(1.0)
		else: 
			results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

OnWestPred = ConstraintType("OnWest",
	                               [SpatialObjectType.FACILITY],
	                               EvaluateOnWest,
				       			   GetSATConstraintsForOnWest,
	                               "utterances_OnWest.txt",
	                               built_in_args = [])


def GetSATConstraintsForOnSouth(args, var_map):
	assert len(args) == 1

	cst = True
	for f in args[0]:
		cst = And(cst, var_map[f][1] / COORD_NORMALIZING_FACTOR > 0.5 + LOCATION_SIZE_BUFFER)

	return cst

def EvaluateOnSouth(args):

	assert len(args) == 1

	results = []
	for f in args[0]:
		pt = Point(f.Polygon[0])

		if pt.y / COORD_NORMALIZING_FACTOR > 0.5:
			results.append(1.0)
		else: 
			results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

OnSouthPred = ConstraintType("OnSouth",
	                               [SpatialObjectType.FACILITY],
	                               EvaluateOnSouth,
				       			   GetSATConstraintsForOnSouth,
	                               "utterances_OnSouth.txt",
	                               built_in_args = [])

def GetSATConstraintsForOnNorth(args, var_map):
	assert len(args) == 1

	cst = True
	for f in args[0]:
		cst = And(cst, var_map[f][1] / COORD_NORMALIZING_FACTOR + LOCATION_SIZE_BUFFER < 0.5)

	return cst

def EvaluateOnNorth(args):

	assert len(args) == 1

	results = []
	for f in args[0]:
		pt = Point(f.Polygon[0])

		if pt.y / COORD_NORMALIZING_FACTOR < 0.5:
			results.append(1.0)
		else: 
			results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

OnNorthPred = ConstraintType("OnNorth",
	                               [SpatialObjectType.FACILITY],
	                               EvaluateOnNorth,
				       			   GetSATConstraintsForOnNorth,
	                               "utterances_OnNorth.txt",
	                               built_in_args = [])

def GetSATConstraintsForInBetweenTwoFacilities(args, var_map):
	
	assert len(args) == 3

	cst = False
	for f in args[0]:
		for f1 in args[1]:
			for f2 in args[2]:
				slope = (var_map[f2][1] - var_map[f1][1]) / (var_map[f2][0] - var_map[f1][0])
				intercept = var_map[f1][1] - slope * var_map[f1][0]
				cst = Or(cst, 
	     				And(
							var_map[f][1] == var_map[f][0] * slope + intercept,
							And(
								Or(And(var_map[f][0] <= var_map[f1][0], var_map[f][0] >= var_map[f2][0]),
	   							   And(var_map[f][0] <= var_map[f2][0], var_map[f][0] >= var_map[f1][0])),
								Or(And(var_map[f][1] <= var_map[f1][1], var_map[f][1] >= var_map[f2][1]),
	   							   And(var_map[f][1] <= var_map[f2][1], var_map[f][1] >= var_map[f1][1]))      
							)
						))

	return cst		

def EvaluateInBetweenTwoFacilities(args):

	assert len(args) == 3

	results = []
	for f in args[0]:
		buffer_pt = Point(f.Polygon[0]).buffer(FACILITY_SIZE * DIST_NORMALIZING_FACTOR)
		for f1 in args[1]:
			pt1 = Point(f1.Polygon[0])
			for f2 in args[2]:
				pt2 = Point(f2.Polygon[0])
				line = LineString([pt1, pt2])
				intersection = buffer_pt.boundary.intersection(line)
				#print('intersection:', intersection)
				if isinstance(intersection, LineString) or isinstance(intersection, MultiPoint):
					if isinstance(intersection, MultiPoint):
						intersection = LineString(intersection)
					fuzzy_value = intersection.length / (FACILITY_SIZE * DIST_NORMALIZING_FACTOR * 2)
					results.append(fuzzy_value)
				else:
					results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

InBetweenTwoFacilitiesPred = ConstraintType("InBetween",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateInBetweenTwoFacilities,
				       			   GetSATConstraintsForInBetweenTwoFacilities,
	                               "utterances_InBetweenTwoFacilities.txt",
	                               built_in_args = [])


def GetSATConstraintsForAwayFromAllFacility(args, var_map):

	assert len(args) == 2

	cst = True
	for f1 in args[0]:
		for f2 in args[1]:
			cst = And(cst, ((var_map[f1][0] - var_map[f2][0])**2 + (var_map[f1][1] - var_map[f2][1])**2) / DIST_NORMALIZING_FACTOR**2 >= AWAY_FROM_TH**2)
	return cst

def EvaluateAwayFromAllFacility(args):

	assert len(args) == 2

	results = []
	for f1 in args[0]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[1]:
			pt2 = Point(f2.Polygon[0])
			normalized_distance = (pt1.distance(pt2)) / DIST_NORMALIZING_FACTOR
			if normalized_distance >= AWAY_FROM_TH:
				sub_results.append(1.0)
			else:
				fuzzy_value = normalized_distance
				sub_results.append(fuzzy_value)
			results.append(FORALL_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

AwayFromAllFacilityPred = ConstraintType("AwayFromAll",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateAwayFromAllFacility,
				       			   GetSATConstraintsForAwayFromAllFacility,
	                               "utterances_AwayFromAllFacility.txt",
	                               built_in_args = [])


def GetSATConstraintsForCloseToSomeFacility(args, var_map):

	assert len(args) == 2

	cst = False
	for f1 in args[0]:
		for f2 in args[1]:
			cst = Or(cst, ((var_map[f1][0] - var_map[f2][0])**2 + (var_map[f1][1] - var_map[f2][1])**2) / DIST_NORMALIZING_FACTOR**2 <= CLOSE_TO_TH**2)
	return cst

def EvaluateCloseToSomeFacility(args):

	assert len(args) == 2

	results = []
	for f1 in args[0]:
		pt1 = Point(f1.Polygon[0])
		sub_results = []
		for f2 in args[1]:
			pt2 = Point(f2.Polygon[0])
			normalized_distance = (pt1.distance(pt2)) / DIST_NORMALIZING_FACTOR
			if normalized_distance <= CLOSE_TO_TH:
				sub_results.append(1.0)
			else:
				fuzzy_value = 1.0 - normalized_distance
				sub_results.append(fuzzy_value)
			results.append(EXIST_AGGREGATION(sub_results))

	return TRUTH_VALUE_AGGREGATION(results)

CloseToSomeFacilityPred = ConstraintType("CloseToSome",
	                               [SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateCloseToSomeFacility,
				        		   GetSATConstraintsForCloseToSomeFacility,
	                               "utterances_CloseToSomeFacility.txt",
	                               built_in_args = [])

def GetSATConstraintsForToTheEastOfTerrain(args, var_map):

	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the rightmost point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	x_largest = max([coord[0] for coord in merged_coords])
	if isinstance(x_largest, list) or isinstance(x_largest, tuple):
		x_largest = max(x_largest)

	cst = True
	for f1 in args[1]:
		# Ignore the case where the facility is inside the terrain
		cst = And(cst, var_map[f1][0] > x_largest)

	return cst

def EvaluateToTheEastOfTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the rightmost point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	x_largest = max([coord[0] for coord in merged_coords])
	if isinstance(x_largest, list) or isinstance(x_largest, tuple):
		x_largest = max(x_largest)

	results = []
	for f1 in args[1]:
		pt = Point(f1.Polygon[0])
		# The facility can't be inside the terrain
		if terrain_polygon[1].contains(pt):
			results.append(0.0)
		else:
			if pt.x > x_largest:
				results.append(1.0)
			else:
				results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

ToTheEastOfTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	ToTheEastOfTerrainPreds[terrain_type] = ConstraintType("ToTheEastOf" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateToTheEastOfTerrain,
				                   GetSATConstraintsForToTheEastOfTerrain,
	                               "utterances_ToTheEastOf" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])

def GetSATConstraintsForToTheWestOfTerrain(args, var_map):

	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the leftmost point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	x_smallest = min([coord[0] for coord in merged_coords])
	if isinstance(x_smallest, list) or isinstance(x_smallest, tuple):
		x_smallest = min(x_smallest)

	cst = True
	for f1 in args[1]:
		# Ignore the case where the facility is inside the terrain
		cst = And(cst, var_map[f1][0] < x_smallest)

	return cst

def EvaluateToTheWestOfTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the leftmost point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	x_smallest = min([coord[0] for coord in merged_coords])
	if isinstance(x_smallest, list) or isinstance(x_smallest, tuple):
		x_smallest = min(x_smallest)

	results = []
	for f1 in args[1]:
		pt = Point(f1.Polygon[0])
		# The facility can't be inside the terrain
		if terrain_polygon[1].contains(pt):
			results.append(0.0)
		else:
			if pt.x < x_smallest:
				results.append(1.0)
			else:
				results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

ToTheWestOfTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	ToTheWestOfTerrainPreds[terrain_type] = ConstraintType("ToTheWestOf" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateToTheWestOfTerrain,
				       			   GetSATConstraintsForToTheWestOfTerrain,
	                               "utterances_ToTheWestOf" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])

def GetSATConstraintsForToTheNorthOfTerrain(args, var_map):
	
	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the lowest point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	y_smallest = min([coord[1] for coord in merged_coords])
	if isinstance(y_smallest, list) or isinstance(y_smallest, tuple):
		y_smallest = min(y_smallest)

	cst = True
	for f1 in args[1]:
		# Ignore the case where the facility is inside the terrain
		cst = And(cst, var_map[f1][1] < y_smallest)

	return cst

def EvaluateToTheNorthOfTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the lowest point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	y_smallest = min([coord[1] for coord in merged_coords])
	if isinstance(y_smallest, list) or isinstance(y_smallest, tuple):
		y_smallest = min(y_smallest)


	results = []
	for f1 in args[1]:
		pt = Point(f1.Polygon[0])
		# The facility can't be inside the terrain
		if terrain_polygon[1].contains(pt):
			results.append(0.0)
		else:
			if pt.y < y_smallest:
				results.append(1.0)
			else:
				results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

ToTheNorthOfTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	ToTheNorthOfTerrainPreds[terrain_type] = ConstraintType("ToTheNorthOf" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateToTheNorthOfTerrain,
				       			   GetSATConstraintsForToTheNorthOfTerrain,
	                               "utterances_ToTheNorthOf" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])

def GetSATConstraintsForToTheSouthOfTerrain(args, var_map):
	
	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the lowest point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	y_largest = max([coord[1] for coord in merged_coords])
	if isinstance(y_largest, list) or isinstance(y_largest, tuple):
		y_largest = max(y_largest)

	cst = True
	for f1 in args[1]:
		# Ignore the case where the facility is inside the terrain
		cst = And(cst, var_map[f1][1] > y_largest)

	return cst

def EvaluateToTheSouthOfTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]

	# Get the lowest point on the polygon 
	mapped_polygon = mapping(terrain_polygon[0])
	merged_coords = []

	for poly in mapped_polygon['coordinates']:
		merged_coords += poly

	y_largest = max([coord[1] for coord in merged_coords])
	if isinstance(y_largest, list) or isinstance(y_largest, tuple):
		y_largest = max(y_largest)

	results = []
	for f1 in args[1]:
		pt = Point(f1.Polygon[0])
		# The facility can't be inside the terrain
		if terrain_polygon[1].contains(pt):
			results.append(0.0)
		else:
			if pt.y > y_largest:
				results.append(1.0)
			else:
				results.append(0.0)

	return TRUTH_VALUE_AGGREGATION(results)

ToTheSouthOfTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	ToTheSouthOfTerrainPreds[terrain_type] = ConstraintType("ToTheSouthOf" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateToTheSouthOfTerrain,
				       			   GetSATConstraintsForToTheSouthOfTerrain,
	                               "utterances_ToTheSouthOf" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])


def GetSATConstraintsForCloseToTerrain(args, var_map):
	return False
	assert len(args) == 2

	terrain_polygon = args[0]

	cst = False
	for f in args[1]:
		# Should be comparing with the closest point on the polygon from the facility, instead of the centeroid, but not sure how to express this in Z3 constraints
		centroid =  list(terrain_polygon.centroid.coords)

		lines = GetNonhorizontalLinesFromPolygon(terrain_polygon[0])
		num_intersection = sum([If(
			And(
				Or(And(var_map[f][1] >= line[0][1], var_map[f][1] <= line[1][1]), 
       			   And(var_map[f][1] <= line[0][1], var_map[f][1] >= line[1][1])),
				And(var_map[f][0] <= line[0][0], var_map[f][0] <= line[1][0])  
			),
			1, 0) for line in lines])

		cst = Or(cst,
	   			And(
					((var_map[f][0] - centroid[0])**2 + (var_map[f][1] - centroid[1])**2) / DIST_NORMALIZING_FACTOR**2 <= CLOSE_TO_TH**2),
					 num_intersection % 2 == 1
					)
	return cst

def EvaluateCloseToTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]
	
	results = []
	for f1 in args[1]:
		if terrain_polygon[1].contains(Point(f1.Polygon[0])):
			results.append(1.0)
		else:
			normalized_distance = (terrain_polygon[0].distance(Point(f1.Polygon[0]))) / DIST_NORMALIZING_FACTOR
			if normalized_distance <= CLOSE_TO_TH:
				results.append(1.0)
			else:
				fuzzy_value = 1.0 - normalized_distance
				results.append(fuzzy_value)

	return TRUTH_VALUE_AGGREGATION(results)

CloseToTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	CloseToTerrainPreds[terrain_type] = ConstraintType("CloseTo" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateCloseToTerrain,
				       			   GetSATConstraintsForCloseToTerrain,
	                               "utterances_CloseTo" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])


def GetSATConstraintsForAwayFromTerrain(args, var_map):
	return False
	assert len(args) == 2

	terrain_polygon = args[0]

	cst = True
	for f in args[1]:
		# Should be comparing with the closest point on the polygon from the facility, instead of the centeroid, but not sure how to express this in Z3 constraints
		centroid =  list(terrain_polygon.centroid.coords)
		lines = GetNonhorizontalLinesFromPolygon(terrain_polygon[0])
		num_intersection = sum([If(
			And(
				Or(And(var_map[f][1] >= line[0][1], var_map[f][1] <= line[1][1]), 
       			   And(var_map[f][1] <= line[0][1], var_map[f][1] >= line[1][1])),
				And(var_map[f][0] <= line[0][0], var_map[f][0] <= line[1][0])  
			),
			1, 0) for line in lines])
		cst = And(cst, ((var_map[f][0] - centroid[0])**2 + (var_map[f][1] - centroid[1])**2) / DIST_NORMALIZING_FACTOR**2 >= AWAY_FROM_TH**2)
		cst = And(cst, num_intersection % 2 == 0)

	return cst

def EvaluateAwayFromTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]
	
	results = []
	for f1 in args[1]:
		
		normalized_distance = (terrain_polygon[0].distance(Point(f1.Polygon[0]))) / DIST_NORMALIZING_FACTOR
		if normalized_distance >= AWAY_FROM_TH:
			results.append(1.0)
		elif terrain_polygon[1].contains(Point(f1.Polygon[0])):
			results.append(0.0)
		else:
			fuzzy_value = normalized_distance
			results.append(fuzzy_value)

	return TRUTH_VALUE_AGGREGATION(results)

AwayFromTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	AwayFromTerrainPreds[terrain_type] = ConstraintType("AwayFrom" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateAwayFromTerrain,
				       			   GetSATConstraintsForAwayFromTerrain,
	                               "utterances_AwayFrom" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])
def GetNonhorizontalLinesFromPolygon(poly):
	polygons = []
	lines = []
	if (isinstance(poly, MultiPolygon)):
		#print('it is a multipolygon')
		for p in poly:
			polygons.append(p)
	else:
		polygons.append(poly)

	for p in polygons:
		#print('poly:', p)
		xx, yy = p.exterior.coords.xy
		for i in range(len(xx) - 1):
			# Skip horizontal lines
			if yy[i] == yy[i+1]:
				continue
			lines.append([[xx[i], yy[i]], [xx[i+1], yy[i+1]]])

	return lines


def GetSATConstraintsForInsideTerrain(args, var_map):
	assert len(args) == 2

	# Unprepared terrain polygon
	terrain_polygon = args[0]

	cst = True
	for f in args[1]:
		# Count number of lines on the polygon that has intersection with the point
		lines = GetNonhorizontalLinesFromPolygon(terrain_polygon[0])
		num_intersection = sum([If(
			And(
				Or(And(var_map[f][1] >= line[0][1], var_map[f][1] <= line[1][1]), 
       			   And(var_map[f][1] <= line[0][1], var_map[f][1] >= line[1][1])),
				And(var_map[f][0] <= line[0][0], var_map[f][0] <= line[1][0])  
			),
			1, 0) for line in lines])
		cst = And(cst, num_intersection % 2 == 1)
	
	return cst

def EvaluateInsideTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]
	# Every facility in args[1] has to be inside the specified terrain type
	results = []
	for f1 in args[1]:
		if terrain_polygon[1].contains(Point(f1.Polygon[0])):
			results.append(1.0)
		else:
			fuzzy_value = 1.0 - (terrain_polygon[0].distance(Point(f1.Polygon[0]))) / DIST_NORMALIZING_FACTOR
			results.append(fuzzy_value)

	return TRUTH_VALUE_AGGREGATION(results)

InsideTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	InsideTerrainPreds[terrain_type] = ConstraintType("Inside" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateInsideTerrain,
				       			   GetSATConstraintsForInsideTerrain,
	                               "utterances_Inside" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])

def GetSATConstraintsForOutsideTerrain(args, var_map):
	assert len(args) == 2

	# Unprepared terrain polygon
	terrain_polygon = args[0]
	#print('terrain polygon:', terrain_polygon)

	cst = True
	for f in args[1]:
		# Count number of lines on the polygon that has intersection with the horizontal line starting from the point
		lines = GetLinesFromPolygon(terrain_polygon[0])
		num_intersection = sum([If(
			And(
				Or(And(var_map[f][1] >= line[0][1], var_map[f][1] <= line[1][1]), 
       			   And(var_map[f][1] <= line[0][1], var_map[f][1] >= line[1][1])),
				And(var_map[f][0] <= line[0][0], var_map[f][0] <= line[1][0])  
			),
			1, 0) for line in lines])
		cst = And(cst, num_intersection % 2 == 0)
	
	return cst


def EvaluateOutsideTerrain(args):

	assert len(args) == 2

	terrain_polygon = args[0]
	# Every facility in args[1] has to be outside the specified terrain type
	results = []
	for f1 in args[1]:
		if terrain_polygon[1].contains(Point(f1.Polygon[0])) == False:
			results.append(1.0)
		else:
			fuzzy_value = 0.0
			if terrain_polygon[0].geom_type == 'MultiPolygon':
				vals = []
				for poly in terrain_polygon[0]:
					vals.append(poly.exterior.distance(Point(f1.Polygon[0])))
				fuzzy_value = 1.0 - min(vals) / DIST_NORMALIZING_FACTOR
			else:
				fuzzy_value = 1.0 - terrain_polygon[0].exterior.distance(Point(f1.Polygon[0])) / DIST_NORMALIZING_FACTOR
			results.append(fuzzy_value)

	return TRUTH_VALUE_AGGREGATION(results)

OutsideTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	OutsideTerrainPreds[terrain_type] = ConstraintType("Outside" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY],
	                               EvaluateOutsideTerrain,
				       			   GetSATConstraintsForOutsideTerrain,
	                               "utterances_Outside" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])

def GetSATConstraintsForAcrossTerrainTypeFrom(args, var_map):
	assert len(args) == 3

	terrain_polygon = args[0]
	lines = GetLinesFromPolygon(terrain_polygon[0])

	cst = True
	for f1 in args[1]:
		for f2 in args[2]:
			# See if the line segment formed by f1 and f2 has any intersection with any line segment on the polygon
			sub_cst = True
			for line in lines:
				slope_f = (var_map[f2][1] - var_map[f1][1]) / (var_map[f2][0] - var_map[f1][0])
				if (line[1][0] - line[0][0]) == 0:
					slope_tr = 9999.0
				else:
					slope_tr = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])

				inter_x = (line[0][1] - var_map[f1][1] + slope_f * var_map[f1][0]- slope_tr * line[0][0]) / (slope_f - slope_tr)
				inter_y = var_map[f1][1] + slope_f * (inter_x - var_map[f1][0])
				
				# The two lines have intersection when their slopes are different and the intersection point is in between the two points of the line segment
				has_intersect = And(
					slope_f != slope_tr, 
					Or(And(inter_x <= var_map[f2][0], inter_x >= var_map[f1][0]),
					And(inter_x <= var_map[f1][0], inter_x >= var_map[f2][0])),
					And(var_map[f2][0] == var_map[f1][0], Or(And(var_map[f2][0] <= line[0][0], var_map[f2][0] >= line[1][0]), And(var_map[f2][0] <= line[1][0], var_map[f2][0] >= line[0][0])))
					)
				sub_cst = Or(sub_cst, has_intersect)
			cst = And(cst, sub_cst)

	return cst

def EvaluateAcrossTerrainTypeFrom(args):

	assert len(args) == 3

	terrain_polygon = args[0]
	# Every facility in args[1] has to have at least one facility in args[2] that is across the terrain type
	results = []
	for f1 in args[1]:
		result = 0.0
		for f2 in args[2]:
			line = LineString([Point(f1.Polygon[0]), Point(f2.Polygon[0])])
			if terrain_polygon[1].intersects(line):
			#if tmp.geom_type == 'LineString' and len(tmp.coords) > 0:
				result = 1.0
		results.append(result)

	return TRUTH_VALUE_AGGREGATION(results)


AcrossTerrainPreds = {}
for terrain_type in TERRAIN_TYPES:
	AcrossTerrainPreds[terrain_type] = ConstraintType("Across" + terrain_type,
	                               [SpatialObjectType.TERRAIN, SpatialObjectType.FACILITY, SpatialObjectType.FACILITY],
	                               EvaluateAcrossTerrainTypeFrom,
				       				GetSATConstraintsForAcrossTerrainTypeFrom,
	                               "utterances_Across" + terrain_type + ".txt",
	                               built_in_args = [terrain_type])