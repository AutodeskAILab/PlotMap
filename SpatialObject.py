from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import numpy as np
import shapely

SCALE = 100.0
#inf = 10e15

def constructShapelyPolygon(vertices):
	array = np.array(vertices)
	if len(array.shape) == 2:
		if len(vertices) < 3:
			return None
		else:
			return Polygon(vertices)
	else:
		list_of_polygons = [Polygon(vs) for vs in vertices]
		combined_polygon = MultiPolygon([])
		for poly in list_of_polygons:
			combined_polygon = unary_union([combined_polygon, poly])
		#print('combined_polygon:', combined_polygon)
		return combined_polygon

def ConvertSingleCoord(coord):
	newCoord = [coord[0], coord[1]]
	if newCoord[0] <= 0.0:
		newCoord[0] = -SCALE
	if newCoord[1] <= 0.0:
		newCoord[1] = -SCALE
	if newCoord[0] >= SCALE:
		newCoord[0] += SCALE
	if newCoord[1] >= SCALE:
		newCoord[1] += SCALE
	return newCoord

def GetBufferedPolygon(polygon):
	res = []

	if len(polygon) <= 0:
		return polygon
	elif len(polygon[0]) == 2:
		for coord in polygon:
			res.append(ConvertSingleCoord(coord))
	else:
		for poly in polygon:
			res.append(GetBufferedPolygon(poly))
	return res

class SpatialObject():
	def __init__(self, id, polygon, tags, displayName = None):
		self.Id = id
		self.Polygon = GetBufferedPolygon(polygon)
		self.Tags = tags
		self.DisplayName = displayName
		if self.DisplayName is None:
			self.DisplayName = self.Id
		self.ShapelyPolygon = constructShapelyPolygon(self.Polygon)
		self.PreparedShapelyPolygon = None

	def GetBBox(self):
		minX = min([pt[0] for pt in self.Polygon])
		maxX = max([pt[0] for pt in self.Polygon])
		minY = min([pt[1] for pt in self.Polygon])
		maxY = max([pt[1] for pt in self.Polygon])
		
		return [minX, minY, maxX, maxY]

	def GetCenterPoint(self):
		x_mean = [pt[0] for pt in self.Polygon]
		x_mean = sum(x_mean) / len(x_mean)
		y_mean = [pt[1] for pt in self.Polygon]
		y_mean = sum(y_mean) / len(y_mean)
		
		return [x_mean, y_mean]

	def GetBBoxCenterPoint(self):
		bbox = self.GetBBox()
		return [(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.]
		
	def Copy(self):
		newPolygon = [pt.copy() for pt in self.Polygon]
		newInstance = SpatialObject(self.Id, newPolygon, self.Tags.copy())
		
		return newInstance
		
	def __str__(self):
		return 'spatial object: ' + self.Id + '\n polygon: ' + str(self.Polygon) + '\n tags: ' + str(self.Tags) + '\n'
		
	def __repr__(self):
		return str(self)