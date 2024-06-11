import cv2
#from FacilityPlacementTask import FacilityPlacementTask
import numpy as np
from shapely.geometry import Point, Polygon
from ConstraintTemplates import TERRAIN_TYPES
from ColorScheme import *


class FacilityPlacementTaskRenderer():

	def __init__(self, fpTask, draw_scale = 3, font = cv2.FONT_HERSHEY_SIMPLEX):

		self.FpTask = fpTask
		self.Draw_scale = draw_scale
		self.Font = font

	def draw_single_terrain_polygon(self, img, polygon, tags):
		pts = np.array([[p[0] * self.Draw_scale + 50, p[1] * self.Draw_scale + 50] for p in polygon], np.int32)
		pts = pts.reshape((-1,1,2))
		tag = tags[0]
		if tag in TERRAIN_TYPES:
			idx = TERRAIN_TYPES.index(tag)
			img = cv2.fillPoly(img, [pts], TerrainColors[idx])
		else:
			img = cv2.polylines(img, [pts], True, (1, 1, 1), thickness=1)
		ctr = list(Polygon(polygon).centroid.coords)[0]	

		return img

	def render_obs(self, obs, waitKey):
		if obs is tuple:
			obs = obs[0]

		obs = obs.astype(np.uint8)

		cv2.namedWindow('vision obs', cv2.WINDOW_NORMAL)
		cv2.imshow('vision obs', obs)
		cv2.waitKey(waitKey)

	def render_task(self, waitKey):
		
		map_scale = self.FpTask.Map_scale
		draw_size_w = int(map_scale[0]) * self.Draw_scale + 100
		draw_size_h = int(map_scale[1]) * self.Draw_scale + 100
		img = np.zeros([draw_size_w, draw_size_h, 3], dtype=np.float32)

		for obj in self.FpTask.Terrain_objects:
			arr = np.array(obj.Polygon)
			if len(arr.shape) == 2:
				img = self.draw_single_terrain_polygon(img, obj.Polygon, obj.Tags)
			else:
				for poly in obj.Polygon:
					img = self.draw_single_terrain_polygon(img, poly, obj.Tags)
			#img = cv2.putText(img,','.join(obj.Tags),(int(ctr[0]) * self.Draw_scale + 200 - 50, 30 + int(ctr[1] + 200) * self.Draw_scale), self.Font, 2,(255,255,255),2,cv2.LINE_AA)

		for obj in self.FpTask.Facillities:
			pt = (int(obj.Polygon[0][0]) * self.Draw_scale + 50, int(obj.Polygon[0][1]) * self.Draw_scale + 50)
			img = cv2.circle(img, pt, radius=1 * self.Draw_scale, color=(0, 0, 255), thickness=3)
			img = cv2.putText(img, obj.Id,(pt[0] + 10, pt[1]), self.Font, 1,(0,0,255),2,cv2.LINE_AA)

		cv2.namedWindow('facility placement task', cv2.WINDOW_NORMAL)
		cv2.imshow('facility placement task', img)
		cv2.waitKey(waitKey)
		