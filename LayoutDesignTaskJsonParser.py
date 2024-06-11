from SpatialObject import SpatialObject
from ConstraintType import *
from ConstraintTemplates import TERRAIN_TYPES
import sys

VISION_BLOCKING_TERRAIN_TYPES = ['MOUNTAIN', 'FOREST', 'HILLS', 'WOODED_HILLS']
VISION_BLOCKING_TAG = 'VISION_BLOCKING'

def LoadConstraints(iptJson):
	constraintField = 'requirements'
	constraintTypeField = 'constraint'
	#negatedField = 'negated'
	argumentField = 'arguments'
	utteranceField = 'utterance'

	constraints = []
	constraintText = []
	try:
		for jobj in iptJson[constraintField]:
			if jobj[constraintTypeField] not in ConstraintType.constraint_library:
				continue
			constraintType = ConstraintType.constraint_library[jobj[constraintTypeField]]
			constraint = (constraintType, jobj[argumentField])
			constraints.append(constraint)
			if utteranceField in jobj:
				text = jobj[utteranceField]
				constraintText.append((text, jobj[argumentField]))
	except Exception:
		raise Exception("Loading constraints failed.")

	return constraints, constraintText

def LoadObjectsToPlace(iptJson):
	objectsToPlaceField = 'plotFacilities'
	objectIdField = 'id'
	objectTagsField = 'tags'
	initialLocationField = 'initial-location'

	objectsToplace = []
	objectsToplaceMapping = {}
	try:
		for jobj in iptJson[objectsToPlaceField]:
			obj = SpatialObject(jobj[objectIdField], [[0., 0.]], jobj[objectTagsField])
			if initialLocationField in jobj:
				obj.Polygon.append(jobj[initialLocationField])
			objectsToplace.append(obj)
			objectsToplaceMapping[jobj[objectIdField]] = len(objectsToplace) - 1
	except Exception:
		raise Exception("Loading objects to place failed.")
	return objectsToplace, objectsToplaceMapping


def LoadTerrainObjects(iptJson):
	existingObjectField = 'zones'
	objectIdField = 'id'
	objectPolygonField = 'polygon'
	objectTagsField = 'tags'
	objectDisplayNameField = 'display_name'

	existingObjects = []
	existingObjectMapping = {}
	try:
		for jobj in iptJson[existingObjectField]:
			tags = list(set(jobj[objectTagsField]).intersection(set(TERRAIN_TYPES)))
			if len(set(tags).intersection(set(VISION_BLOCKING_TERRAIN_TYPES))) > 0:
				tags.append(VISION_BLOCKING_TAG)

			obj = SpatialObject(jobj[objectIdField], jobj[objectPolygonField], tags, jobj[objectDisplayNameField])

			existingObjects.append(obj)
			existingObjectMapping[jobj[objectIdField]] = len(existingObjects) - 1
	except Exception:
		raise Exception("Loading terrain objects failed.")
	
	return existingObjects, existingObjectMapping