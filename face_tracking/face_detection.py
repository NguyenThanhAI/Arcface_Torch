import numpy as np


class VehicleDetection(object):
    def __init__(self,bbox, detection_id):
        self.bbox = bbox
        self.detection_id = detection_id