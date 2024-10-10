# import dependencies
import torch
from ultralytics import YOLO
import os
import numpy
import cv2
import sys
from typing import List, Dict
sys.path.append('..')
sys.path.append('././.')
device = torch.device('cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MODEL_DETECT_OBJ    = YOLO('yolov8n.pt')
MODEL_DETECT_OBJ.to(device) 

DICT_LABEL_YOLOV8 = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}



def extract_boxes_yolovn(model: YOLO, image: numpy.ndarray):#->List[List[float]]:
    """
        Detect bounding boxes using yolo model

        Args:
            model (YOLO): 
                the yolo model we will use
                
            image (numpy.ndarray): 
                the image we want to extract objects

        Returns:
            bounding_boxes (List[List[float]]): 
                list of tuple for bounding boxes
            
            label_boxes (List[str]): 
                list of label for each detected objects
                
        Raise:
        ------
            - if model is not an ultralytics.YOLO object
    """
    
    if not isinstance(model, YOLO):
        raise TypeError(f"wrong type object given, expected YOLO found : {type(model).__name__}")
    
    bounding_boxes = []
    label_boxes    = []
    results = model.predict(image, verbose=False)
    for res in results:
        bounding_boxes.append(res.boxes.xyxy)
        label_boxes.append(res.boxes.cls.tolist())  # .cls contains the labels of the boxes
                
    return bounding_boxes[0].tolist(), label_boxes[0]

def bounding_yolovn(bounding_box: List[float], image: numpy.ndarray):
    """
        Given a bounding boxe and the corresponding image
        return the image center on the bounding boxe
        
        Args:
            bounding_box (List[float]):
                the bounding boxe
                
            image (numpy.ndarray):
                the image we want 
                
        Returns:
            the image centered 
            
        Raise:
        -----
            - if no objects in bounding box
            - if bounding box is not a list
    
    """
    if len(bounding_box) == 0:
        raise TypeError(f"no bounding box found: {bounding_box}")
    
    if not isinstance(bounding_box, list):
        raise TypeError(f"wrong type object given, expected list found : {type(bounding_box).__name__}")
    
    x_min, y_min, x_max, y_max =  bounding_box
    centered_image = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    return centered_image



def detect_object_dataset(path_img: list, model: str= MODEL_DETECT_OBJ):
    """Detect objects on images

        Args:
            path_img (list): 
                list of path of images we want to detect objects on
                
            model (str): 
                the model to detect object (ex : yolov8)
                
        Returns:
            detected_objects (list):
                list of numpy object detected
            
            label_img (Dict):
                dict with the list of label from which images the objects comes from
                
        Example:
        --------
            >>> detect_object_dataset(path_img = [path/to/img1.png, path/to/img2.jpeg], model = YOLOV8)
            >>> returns : 
            
    """
    
    origin_img       = []
    detected_objects = []
    label_img        = []
    
    
    for img in path_img:
        img_cv2 = cv2.imread(img)
        boxes, labels_boxes = extract_boxes_yolovn(model, img_cv2)
        for bounding_objects, label_object in zip(boxes, labels_boxes):  # if no detected object, image not considerated
            detected_objects.append(bounding_yolovn(bounding_objects, img_cv2))
            origin_img.append(img)
            label_img.append(DICT_LABEL_YOLOV8[label_object])
            
    return detected_objects, {"images_path": origin_img, "image_label" : label_img}

def detect_object_query(img_cv2: numpy.ndarray, model: str= MODEL_DETECT_OBJ):
    """Detect objects on images

        Args:
            img_cv2 (numpy.ndarray): 
                the cv2 query image
                
            model (str): 
                the model to detect object (ex : yolov8)
                
        Returns:
            detected_objects (list):
                list of numpy object detected
            
            label_img (Dict):
                dict with the list of label from which images the objects comes from
                
        Example:
        --------
            >>> detect_object_dataset(path_img = [path/to/img1.png, path/to/img2.jpeg], model = YOLOV8)
            >>> returns : 
            
    """
    
    detected_objects = []
    label_img        = []
    
    boxes, labels_boxes = extract_boxes_yolovn(model, img_cv2)
    for bounding_objects, label_object in zip(boxes, labels_boxes):  # if no detected object, image not considerated
        detected_objects.append(bounding_yolovn(bounding_objects, img_cv2))
        label_img.append(DICT_LABEL_YOLOV8[label_object])
            
    return detected_objects, {"image_label" : label_img}