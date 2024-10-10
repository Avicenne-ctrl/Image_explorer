# import dependencies
import torch
from torchvision.models import ResNet18_Weights
from ultralytics import YOLO
import os
import numpy
import cv2
import sys
sys.path.append('..')
import scripts.object_detection as object_detection
import scripts.dataloader_utilities as dataloader_utilities
import scripts.images_vectorizer as images_vectorizer



device = torch.device('cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PATH_FOLDER_IMG     = "./static/images/"
PATH_DB             = "./static/vector_store"
PATH_VECTOR_STORE   = "./static/vector_store/"

EXTENSION_IMG       = ["jpeg", "png", "jpg"]

BATCH_SIZE          = 5
SIZE_IMG            = (256, 256)

MODEL_VECTORIZE_IMG = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)
MODEL_DETECT_OBJ    = YOLO('yolov8n.pt')

MODEL_VECTORIZE_IMG.to(device) 
MODEL_DETECT_OBJ.to(device) 

DICT_LABEL_YOLOV8 = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

#print(MODEL_DETECT_OBJ.names)

def vector_store_exist():
    """Check if the vector store exist and is not empty

    Returns:
        boolean : 
    """
    if os.path.exists(PATH_VECTOR_STORE) and len(os.listdir(PATH_VECTOR_STORE))>0:
        faiss_extension = False
        pkl_extension   = False
        for file in os.listdir(PATH_VECTOR_STORE):
            if file.endswith("pkl"):
                pkl_extension = True
            if file.endswith("faiss"):
                faiss_extension = True
        if faiss_extension and pkl_extension:
            return True
        else:
            return False
    return False


def init_search_engine_images(path_folder: str, 
                              path_store_db: str,
                              path_db: str = "", 
                              extension_img: list = EXTENSION_IMG, 
                              batch_size: int = BATCH_SIZE, 
                              size_img: tuple = SIZE_IMG, 
                              model_vectorize_img: str = MODEL_VECTORIZE_IMG,
                              ):
    """
        Initialize the search engine, create the vector if not created
        else load it
        
        Args:
            path_folder (str):
                the path to the folder of images
                
            path_db (str):
                path to the vector store if created
                
        Returns:
            the database, the vectorizer of images and the Tensor data loader
    """
    
    # check if path folder of images -> create database and save, add functionalities if folder modified -> update database
    # if db no need to generate the db again
    
    # maybe stored in a params txt file
    
    # extract images from repo
    list_path_images, list_names = dataloader_utilities.get_extension_folder(path_folder, extension_img)
    
    print(f"{len(list_path_images)} images found in folder")
    
    # extract objects from images, and labels save from which images it comes from
    list_images, labels          = object_detection.detect_object_dataset(list_path_images)
                
    # create dataloader, tensor object made of batch
    data_loader = dataloader_utilities.create_dataloader_from_images(list_img  = list_images, 
                                                                list_index = [""]*len(list_images), # we don't care about index here, only in vector store 
                                                                batch_size = batch_size,
                                                                size_img   = size_img)
    
    # vectorizer from a pretrained resnet modified
    vectorizer = images_vectorizer.ImageVectorizer(model_vectorize_img)

    # if db already exist or not
    if len(path_db) == 0:
        # Utilisez un DataLoader avec vos images
        embeddings = vectorizer.vectorize_images(data_loader)  # Vectorise les images dans le DataLoader
        db = images_vectorizer.create_images_vector_stores(embeddings, labels)
        images_vectorizer.save_vector_store(db = db, path_store_db = path_store_db)
        
    else:
        db = images_vectorizer.load_vector_store(path_db, None)
        
    return db, vectorizer, data_loader

def search_engine_image(query: numpy.ndarray, 
                        db: str, 
                        vectorizer: str, 
                        data_loader: str, 
                        nb_similar: int = 5):
    
    # we detect objects in the query img
    object_queries, dict_metadata = object_detection.detect_object_query(query)
    object_labels = dict_metadata["image_label"]
    n = len(object_queries)
    print(f"{n} objects detected")
    
    bounding_boxes, _ = object_detection.extract_boxes_yolovn(model=MODEL_DETECT_OBJ, image= query)
    draw_objects      = dataloader_utilities.display_objects_image(query, bounding_boxes)
    
    if n>0 : # object detected
    
        # we create the tensor dataloader for each objects detected
        
        loader_one        = dataloader_utilities.create_dataloader_from_images(object_queries, [""]*n, batch_size = n)
        # we embed the dataloader
        embedding_query  = vectorizer.vectorize_images(loader_one)
        
        dict_results = {}
        
        # for each vector we search for the most similar among the database (db)
        for i, embed in enumerate(embedding_query):
            dict_obj = {"images_path": [], "labels": [], "img": []}
            
            metadata_filter = {"image_label": object_labels[i]}
            result_doc  = db.similarity_search_by_vector(embed, nb_similar, metadata_filter) 
            for result in result_doc:        
                print(result.metadata)
                position = int(result.metadata["image_position"])        
                img, _ = dataloader_utilities.get_image_from_batch(data_loader, position, BATCH_SIZE)
                
                dict_obj["images_path"].append(result.metadata["images_path"])
                dict_obj["labels"].append(result.metadata["image_label"])
                dict_obj["img"].append(img)
                            
            dict_results[f"object_{i}"] = dict_obj
            
        return dict_results, draw_objects
    
    else:
        print("no detected object")
        return None




