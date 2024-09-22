# import dependencies
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
from torchvision import datasets, transforms
import tensorflow as tf
from tensorflow import keras
from torch.utils.data import Subset
from ultralytics import YOLO
import os

import sys
sys.path.append('..')
sys.path.append('././.')
import utilities.utilities_images as utilities_images
import utilities.get_vector_store as get_vector_store
import utilities.get_similar_test as get_similar_test
import utilities.getter as getter

PATH_FOLDER_IMG     = "./static/images/"
PATH_DB             = "./static/vector_store"
PATH_VECTOR_STORE   = "./static/vector_store/"

EXTENSION_IMG       = ["jpeg"]

BATCH_SIZE          = 4
SIZE_IMG            = (512, 512)

MODEL_VECTORIZE_IMG = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
MODEL_DETECT_OBJ    = model = YOLO('yolov8n.pt')



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


def detect_object_dataset(path_img: list, model: str= MODEL_DETECT_OBJ):
    """Detect objects on images

        Args:
        ------
            path_img (list): 
                list of path of images we want to detect objects on
                
            model (str): 
                the model to detect object (ex : yolov8)
                
        Returns:
        --------
            detected_objects (list):
                list of numpy object detected
            
            label_img (list):
                list of label from which images the objects comes from
                
        Example:
        --------
            >>> detect_object_dataset(path_img = [path/to/img1.png, path/to/img2.jpeg], model = YOLOV8)
            >>> returns : 
            
    """
    
    label_img        = []
    detected_objects = []
    for img in path_img:
        img_cv2 = cv2.imread(img)
        boxes = utilities_images.extract_boxes_yolovn(model, img_cv2)
        print(len(boxes))
        for bounding_objects in boxes:
            detected_objects.append(utilities_images.bounding_yolovn(bounding_objects, img_cv2))
            label_img.append(img)
            
    return detected_objects, label_img

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
        --------
            path_folder (str):
                the path to the folder of images
                
            path_db (str):
                path to the vector store if created
                
        Returns:
        ---------
            the database, the vectorizer of images and the Tensor data loader
    """
    
    # check if path folder of images -> create database and save, add functionalities if folder modified -> update database
    # if db no need to generate the db again
    
    # maybe stored in a params txt file
    
    # extract images from repo
    list_path_images, list_names = getter.get_extension_folder(path_folder, extension_img)
    
    # extract objects from images, and labels save from which images it comes from
    list_images, labels          = detect_object_dataset(list_path_images)
        
    # create dataloader, tensor object made of batch
    data_loader = utilities_images.create_dataloader_from_images(list_img  = list_images, 
                                                                list_index = labels, 
                                                                batch_size = batch_size,
                                                                size_img   = size_img)

    # vectorizer from a pretrained resnet modified
    vectorizer = get_vector_store.ImageVectorizer(model_vectorize_img)

    # if db already exist or not
    if len(path_db) == 0:
        # Utilisez un DataLoader avec vos images
        embeddings, labels = vectorizer.vectorize_images(data_loader)  # Vectorise les images dans le DataLoader
        db = get_vector_store.create_images_vector_stores(embeddings, labels)
        get_vector_store.save_vector_store(db = db, path_store_db = path_store_db)
        
    else:
        db = get_vector_store.load_vector_store(path_db, None)
        
    return db, vectorizer, data_loader

def search_engine_image(query: list, db: str, vectorizer: str, data_loader: str, nb_similar: int = 5):
    
    # we detect objects in the query img
    object_queries, _ = detect_object_dataset(query)
    n = len(object_queries)
    
    print(f"{n} objects detected")
    
    # we create the tensor dataloader for each objects detected
    loader_one        = utilities_images.create_dataloader_from_images(object_queries, [""]*n, batch_size = n)
    # we embed the dataloader
    embedding_query, _       = vectorizer.vectorize_images(loader_one)
    
    dict_results = {}
    
    # for each vector we search for the most similar among the database (db)
    for i, embed in enumerate(embedding_query):
        dict_obj = {"labels": [], "img": []}
        
        result_doc  = db.similarity_search_by_vector(embed, nb_similar) 
        
        for result in result_doc:        
            position = int(result.metadata["image_position"])        
            img, _ = utilities_images.get_image_from_batch(data_loader, position, BATCH_SIZE)
            
            dict_obj["labels"].append(result.metadata["image_label"])
            dict_obj["img"].append(img)
                        
        dict_results[f"object{i}"] = dict_obj
        
    return dict_results







