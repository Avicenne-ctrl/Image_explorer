import numpy as np
import cv2
import os 
import numpy
import torch
from torchvision import transforms
import sys
from typing import List, Dict
sys.path.append("..")
import scripts.object_detection as object_detection

def create_dataloader_from_images(list_img: List[numpy.ndarray], 
                                  list_index: List[int], 
                                  batch_size: int = 10, 
                                  size_img: tuple = (512, 512), 
                                  shuffle=False)-> torch.utils.data.DataLoader:
    """
        Convert list of images and index into Tensor DataLoader

    Args:
        list_img (List[numpy.ndarray]): 
            list of images we want to transform
            
        list_index (List[int]): 
            list of index we want to transform
            
        batch (int, optional): 
            Defaults to 10.
            
        shuffle (bool, optional): 
            Defaults to False, we don't want to shuffle to not lose the order.
            
    Returns:
        torch.utils.data.DataLoader
        
    """
    # transformer d'image
    tc = transforms.Compose([
        transforms.ToPILImage(),                      # Convertit l'image NumPy en image PIL
        transforms.Resize(size_img),                # Redimensionne l'image à 256x256 pixels
        transforms.ToTensor()                         # Convertit l'image en tenseur PyTorch
    ])
    images_list = [(tc(img), idx) for idx, img in zip(list_index, list_img)]
    images_loader = torch.utils.data.DataLoader(images_list, batch_size=batch_size, shuffle=shuffle)
    return images_loader

def get_image_from_batch(image_loader: torch.utils.data.DataLoader, index_result: int, batch_size: int):
    """
        Once we have the position of the most similar image
        we need to find it among the batch in the data loader
        
        Args :
            image_loader (torch.utils.data.DataLoader):
                the Tensor object of images with their label
                
            index_result (int):
                the index of the most similar image
                
            show_image
                
        Returns :
            the tensor of the image and its label
            
        Example :
        ---------
            you embedded your images with a batch 10, the position of the most similar imge
            is 213. You need to find the batch and then the position in the batch
            >>> get_image_from_batch(image_loader, 213) 
            >>> result -> the image Tensor, the label
    """

    ind_batch    = index_result // batch_size
    ind_in_batch = index_result % batch_size
    for i , (images, label) in enumerate(image_loader):
        if i == ind_batch:

            return images[ind_in_batch], label[ind_in_batch]
        
def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, corner_radius):
    """
        Draws a rectangle with rounded corners around the detected object.

        Args:
            image (numpy.ndarray): 
                The input image on which to draw.
                
            top_left (tuple): 
                Coordinates of the top-left corner of the bounding box.
                
            bottom_right (tuple): 
                Coordinates of the bottom-right corner of the bounding box.
                
            color (tuple): 
                The color of the rectangle (BGR format).
                
            thickness (int):    
                Thickness of the lines for the rectangle.
                
            corner_radius (int): 
                Radius for rounding the corners.
    """

    # Calculate corner points
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    
    
    
    #proportion
    large = abs(x_max - x_min)
    long = abs(y_max - y_min)

    # Ensure the radius is not too large
    corner_radius = min(corner_radius, (x_max - x_min) // 2, (y_max - y_min) // 2)

    # Draw straight lines for the sides
    # cv2.line(image, (x_min + corner_radius, y_min), (x_max - corner_radius, y_min), color, thickness)
    # cv2.line(image, (x_min + corner_radius, y_max), (x_max - corner_radius, y_max), color, thickness)
    # cv2.line(image, (x_min, y_min + corner_radius), (x_min, y_max - corner_radius), color, thickness)
    # cv2.line(image, (x_max, y_min + corner_radius), (x_max, y_max - corner_radius), color, thickness)
    
    # Draw short lines at each corner (horizontal and vertical segments)
    
    # Top-left corner
    cv2.line(image, (x_min + corner_radius, y_min), (int(large*0.2) + x_min + corner_radius, y_min), color, thickness)  # Horizontal line
    cv2.line(image, (int((0.8*large) + x_min - corner_radius), y_min), (x_max - corner_radius, y_min), color, thickness)  # Horizontal line
    
    cv2.line(image, (x_min + corner_radius, y_max), (int(large*0.2) + x_min, y_max), color, thickness)  # Horizontal line
    cv2.line(image, (int((0.8*large) + x_min), y_max), (x_max - corner_radius, y_max), color, thickness)  # Horizontal line
    
    cv2.line(image, (x_min, y_min + corner_radius), (x_min, y_min + corner_radius + int(long*0.2)), color, thickness)
    cv2.line(image, (x_min, y_min + int(long*0.8)), (x_min, y_max - corner_radius), color, thickness)
    
    cv2.line(image, (x_max, y_min + corner_radius), (x_max, y_min + corner_radius + int(long*0.2)), color, thickness)
    cv2.line(image, (x_max, y_min + int(long*0.8)), (x_max, y_max - corner_radius), color, thickness)

    # Draw the rounded corners
    cv2.ellipse(image, (x_min + corner_radius, y_min + corner_radius), (corner_radius, corner_radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x_max - corner_radius, y_min + corner_radius), (corner_radius, corner_radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x_min + corner_radius, y_max - corner_radius), (corner_radius, corner_radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x_max - corner_radius, y_max - corner_radius), (corner_radius, corner_radius), 0, 0, 90, color, thickness)

    
def display_objects_image(img_cv2: numpy.ndarray, bounding_boxes: List[List[float]])->numpy.ndarray:
    """
        This function takes an image and try to detect object on it

        Args:
            img_cv2 (np.numpy): 
                the image we want to detect objects on, read by cv2
                
            bounding_boxes (List[List[float]]):
                List of list of coordinates
                
        Returns:
            image_illustrated (numpy.ndarray):
                the image with rectangle on detected object
    """
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # adapted to cv2.imread
    img_illustrated =  np.clip(img_rgb * 0.4, 0, 255).astype(np.uint8)
    
    for b in bounding_boxes:
        x_min, y_min, x_max, y_max =  b
        img_illustrated[int(y_min) : int(y_max), int(x_min) : int(x_max)] = object_detection.bounding_yolovn(b, img_rgb)
        # img_illustrated = cv2.rectangle(img_illustrated, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 0), 5)
        
        draw_rounded_rectangle(
            img_illustrated, 
            (int(x_min), int(y_min)), 
            (int(x_max), int(y_max)), 
            color=(200, 0, 0),  # Cyan color (like Google Lens)
            thickness=5,
            corner_radius=20
        )
    
    #plt.imshow(img_illustrated)
    return img_illustrated

def get_extension_folder(folder_path: str, extensions: List[str])->List[str]:
    """
        Get all the extension file in a specified folder
        ignoring the sub folder
    

        Args:
            folder_path (str): 
                the path to the folder
                
            extensions (list): 
                list of extension we want

        Returns:
            List[str]: 
                list of the path to the wanted extension file
    """
    file_paths = []
    file_names = []
    
    # Read the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # Check if its the right extension
        for ext in extensions:
            if os.path.isfile(file_path) and file.endswith(ext):
                # Add the path to the list
                file_paths.append(file_path)
                file_names.append(file)
    
    return file_paths, file_names