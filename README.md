# Image_explorer

## Image Search Engine
This repository implements an image search engine that leverages object detection and image vectorization using YOLO for object detection and ResNet for image vectorization. The engine creates a vector store to enable efficient image search based on similarity.

### Features:
- Object Detection: Detects objects in images using the YOLO model.
- Image Vectorization: Vectorizes images using a pre-trained ResNet model.
- Vector Store Management: Creates and manages vector stores using FAISS for fast image similarity search.
- Search Engine: Enables querying an image, detects objects, vectorizes them, and retrieves the most similar images from the database.  

### Dependencies
To use this repository, the following dependencies are required:  

- pandas
- numpy
- opencv-python
- matplotlib
- langchain
- torch
- torchvision
- tensorflow
- ultralytics
- faiss
- Setup


### 1. Clone the repository

`git clone https://github.com/your-repo/image-search-engine.git`


### 2. Install required dependencies
`pip install -r requirements.txt`


### 3. Pre-trained Models
- Object Detection Model: YOLOv8 is used for object detection.
- Vectorization Model: A pre-trained ResNet18 is used for feature extraction.

```python
MODEL_VECTORIZE_IMG = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
MODEL_DETECT_OBJ = YOLO('yolov8n.pt')```


### extract_objects.py : Functions Overview
`vector_store_exist()`
Checks if the vector store exists and if it's not empty.


`detect_object_dataset(path_img: list, model: str= MODEL_DETECT_OBJ)`
Detects objects in images using the YOLO model.

Args:

path_img (list): List of paths to images.
model (str): Model used for detection (default is YOLOv8).


Returns:

detected_objects (list): List of detected objects.
label_img (list): Dict of Corresponding image labels.

Example : 

```python
detected_objects, label_img =  = detect_object_dataset(path_img = ["img1.png", "img2.jpeg"])```


`detect_object_query(img_cv2, model = YOLOV8)`
Detects objects in images using the YOLO model.

Args:

img_cv2 (numpy.ndarray): the cv2 loaded image
model (str): Model used for detection (default is YOLOv8).


Returns:

detected_objects (List[numpy.ndarray]): List of detected objects.
label_img (Dict[str:str]): Dict of corresponding image labels.

Example : 

```python
img_cv2 = cv2.imread("./static/images/cat0.jpeg")
detected_objects, label_img = detect_object_query(img_cv2)
```

`init_search_engine_images(path_folder, path_store_db, path_db="", extension_img=EXTENSION_IMG, batch_size=BATCH_SIZE, size_img=SIZE_IMG
model_vectorize_img=MODEL_VECTORIZE_IMG)  `

Initializes the image search engine by creating or loading the vector store.

Args:

path_folder (str): Path to the folder containing images.
path_store_db (str): Path to store the vector database.
path_db (str): Path to the pre-existing vector store (optional).
extension_img (list): List of image extensions.
batch_size (int): Batch size for processing.
size_img (tuple): Size of the images for vectorization.
model_vectorize_img (str): Model for image vectorization.


Returns:

The database, image vectorizer, and DataLoader.

Example : 

```python
db, vectorizer, data_loader = init_search_engine_images(path_folder="./images", path_store_db="./vector_store")``


search_engine_image(query, db, vectorizer, data_loader, nb_similar=5)
Searches the database for images similar to the provided query.

Args:

query (list): List of images for querying.
db (str): Vector store database.
vectorizer (str): Image vectorizer model.
data_loader (str): DataLoader for batch processing.
nb_similar (int): Number of similar images to retrieve.

Returns:

A dictionary of results containing the labels and images of the most similar objects.

Example : 
```python
results = search_engine_image(query=["query_image.jpg"], db=db, vectorizer=vectorizer, data_loader=data_loader)```


init_search_engine_images(path_folder="./static/images", path_store_db="./static/vector_store")`
Initialize the search engine and create the vector store:

Args:

path_folder (str): path to the folder which contains the images we want to vectorize
path_store_db (str): where we want to store the vector store

Returns:
db (FAISS): the database, vector store 
vectorizer (ResNet): the vectorizer of images
data_loader (Tensor): the batches of Tensor dataset of images 

Example :

```python
db, vectorizer, data_loader = init_search_engine_images(path_folder="./static/images", 
                                                        path_store_db="./static/vector_store")```


`search_engine_image(query, db, vectorizer, data_loader, nb_similar=5) `                                       
Search for similar images using a query image:

Args:

query (numpy.ndarray): the image we want to detect object and find simlarity
db (FAISS): faiss vector store
vectorizer (ResNet): resnet vectorizer
data_loader (Tensor): the tensor batch dataset of images
nb_similar=5 (int): the number of similar images we want, 5 by default

Returns:

dict_results(Dict[str:str, numpy.ndarray]): the dict result with these following keys ("{label}", "images_path", "labels", "img")
draw_objects(numpy.ndarray): the query image with bounding boxes drew

Example : 

```python
dict_results, draw_objects = search_engine_image(query=["./query_images/query1.jpg"], db=db, vectorizer=vectorizer, data_loader=data_loader, nb_similar=5)```


License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your changes.

Contact
For any questions or issues, feel free to contact us at [your-email@example.com].

