# Image_explorer

Image Search Engine
This repository implements an image search engine that leverages object detection and image vectorization using YOLO for object detection and ResNet for image vectorization. The engine creates a vector store to enable efficient image search based on similarity.

Features:
Object Detection: Detects objects in images using the YOLO model.
Image Vectorization: Vectorizes images using a pre-trained ResNet model.
Vector Store Management: Creates and manages vector stores using FAISS for fast image similarity search.
Search Engine: Enables querying an image, detects objects, vectorizes them, and retrieves the most similar images from the database.
Dependencies
To use this repository, the following dependencies are required:

bash
Copier le code
pandas
numpy
opencv-python
matplotlib
langchain
torch
torchvision
tensorflow
ultralytics
faiss
Setup
1. Clone the repository
bash
Copier le code
git clone https://github.com/your-repo/image-search-engine.git
cd image-search-engine
2. Install required dependencies
bash
Copier le code
pip install -r requirements.txt
3. Pre-trained Models
Object Detection Model: YOLOv8 is used for object detection.
Vectorization Model: A pre-trained ResNet18 is used for feature extraction.
python
Copier le code
MODEL_VECTORIZE_IMG = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
MODEL_DETECT_OBJ = YOLO('yolov8n.pt')
Functions Overview
vector_store_exist()
Checks if the vector store exists and if it's not empty.

python
Copier le code
vector_store_exist()
detect_object_dataset(path_img: list, model: str= MODEL_DETECT_OBJ)
Detects objects in images using the YOLO model.

Args:

path_img (list): List of paths to images.
model (str): Model used for detection (default is YOLOv8).
Returns:

detected_objects (list): List of detected objects.
label_img (list): Corresponding image labels.
python
Copier le code
detect_object_dataset(path_img = ["img1.png", "img2.jpeg"], model = YOLOV8)
init_search_engine_images(path_folder, path_store_db, path_db="", extension_img=EXTENSION_IMG, batch_size=BATCH_SIZE, size_img=SIZE_IMG, model_vectorize_img=MODEL_VECTORIZE_IMG)
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
python
Copier le code
db, vectorizer, data_loader = init_search_engine_images(path_folder="./images", path_store_db="./vector_store")
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
python
Copier le code
results = search_engine_image(query=["query_image.jpg"], db=db, vectorizer=vectorizer, data_loader=data_loader)
Example Usage
Initialize the search engine and create the vector store:
python
Copier le code
db, vectorizer, data_loader = init_search_engine_images(path_folder="./static/images", 
                                                        path_store_db="./static/vector_store")
Search for similar images using a query image:
python
Copier le code
results = search_engine_image(query=["./query_images/query1.jpg"], db=db, vectorizer=vectorizer, data_loader=data_loader, nb_similar=5)

# Display the results
for key, value in results.items():
    print(f"Object: {key}, Labels: {value['labels']}")
    # Visualize images here
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your changes.

Contact
For any questions or issues, feel free to contact us at [your-email@example.com].

