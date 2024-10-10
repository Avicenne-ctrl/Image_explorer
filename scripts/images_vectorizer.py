import os
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from typing import Union, List, Dict

# source : https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/

def save_vector_store(db : FAISS, path_store_db : str)-> None:
    """
        save a vector store locally given the object, its name and path to store

        Args:
            db (str): 
                the vector store object
                
            path_store_db (str): 
                the path we want to store it
                
        Returns:
            None
            
        Raise:
        ------
            - if db is not FAISS type
    """
    
    path_vector = path_store_db 
    
    if not isinstance(db, FAISS):
        raise TypeError(f"wrong type object, expected FAISS, got : {type(db).__name__}")

    if not os.path.exists(path_store_db):
        print("Vector_Store folder doesn't exist")
        print("Vector_Store folder created in {}".format(path_store_db))

    elif os.path.exists(path_vector):
        print("be carefull {} already exists and is being replaced".format(path_vector))

    print("{} stored in {}".format(path_store_db, path_vector))
    
    db.save_local(path_vector)

class ImageVectorizer:
    def __init__(self, model: torch.nn.Module, get_layer: str = "avgpool"):
        """
        Initialise le vectoriseur d'images avec un modèle et la couche d'extraction.
        
        Arguments:
        - model: Un modèle PyTorch, par exemple ResNet.
        - get_layer: La couche à partir de laquelle extraire les embeddings (par défaut 'avgpool').
        """
        self.model = model
        self.layer = model._modules.get(get_layer)
        self.outputs = []
        self.hook = None  # Hook temporaire

    def _copy_embeddings(self, module, input, output):
        """
        Fonction interne pour copier les embeddings depuis la couche sélectionnée.
        """
        # Extraire les vecteurs d'embeddings de la sortie de la couche
        output = output[:, :, 0, 0].detach().cpu().numpy().tolist()
        self.outputs.append(output)

    def _attach_hook(self):
        """
        Attache un hook temporaire pour capturer les embeddings.
        """
        if self.hook is None:
            self.hook = self.layer.register_forward_hook(self._copy_embeddings)

    def _remove_hook(self):
        """
        Enlève le hook temporaire après utilisation pour éviter les fuites de mémoire.
        """
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def _clear_memory(self):
        """
        Libère la mémoire GPU si nécessaire et réinitialise les variables.
        """
        self.outputs = []  # Réinitialiser la liste des embeddings
        torch.cuda.empty_cache()  # Libérer la mémoire GPU si applicable
    
    def vectorize_images(self, dataloader: torch.utils.data.DataLoader):
        """
        Vectorise toutes les images dans le dataloader en utilisant le modèle spécifié.
        
        Arguments:
        - dataloader: Un PyTorch DataLoader contenant les images à vectoriser.
        
        Retourne:
        - Une liste de vecteurs d'embeddings pour toutes les images.
        """
        self._clear_memory()  # Nettoyer la mémoire et réinitialiser les variables
        self._attach_hook()   # Attacher le hook
        self.model.eval()     # Met le modèle en mode évaluation

        print("Vectorisation des images...")
        with torch.no_grad():  # Pas besoin de calcul des gradients
            for X, y in dataloader:
                X = X.to(next(self.model.parameters()).device)  # Assurer que les images sont sur le même device que le modèle
                _ = self.model(X)  # Passe les images dans le modèle

        self._remove_hook()  # Enlever le hook après utilisation
        print("...Vectorisation terminée")
        
        # Aplatir la liste des embeddings
        list_embeddings = [item for sublist in self.outputs for item in sublist]
        
        return list_embeddings
    
    def vectorize_single_image(self, image: torch.Tensor) -> list:
        """
        Vectorise une seule image.
        
        Arguments:
        - image: Un tenseur PyTorch de forme [C, H, W] représentant une image.
        
        Retourne:
        - Le vecteur d'embedding pour l'image.
        """
        self._clear_memory()  # Réinitialiser la mémoire
        self._attach_hook()   # Attacher le hook temporaire
        self.model.eval()     # Met le modèle en mode évaluation

        with torch.no_grad():  # Pas besoin de calcul des gradients
            image = image.unsqueeze(0).to(next(self.model.parameters()).device)  # Ajoute la dimension batch si nécessaire et envoie sur le bon device
            _ = self.model(image)  # Passe l'image dans le modèle
        
        self._remove_hook()  # Enlever le hook après utilisation
        
        return self.outputs[0]  # Retourner le vecteur d'embedding


def get_len_key_dict(dico: dict)->List[int]:
    """Get list of len values for each keys in the dict

        Args:
            dico (dict): 
                the dict we want to len of value

        Returns:
            list: 
                the list of length
                
        Raise:
        ------
        - if dico is not a dict
        - if the dict is empty
        
    """
    if not isinstance(dico, dict):
        raise TypeError(f"wrong type object, expected dict, got : {type(dico).__name__}")
    
    if len(dico.keys()) == 0:
        raise TypeError(f"dict empty no key found : {dico.keys()}")

    list_len = []
    for cle, valeur in dico.items():
        list_len.append(len(valeur))
    return list_len
            

def create_list_metadata(metadata_dict: Dict[str, list])-> List[Dict[str, list]]:
    """
        Given a dict of list, we create a list of dicts
        in order to create a metadata dict for each elements

        Args:
            metadata_dict (dict): 
                the dict of list with each keys with the same length of values
                
        Returns:
            list_dict_metadata (List[dict]):
                the list of dict for each element
                
        Example:
            >>> create_list_meatadata(metadata_dict = {"id" : [id0, id1, id2], "label" : ["label0", "label1", "label2"]})
            >>> list_dict_metadata = [{"id": id0, "label": "label0"}, {"id": id1, "label": "label1"}, {"id": id2, "label": "label2"}]
    """
    
    if not isinstance(metadata_dict, dict):
        raise TypeError(f"wrong type object, expected dict, got : {type(metadata_dict).__name__}")
    
    list_length_value_key = get_len_key_dict(metadata_dict)
    unique_length         = set(list_length_value_key)
    
    if len(unique_length) > 1:
        raise TypeError(f"list for each keys should be the same length, here the list of length for each key : {list_length_value_key}")
    
    if len(metadata_dict.keys()) == 0:
        raise TypeError(f"metadata_dict is empty : {metadata_dict.keys()}")
    

    list_dict_metadata = []
    n = list(unique_length)[0]
    
    for i in range(n):
        aux_dict = {}
        for key in metadata_dict.keys():
            aux_dict[key] = metadata_dict[key][i]
        list_dict_metadata.append(aux_dict)
            
    return list_dict_metadata
    
# https://www.analyticsvidhya.com/blog/2024/05/multimodal-search-image-application/

def create_images_vector_stores(images_vectors : List[list], images_metadata : dict)->FAISS:
    """
        Generate vector store for images embeddings,
        given a Torch dataloader


        Args :
            images_vectors (List[list]): 
                list of image vectors
                
            images_labels (List[int]): 
                list of labels for images

        Returns :
            db (FAISS): the vector store
    """
    # we don't really care about the embeddings here
    EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
        
    # trick : text vector store but embedding text is embedding image
    text_embeddings = [("", item) for item in images_vectors]  # Empty string, vector
    images_metadata["image_position"] = [idx for idx in range(len(images_vectors))]
    metadatas = create_list_metadata(images_metadata)

    # Create a FAISS index using the extracted text embeddings (might be empty)
    # and image paths as metadata
    db = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding= None,  # Not explicitly setting embedding (might depend on image_vectors)
        metadatas=metadatas
    )

    # Print information about the created database
    print(f"Vector Database: {db.index.ntotal} docs")

    # Return the created FAISS index object (database)
    return db


def load_vector_store(vector_store_path: str, embeddings: HuggingFaceEmbeddings)->FAISS:
    """
        Load a vector store given a path and embedding
        FAISS vectore store

        Args:
            vector_store_path (str): 
                path where the vector store is
                
            embeddings (str): 
                name of the embedder, could be None for images vector store for example

        Returns:
            db (FAISS): 
                the index vector store
                
        Raise:
        ------
            if embeddings not HuggingFaceEmbeddings type
            if vector_store_path doesn't exist
    """
    if not isinstance(embeddings, HuggingFaceEmbeddings) and embeddings is not None:
        raise TypeError(f"wrong type object, expected HuggingFaceEmbeddings, got : {type(embeddings).__name__}")

    if not os.path.exists(vector_store_path):
        raise TypeError("Error : vector store doesn't exist, make sure you provided the correct path : {vector_store_path}")
    
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
