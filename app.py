from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
import utilities.utilities_images as utilities_images
import utilities.getter as getter
import scripts.extract_objects as extract_objects

PATH_FOLDER_IMG  = "./static/images/"
PATH_DB          = "./static/vector_store/"

app = Flask(__name__)
@app.route("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        
        update_db     = request.form.get('update_db')
        image_query   = request.files["image"]
        nb_similar    = int(request.form.get('nb_similar'))
        print(update_db)
        print(image_query)
        print(nb_similar)
        
        image_bytes = image_query.read()
        # Convertir les bytes en tableau numpy
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # Décoder l'image en format OpenCV
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
        if not extract_objects.vector_store_exist():
            print("not created")
            db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB)
        elif update_db == "True":
            print("need update")
            db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB)  
        else :
            db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB, 
                                                                                    PATH_DB)  

            
        images_results, draw_img = extract_objects.search_engine_image(query= image_cv2, 
                                                                        db=db, 
                                                                        vectorizer=vectorizer, 
                                                                        data_loader=data_loader, 
                                                                        nb_similar=nb_similar)
        
        # save image input with object detection and then reload it to display it
        im = Image.fromarray(draw_img)
        im.save(f"./static/inputs/query.png")
        
        # display similar objects for the first input detected object
        key_result = list(images_results.keys())
        image_result_first_object = list(set(images_results[key_result[0]]["images_path"]))
        print(key_result[0], list(set(image_result_first_object)))
            
        return render_template('resultats.html', main_image=f"./static/inputs/query.png", imagelist = image_result_first_object)
    return render_template('index.html')

@app.route('/search_by_image')
def search_by_image():
    db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB, 
                                                                                    PATH_DB)  
    # Récupérer le chemin de l'image passée comme argument
    image_path = request.args.get('image_path')
    image_cv2 = cv2.imread(image_path)
    
    images_results, draw_img = extract_objects.search_engine_image(query= image_cv2, 
                                                                    db=db, 
                                                                    vectorizer=vectorizer, 
                                                                    data_loader=data_loader, 
                                                                    nb_similar=5)
        
    # save image input with object detection and then reload it to display it
    im = Image.fromarray(draw_img)
    im.save(f"./static/inputs/query.png")
    
    # display similar objects for the input detected object
    print(images_results)
    image_result_first_object = []
    for obj in images_results.keys():
        image_result_first_object += images_results[obj]["images_path"]
        
    print(list(set(image_result_first_object)))
        
    return render_template('resultats.html', main_image=f"./static/inputs/query.png", imagelist = list(set(image_result_first_object)))

if __name__ == '__main__':
    app.run(debug=True, threaded=False)


