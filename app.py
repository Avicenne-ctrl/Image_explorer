from flask import Flask, render_template, request
import zipp
import os
import pandas as pd
import os
import sys
sys.path.append('..')
import utilities.utilities_images as utilities_images
import scripts.extract_objects as extract_objects

PATH_FOLDER_IMG  = "./static/images/"
PATH_DB          = "./static/vector_store/"

app = Flask(__name__)


@app.route("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        
        UPDATE_DB     = request.form.get('update_db')
        print(UPDATE_DB)
        
        if not extract_objects.vector_store_exist():
            print("not created")
            db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB)
        elif UPDATE_DB == True:
            print("need update")
            db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB)    
        else:
            print("already created")
            db, vectorizer, data_loader = extract_objects.init_search_engine_images(PATH_FOLDER_IMG, 
                                                                                    PATH_DB, PATH_DB)
            
            
        
        
            
        
        

        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False)


