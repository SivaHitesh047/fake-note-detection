from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator,image
import numpy as np
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    print("hello 1")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img=image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename),target_size=(300,300)) #The path of the testing image,the pic taken from the phone should come her   
        img=np.asarray(img)
        #plt.imshow(img)
        img=np.expand_dims(img,axis=0) / 255.0
        output=loaded_model.predict(img) 
        print(output)
        if(output[0][0]>output[0][1]): #comparison
            print("inside fake")
            flash("Note is fake")
        else:
            print("inside real")
            flash("Note is real")
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<string:filename>')
def display_image(filename):
    print("hello 2")
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == '__main__':
    app.run()