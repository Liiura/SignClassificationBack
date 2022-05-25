import os
from flask import Flask, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import pickle
import numpy as np
import cv2
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', }

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            Flask('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            Flask('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            scaleImage(filename)
            imageResized = convertImageToCsvPandas(filename)
            imageClassifiedMessage = ClassificImage(imageResized)
            return jsonify({ 'message': imageClassifiedMessage })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def scaleImage(fileName):
    img = Image.open("./images/"+fileName)
    resized_img = img.resize((28, 28))
    resized_img.save("./images/"+fileName)

def convertImageToCsvPandas(fileName):
    img=Image.open("./images/"+fileName) 
    img_28x28=np.array(img)
    gray = cv2.cvtColor(img_28x28, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, 2)
    return gray.flatten()

def ClassificImage(some_letter):
    classification = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z' ]
    with open('./model/model_pickle.pkl','rb') as file:
        mp = pickle.load(file)
    sign_pred = mp.predict([some_letter])
    print(sign_pred)
    return "the prediction was "+classification[int(sign_pred)]
if __name__ == "__main__":
    app.run(debug = True, port=3001)
