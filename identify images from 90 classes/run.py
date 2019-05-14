

from flask import Flask, render_template, request, flash ,redirect, url_for, Response
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from random import randint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K


import cv2

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value


# Loading model
model = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')



# load an image and predict the class
def predict():

	image= cv2.imread("images//image.jpg")
	image_height, image_width, _ = image.shape
	model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
	output = model.forward()

	class_name =""
	for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .5:
                class_id = detection[1]
                class_name += " "+id_class_name(class_id,classNames)
                c_name = id_class_name(class_id,classNames)
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=2)
                cv2.putText(image,c_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0, 0, 255),2)
	n = randint(1000, 9999)
	path = "static/img/" + str(n)+ ".png"
	cv2.imwrite( path,image)
	return(class_name,path)		






app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'




@app.route('/', methods=['GET', 'POST'])
def index():
            form = UpdateAccountForm()
            if request.method == 'POST':

                if 'picture' not in request.files:
                    flash('No file part')
                    return redirect(request.url)
                file = request.files['picture']     
                if file:
                    file.save("images//image.jpg")        
                    data,path = predict()
                    return render_template('index.html',data = data,form=form, filename = path)
            else:
                return render_template('index.html',data = "",form=form)

class UpdateAccountForm(FlaskForm):
    picture = FileField('Upload a cloth', validators=[FileAllowed(['jpg'])])
    submit = SubmitField('Upload')                

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)            