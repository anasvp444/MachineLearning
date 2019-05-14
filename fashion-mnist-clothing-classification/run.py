

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
data= ["T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"
]


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
	n = randint(1000, 9999)
	path = "static/img/" + str(n)+ ".png"
	img.save(path)
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img,path

# load an image and predict the class
def predict():
	# load the image
	img,path = load_image('images//image.jpg')
	# load model
	model = load_model('final_model.h5')
	# predict the class

	result = model.predict_classes(img)
	K.clear_session()
	return(data[int(result[0])],path)






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