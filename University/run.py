

from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, flash, redirect, url_for, Response

import numpy as np
import pandas as pd


data = pd.read_csv('Admission_Predict_Ver1.1.csv')
continues_data = data[['GRE Score', 'TOEFL Score',
                       'University Rating', 'SOP', 'LOR ', 'CGPA']].values / 100
categorical_data = data[['Research']].values

X = np.concatenate([continues_data, categorical_data], axis=1)
y = data[['Chance of Admit ']].values


regressor = LinearRegression()
regressor.fit(X, y)


print(X[1])
data = X[1].reshape(1, 7)
print(data.shape)
output = regressor.predict(data)
print(output)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        GRE = int(request.form['GRE'])
        TOEFL = int(request.form['TOEFL'])
        UniversityRating = int(request.form['UniversityRating'])
        SOP = float(request.form['SOP'])
        LOR = float(request.form['LOR'])
        CGPA = float(request.form['CGPA'])
        if(request.form['research'] == "Yes"):
            research = 1
            researchString = 'Yes'
        else:
            research = 0
            researchString = 'No'
        data = [[GRE, TOEFL, UniversityRating, SOP, LOR, CGPA, research]]
        data = np.array(data)/100

        pred = regressor.predict(data)[0]
        if(pred < 0):
            pred = 0
        elif(pred > 1):
            pred = 1
        pred = str(int(pred*100))+'%'
        # pred = ("%0.2f" % pred)

        predString = 'You have '+pred+' chance to get Admission'
        print(predString)
        return render_template('index.html', GRE=GRE, TOEFL=TOEFL, UniversityRating=UniversityRating,
                               SOP=SOP, LOR=LOR, CGPA=CGPA, isResearch=researchString, prob=predString)
    else:
        return render_template('index.html', GRE="", TOEFL="", UniversityRating="",
                               SOP="", LOR="", CGPA="", research="", pred="")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
