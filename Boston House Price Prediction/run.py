

from flask import Flask, render_template, request, flash ,redirect, url_for, Response

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit



data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):

    score = r2_score(y_true, y_predict)
    return score


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):

    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)

    return grid.best_estimator_

reg = fit_model(X_train, y_train)




app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'




@app.route('/', methods=['GET', 'POST'])
def index():

            if request.method == 'POST':
                room = int(request.form['room'])
                poverty = int(request.form['poverty'])
                ratio = int(request.form['ratio'])
                client_data = [[room, poverty, ratio]]
                price = reg.predict(client_data)
                price = ("$%0.2f" % reg.predict(client_data)[0])
                return render_template('index.html',price = price)
            else:
                return render_template('index.html',price = "")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)            