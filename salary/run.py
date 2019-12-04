

from flask import Flask, render_template, request, flash, redirect, url_for, Response


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        try:
            exp = float(request.form['exp'])
            salary = round(exp*12807.919+2400.9705)
            return render_template('index.html', exp=exp, salary=salary)
        except:
            return render_template('index.html', exp=" ", salary=" ")
    else:
        return render_template('index.html', exp=" ", salary=" ")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
