from  flask import Flask,render_template,request
import pandas as pd
import numpy as np
from  sklearn.externals import joblib
from _datetime import datetime

app = Flask(__name__)
model = joblib.load(open('classical_decomposition_regression_model', 'rb'))


@app.route('/test')
def test():
    return 'Classical Time Series Decomposition Model'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    for direct api calls through request
    """
    global model_pred
    if request.method == 'POST':
        date = int(request.form['x'])
        pred_args =[date]
        pred_args_arr=np.array(pred_args)
        pred_features=pred_args_arr.reshape(1,-1)
        model_reg=open("classical_decomposition_regression_model", "rb")
        ml_model=joblib.load(model_reg)
        model_pred=ml_model.predict(pred_features)
        model_pred=round(float(model_pred), 2)

    return render_template('predict.html', predictions=model_pred)



if __name__ == "__main__":
    app.run(debug=True)
