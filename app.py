import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder 
from ohe import OneHotEncoder, dataframe
from pandas import DataFrame

app = Flask(__name__)
model = pickle.load(open('xgb_model3.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('one_hot_encoder (2).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    labels = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term','Credit_History','Property_Area']
    numerical_values = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term']
    categorical_values = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
    cat_features = [request.form.get(key) for key in categorical_values]
    int_features = [int(request.form.get(key)) for key in numerical_values]
    ohe = encoder.transform(np.array(cat_features).reshape(1,-1))
    scaled_features = scaler.transform(np.array(int_features).reshape(1,-1))
    num_data = DataFrame(scaled_features)
    frame = [num_data,ohe]
    final_df = pd.concat(frame,axis=1)
    final_features = np.array(final_df)
    prediction = model.predict(final_features)

    if prediction == 1:
        output = 'Approved'
    else:
        output = 'Rejected'

    return render_template('index.html', prediction_text='Loan is {}'.format(output))


if __name__ == "__main__":
    #port-int(os.environ.get('PORT',5000))
    #app.run(debug=True, host='0.0.0.0', port=port)
    app.run(debug=True)