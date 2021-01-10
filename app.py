import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('xgb_model3.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('one_hot_encoder.pkl', 'rb'))

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
    cat_data = pd.DataFrame(columns=list(categorical_values))
    cat_data.loc[0] = cat_features
    ohe = encoder.transform(cat_data)
    scaled_features = scaler.transform(np.array(int_features).reshape(1,-1))
    frame = [scaled_features,ohe]
    final_df = pd.concat(frame,axis=1)
    final_features = np.array(final_df)
    prediction = model.predict(final_features)

    if prediction == 1:
        output = 'Approved'
    else:
        output = 'Rejected'

    return render_template('index.html', prediction_text='Loan is {}'.format(output))


if __name__ == "__main__":
    # port-int(os.environ.get('PORT',5000))
    app.run(debug=True)