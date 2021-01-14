import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from ohe import dataframe
from pandas import DataFrame
class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False
    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out
    def transform(self, X, **kwargs):
      sparse_matrix = super(OneHotEncoder, self).transform(X)
      new_columns = ['Gender_<Female>',
      'Gender_<Male>',
      'Married_<No>',
      'Married_<Yes>',
      'Dependents_<0>',
      'Dependents_<1>',
      'Dependents_<2>',
      'Dependents_<3+>',
      'Education_<Graduate>',
      'Education_<Not Graduate>',
      'Self_Employed_<No>',
      'Self_Employed_<Yes>',
      'Credit_History_<0.0>',
      'Credit_History_<1.0>',
      'Property_Area_<Rural>',
      'Property_Area_<Semiurban>',
      'Property_Area_<Urban>']
      d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns)
      return d_out
    # def get_new_columns(self, X):
    #   new_columns = []
    #   for i, column in enumerate(X.columns):
    #       j = 0
    #       while j < len(self.categories_[i]):
    #         new_columns.append(f'{column}_<{self.categories_[i][j]}>')
    #         j += 1
    #   return new_columns
    def fit_transform(self, X, **kwargs):
      self.fit(X)
      return self.transform(X)

class dataframe():
    def __init__(self,val,features):
      self.features = features
      self.val = val
    def convert(x):
      data = pd.DataFrame(columns=list(x.val))
      data.loc[0] = x.features
      return data


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
    port-int(os.environ.get('PORT',5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    #app.run(debug=True)