import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
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
      new_columns = self.get_new_columns(X=X)
      d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
      return d_out
    def get_new_columns(self, X):
      new_columns = []
      for i, column in enumerate(X.columns):
          j = 0
          while j < len(self.categories_[i]):
            new_columns.append(f'{column}_<{self.categories_[i][j]}>')
            j += 1
      return new_columns
    def fit_transform(self, X, **kwargs):
      self.fit(X)
      return self.transform(X)

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
    port-int(os.environ.get('PORT',5000))
    app.run(debug=True, host='0.0.0.0', port=port)