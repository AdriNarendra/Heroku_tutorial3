from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd

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