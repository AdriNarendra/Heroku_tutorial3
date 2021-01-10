{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder\n",
    "class OneHotEncoder(SklearnOneHotEncoder):\n",
    "    def __init__(self, **kwargs):\n",
    "        super(OneHotEncoder, self).__init__(**kwargs)\n",
    "        self.fit_flag = False\n",
    "    def fit(self, X, **kwargs):\n",
    "        out = super().fit(X)\n",
    "        self.fit_flag = True\n",
    "        return out\n",
    "    def transform(self, X, **kwargs):\n",
    "      sparse_matrix = super(OneHotEncoder, self).transform(X)\n",
    "      new_columns = self.get_new_columns(X=X)\n",
    "      d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)\n",
    "      return d_out\n",
    "    def get_new_columns(self, X):\n",
    "      new_columns = []\n",
    "      for i, column in enumerate(X.columns):\n",
    "          j = 0\n",
    "          while j < len(self.categories_[i]):\n",
    "            new_columns.append(f'{column}_<{self.categories_[i][j]}>')\n",
    "            j += 1\n",
    "      return new_columns\n",
    "    def fit_transform(self, X, **kwargs):\n",
    "      self.fit(X)\n",
    "      return self.transform(X)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
