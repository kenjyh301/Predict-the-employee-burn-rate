import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import soft
import pandas
import tensor

data= pandas.read_csv("dataset/train.csv")
X= data.loc[:,'Gender':'Burn Rate']
X=X.dropna()
y= X['Burn Rate']
X= X.loc[:,'Gender':'Mental Fatigue Score']

column= X.columns.tolist()
for column_name in column[0:3]:
    X[column_name]=pandas.factorize(X[column_name],sort=True)[0]

print(X)
print(y)

logreg= LogisticRegression(multi_class = 'multinomial',solver = 'lbfgs')
logreg.fit(X,y)
# print(logreg.coef_)




