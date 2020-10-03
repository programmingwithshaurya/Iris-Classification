# importing
#getting the datase Defining the variable
#training
#Get the predcitions and accuracy score

import pandas as pd
import numpy as np 
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')

X = df.drop(columns=['Species'])
y = df['Species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = DecisionTreeClassifier()
classifier.fit(X, y)
predictions = classifier.predict( [ [6, 3, 4, 5] ] )
print(predictions)

# score = accuracy_score(y_test, predictions)
# print(score)
# print(df)
