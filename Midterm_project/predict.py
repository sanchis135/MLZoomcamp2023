import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics   
import pickle

df = pd.read_csv('Course_MLZoomCamp/Midterm_project/data 2.csv')
df.drop(["Unnamed: 32"],axis=1,inplace=True)

# replacing values
df['diagnosis'].replace(['M', 'B'], [0, 1], inplace=True)

X = df.drop(columns="diagnosis" , axis=1)
y = df["diagnosis"]

X = pd.get_dummies(X,drop_first=True)
X.head()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

filename = 'Course_MLZoomCamp/Midterm_project/model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)