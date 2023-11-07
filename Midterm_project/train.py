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

# creating a RF classifier 
model = RandomForestClassifier(n_estimators = 100)   
  
# Training the model on the training dataset 
# fit function is used to train the model using the training sets as parameters 
model.fit(X_train, y_train) 

# save the model to disk
filename = 'Course_MLZoomCamp/Midterm_project/model.sav'
pickle.dump(model, open(filename, 'wb'))

# performing predictions on the test dataset 
y_pred = model.predict(X_test)  
  
# using metrics module for accuracy calculation 
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 