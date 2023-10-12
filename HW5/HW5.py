#create virtual environment
#py -m venv ML_hw5
#activate virtual environment
#& E:\Cursos\Zoomcamp_ML\ML_hw5\Scripts\activate.ps1
#############################################################################################################################
print('QUESTION 1: version 2023.10.3')

#pip install pipenv
#pipenv --version

#############################################################################################################################
print('QUESTION 2: "_meta"')

#pipenv install numpy scikit-learn==1.3.1 flask

#############################################################################################################################
print('QUESTION 3: 0.902')

#PREFIX=https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2023/05-deployment/homework
#wget $PREFIX/model1.bin
#wget $PREFIX/dv.bin

import pickle

def load(name: str):
    with open(name,'rb') as f:
        return pickle.load(f)
    
dv = load('dv.bin')
model = load('model1.bin')

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0,1]

print(y_pred)

#############################################################################################################################
print('QUESTION 4: 0.13968947052356817')

import requests ## to use the POST method we use a library named requests

url = 'http://localhost:9696/predict' ## this is the route we made for prediction
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
response = requests.post(url, json=client) ## post the customer information in json format
result1 = response.json() ## get the server response
print(result1)

#############################################################################################################################
print('QUESTION 5: 147MB')

#install docker: 
#dowload image of docker: docker image pull svizor/zoomcamp-model:3.10.12-slim
#docker images

#############################################################################################################################
print('QUESTION 6: ')

client2 = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json=client2) ## post the customer information in json format
result2 = response.json() ## get the server response
print(result2)