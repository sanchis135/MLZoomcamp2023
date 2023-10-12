import pandas as pd
import requests
import numpy as np
import random

#version of pandas library
#print(pd.__version__) #2.1.0

#download data .csv file
URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
response = requests.get(URL)
open("cars_hw3.csv", "wb").write(response.content)

#df = pd.read_csv('cars_hw3.csv')

#Select only the features from above
usecols = [
    'Make', 'Model',
    'Year', 'Engine HP', 'Engine Cylinders','Transmission Type',
    'Vehicle Style', 'highway MPG', 
    'city mpg', 'MSRP'
]

df = pd.read_csv('cars_hw3.csv', usecols=usecols)

df.columns = df.columns.str.replace(' ', '_').str.lower()
#print(df.head())

#Find columns that contain at least one NaN
#print(df.isnull().sum())
#Fill in the missing values of the selected features with 0
df['engine_hp'] = df.engine_hp.fillna(0)
df['engine_cylinders'] = df.engine_cylinders.fillna(0)

#Rename MSRP variable to price
df=df.rename(columns={"msrp": "price"})
#print(df.head())

################################################################################################################################
print('QUESTION 1')
#What is the most frequent observation (mode) for the column transmission_type?
mode = df.transmission_type.mode()
print(mode)
#Solution: AUTOMATIC

################################################################################################################################
print('QUESTION 2')
#What are the two features that have the biggest correlation in this dataset?
corrM = df.corr(numeric_only=True)
print(corrM)
#Solution: highway_mpg and city_mpg

################################################################################################################################
#Make price binary
#print(df.head())
df = df.assign(above_average = df.price)
df['above_average'] = np.where(df['price'] > df['price'].mean(), 1, 0) 
#print(df.head())
#Split the data
from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

del df_train['above_average']
del df_val['above_average']
del df_test['above_average']
################################################################################################################################
print('QUESTION 3')
#Which of these variables has the lowest mutual information score?
from sklearn.metrics import mutual_info_score
from IPython.display import display

categorical = ['make', 'model','year','transmission_type','vehicle_style']

numerical = [
    'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg',
    'price'
]

def calculate_mi(series):
    return mutual_info_score(series, df_train_full.above_average)

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')

display(df_mi.head())
#display(df_mi.tail())
#Solution: transmission_type

################################################################################################################################
print('QUESTION 4')

dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear', max_iter=1000, C=10, random_state=SEED)
model.fit(X_train, y_train)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict(X_val)

accuracy = np.round(accuracy_score(y_val, y_pred),2)
print(f'Accuracy = {accuracy}')

#Solution: Accuracy = 0.95

################################################################################################################################
print('QUESTION 5')

features = df_train.columns.to_list()
features

original_score = accuracy
scores = pd.DataFrame(columns=['eliminated_feature', 'accuracy', 'difference'])
for feature in features:
    subset = features.copy()
    subset.remove(feature)
    
    dv = DictVectorizer(sparse=False)
    train_dict = df_train[subset].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    model = LogisticRegression(solver='liblinear', max_iter=1000, C=10, random_state=SEED)
    model.fit(X_train, y_train)
    
    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    
    scores.loc[len(scores)] = [feature, score, original_score - score]

min_diff = scores.difference.min()
scores[scores.difference == min_diff]

#Solution: year

################################################################################################################################
print('QUESTION 6')

df = pd.read_csv('cars_hw3.csv', usecols=usecols)

df.columns = df.columns.str.replace(' ', '_').str.lower()
#print(df.head())

#Find columns that contain at least one NaN
#print(df.isnull().sum())
#Fill in the missing values of the selected features with 0
df['engine_hp'] = df.engine_hp.fillna(0)
df['engine_cylinders'] = df.engine_cylinders.fillna(0)

#Rename MSRP variable to price
df=df.rename(columns={"msrp": "price"})

df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=SEED)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values
df_train = df_train.drop('price', axis=1)
df_val = df_val.drop('price', axis=1)
df_test = df_test.drop('price', axis=1)

assert 'price' not in df_train.columns
assert 'price' not in df_val.columns
assert 'price' not in df_test.columns

dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)
scores = {}
for alpha in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=alpha, solver='sag', random_state=SEED)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    score = mean_squared_error(y_val, y_pred, squared=False)
    scores[alpha] = round(score, 3)
    print(f'alpha = {alpha}:\t RMSE = {score}')

print(scores)
print(f'The smallest `alpha` is {min(scores, key=scores.get)}.')

#Solution: The smallest `alpha` is 0.
