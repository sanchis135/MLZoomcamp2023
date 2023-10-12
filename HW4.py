import pandas as pd
import requests
import numpy as np
import random
import matplotlib.pyplot as plt

#download data .csv file
URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
response = requests.get(URL)
open("cars_hw4.csv", "wb").write(response.content)

#Select only the features from above
usecols = [
    'Make', 'Model',
    'Year', 'Engine HP', 'Engine Cylinders','Transmission Type',
    'Vehicle Style', 'highway MPG', 
    'city mpg', 'MSRP'
]

df = pd.read_csv('cars_hw4.csv', usecols=usecols)

df.columns = df.columns.str.replace(' ', '_').str.lower()

#Fill in the missing values of the selected features with 0
df['engine_hp'] = df.engine_hp.fillna(0)
df['engine_cylinders'] = df.engine_cylinders.fillna(0)

#Find columns that contain at least one NaN
#print(df.isnull().sum())

df['above_average'] = df['msrp'].gt(df.msrp.mean()).astype(int)
#print(df.head())

from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.20, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train_price = df_train['msrp']
y_val_price = df_val['msrp']
y_test_price = df_test['msrp']

y_train = df_train['above_average']
y_val = df_val['above_average']
y_test = df_test['above_average']

df_train.drop(['msrp','above_average'],axis=1,inplace=True)
df_val.drop(['msrp','above_average'],axis=1,inplace=True)
df_test.drop(['msrp','above_average'],axis=1,inplace=True)

#############################################################################################################################
print('QUESTION 1: engine_hp, 0.916')

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

numericals = ['year','engine_hp','engine_cylinders','highway_mpg','city_mpg']

for i in numericals:
    auc = roc_auc_score(y_train,df_train[i])
    if auc < 0.5:
        auc = roc_auc_score(y_train,-df_train[i])
    print('%9s, %.3f' % (i,auc))

#############################################################################################################################
print('QUESTION 2: AUC = 0.9784891235596622')

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

categorical = ['make','model','transmission_type','vehicle_style']
train_dict = df_train[categorical + numericals].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train,y_train)

val_dict = df_val[categorical + numericals].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:,1]
#y_pred = model.predict(X_val)

print(roc_auc_score(y_val,y_pred))

#############################################################################################################################
print('QUESTION 3: precision vs recall = 0.48')

from sklearn.metrics import accuracy_score

scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))

columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)

#plt.plot(df_scores.threshold,df_scores.p,label='precision')
#plt.plot(df_scores.threshold,df_scores.r,label='recall')
#plt.legend()
#plt.show()

#############################################################################################################################
print('QUESTION 4: f1 = 0.52')

df_scores['f1'] = 2 * df_scores.p * df_scores.r / (df_scores.p + df_scores.r)

print(df_scores.f1.max())

print(df_scores[df_scores.f1 == 0.8882629107981221])

#############################################################################################################################
print('QUESTION 5: 0.979 +- 0.003')

from sklearn.model_selection import KFold

def train(df,y,C=1.0):
    dicts = df[categorical + numericals].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train,y_train)
    
    return dv,model

dv, model = train(df_train,y_train)

def predict(df,dv,model):
    dicts = df[categorical+numericals].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred

y_pred = predict(df_val,dv,model)

kfold = KFold(n_splits=5,shuffle=True,random_state=1)

from tqdm.auto import tqdm

scores = []
for train_idx,val_idx in tqdm(kfold.split(df_train_full)):
    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]
    
    y_train = df_train.above_average.values
    y_val = df_val.above_average.values
    
    dv,model = train(df_train,y_train)
    y_pred = predict(df_val,dv,model)
    
    auc =roc_auc_score(y_val,y_pred)
    scores.append(auc)
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

#############################################################################################################################
print('QUESTION 6: C=10')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 0.5, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train.iloc[val_idx]

        y_train = df_train.above_average
        y_val = df_val.above_average

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
