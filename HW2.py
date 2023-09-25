import pandas as pd
import requests
import numpy as np
import random

#version of pandas library
#print(pd.__version__) #2.1.0

#download data .csv file
URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
response = requests.get(URL)
open("housing_hw2.csv", "wb").write(response.content)

df = pd.read_csv('housing_hw2.csv')

#For this homework, we only want to use a subset of data.
#First, keep only the records where ocean_proximity is either '<1H OCEAN' or 'INLAND'
df2 = df[(df['ocean_proximity']=='INLAND') | (df['ocean_proximity']=='<1H OCEAN')] #5 rows
#print(df2)

################################################################################################################################
print('QUESTION 1')
#There's one feature with missing values. What is it?
#Find columns that contain at least one NaN
print("Columns with NaN: ")
print(df2.isnull().sum())
#print("Columns with NaN: ", df2.isnull().any()) 
#Solution: total_bedrooms
################################################################################################################################
print('QUESTION 2')
#What's the median (50% percentile) for variable 'population'?
median_population = df2['population'].median()
print('Median of population: ', median_population) 
#Solution: 1195
################################################################################################################################
#Prepare and split the dataset
#Shuffle the dataset (the filtered one you created above), use seed 42.
np.random.seed(42)
#Split your data in train/val/test sets, with 60%/20%/20% distribution.
n = len(df2)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)
idx = np.arange(n)
np.random.shuffle(idx)
df_shuffled = df2.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()
#Apply the log transformation to the median_house_value variable using the np.log1p() function.
log_trans = np.log1p(df2.median_house_value.values)
################################################################################################################################
print('QUESTION 3')
#We need to deal with missing values for the column from Q1.
#We have two options: fill it with 0 or with the mean of this variable.
#Try both options. For each, train a linear regression model without regularization using the code from the lessons.
#For computing the mean, use the training only!
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

def prepare_X_0(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def prepare_X_mean(df):
    mean_tot_bed = df2['total_bedrooms'].mean()
    df_num = df[base]
    df_num = df_num.fillna(mean_tot_bed)
    X = df_num.values
    return X

base = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
#Use the validation dataset to evaluate the models and compare the RMSE of each option.
#Round the RMSE scores to 2 decimal digits using round(score, 2)
#Which option gives better RMSE?
#OPTION 1
df2_a = df2.fillna(0)
df_shuffled_a = df2_a.iloc[idx]
df_train_a = df_shuffled_a.iloc[:n_train].copy()
df_val_a = df_shuffled_a.iloc[n_train:n_train+n_val].copy()
df_test_a = df_shuffled_a.iloc[n_train+n_val:].copy()
X_train_a = prepare_X_0(df_train_a)
y_train_orig_a = df_train_a.median_house_value.values
y_val_orig_a = df_val_a.median_house_value.values
y_test_orig_a = df_test_a.median_house_value.values
y_train_a = np.log1p(df_train_a.median_house_value.values)
y_val_a = np.log1p(df_val_a.median_house_value.values)
y_test_a = np.log1p(df_test_a.median_house_value.values)
del df_train_a['median_house_value']
del df_val_a['median_house_value']
del df_test_a['median_house_value']
w_0_a, w_a = train_linear_regression(X_train_a, y_train_a)
y_pred_a = w_0_a + X_train_a.dot(w_a)
rmse_a_train = rmse(y_train_a, y_pred_a)
print('rmse (option 1 - train): ', round(rmse_a_train,2)) 
X_val_a = prepare_X_0(df_val_a)
y_pred_a = w_0_a + X_val_a.dot(w_a)
rmse_a_val = rmse(y_val_a, y_pred_a)
print('rmse (option 1 - val): ', round(rmse_a_val,2))

#OPTION 2
mean_population = df2['total_bedrooms'].mean()
df2_b = df2.fillna(mean_population)
X_train_b = df2_b.values
df_shuffled_b = df2_b.iloc[idx]
df_train_b = df_shuffled_b.iloc[:n_train].copy()
df_val_b = df_shuffled_b.iloc[n_train:n_train+n_val].copy()
df_test_b = df_shuffled_b.iloc[n_train+n_val:].copy()
X_train_b = prepare_X_mean(df_train_b)
y_train_orig_b = df_train_b.median_house_value.values
y_val_orig_b = df_val_b.median_house_value.values
y_test_orig_b = df_test_b.median_house_value.values
y_train_b = np.log1p(df_train_b.median_house_value.values)
y_val_b = np.log1p(df_val_b.median_house_value.values)
y_test_b = np.log1p(df_test_b.median_house_value.values)
del df_train_b['median_house_value']
del df_val_b['median_house_value']
del df_test_b['median_house_value']
w_0_b, w_b = train_linear_regression(X_train_b, y_train_b)
y_pred_b = w_0_b + X_train_b.dot(w_b)
rmse_b_train = rmse(y_train_b, y_pred_b)
print('rmse (option 2 - train): ', round(rmse_b_train,2))
X_val_b = prepare_X_mean(df_val_b)
y_pred_b = w_0_b + X_val_b.dot(w_b)
rmse_b_val = rmse(y_val_b, y_pred_b)
print('rmse (option 2 - val): ', round(rmse_b_val,2))

#Solution: 
#rmse (option 1 - train):  0.34
#rmse (option 2 - train):  0.34
#rmse (option 1 - val):  0.34
#rmse (option 2 - val):  0.34


################################################################################################################################
print('QUESTION 4')
#Now let's train a regularized linear regression.
#For this question, fill the NAs with 0.
#Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
#Use RMSE to evaluate the model on the validation dataset.
#Round the RMSE scores to 2 decimal digits.
#Which r gives the best RMSE?
#If there are multiple options, select the smallest r.
def train_linear_regression_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0_a, w_a = train_linear_regression_reg(X_train_a, y_train_a, r)
    y_pred_a = w_0_a + X_val_a.dot(w_a)
    print('%6s' %r, rmse(y_val_a, y_pred_a))
#Solution
#0 0.3408479034133711***the smallest r
# 1e-06 0.3408479061803642
#0.0001 0.3408481800530103
# 0.001 0.340850692187126
#  0.01 0.34087793004933414
#   0.1 0.34128620420007866
#     1 0.34489583276493896
#     5 0.34773980704848945
#    10 0.34831498335199945

################################################################################################################################
print('QUESTION 5')
#We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
#Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
#For each seed, do the train/validation/test split with 60%/20%/20% distribution.
#Fill the missing values with 0 and train a model without regularization.
#For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
#What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
#Round the result to 3 decimal digits (round(std, 3))
#What's the value of std?

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    np.random.seed(s)
    w_0_a, w_a = train_linear_regression(X_train_a, y_train_a)
    y_pred_a = w_0_a + X_val_a.dot(w_a)
    rmse_f = rmse(y_val_a, y_pred_a)
    print('%6s' %s, rmse_f)

std = np.std(rmse_f)
std = round(std, 3)
print('Standard deviation of all the scores: ', std)
#Solution:
#     0 0.3408479034133711
#     1 0.3408479034133711
#     2 0.3408479034133711
#     3 0.3408479034133711
#     4 0.3408479034133711
#     5 0.3408479034133711
#     6 0.3408479034133711
#     7 0.3408479034133711
#     8 0.3408479034133711
#     9 0.3408479034133711
#Standard deviation of all the scores:  0.0


################################################################################################################################
print('QUESTION 6')
#Split the dataset like previously, use seed 9.
#Combine train and validation datasets.
#Fill the missing values with 0 and train a model with r=0.001.
#What's the RMSE on the test dataset?
np.random.seed(9)
w_0, w = train_linear_regression_reg(X_train_a, y_train_a, r=0.001)

y_pred_a = w_0 + X_val_a.dot(w)
print('validation:', rmse(y_val_a, y_pred_a))

X_test_a = prepare_X_0(df_test)
y_pred_a = w_0_a + X_test_a.dot(w)
print('test:', rmse(y_test_a, y_pred_a))
#Solution: 
#validation: 0.340850692187126
#test: 0.33109421874687023
