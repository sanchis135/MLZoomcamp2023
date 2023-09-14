import pandas as pd
import requests
import numpy

#QUESTION 1
#version of pandas library
print(pd.__version__) #2.1.0

#download data .csv file
URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
response = requests.get(URL)
open("housing.csv", "wb").write(response.content)

df = pd.read_csv('housing.csv')
print(df)

#QUESTION 2
# computing number of rows
rows = len(df.axes[0])
print("Number of Rows: ", rows) #20640 rows
# computing number of columns
cols = len(df.axes[1])
print("Number of Columns: ", cols) #10 columns

#QUESTION 3
#check if each element is a missing value or not
#print(df.isnull())
#find rows with NaN in a specific column
#print(df[df['total_rooms'].isnull()])
#find columns with NaN in a specific row
#print(df.iloc[2].isnull())
#print(df.loc[:, df.iloc[2].isnull()])
#Find columns that contain at least one NaN
print("Columns with NaN: ", df.isnull().any()) #total_bedrooms
#print(df.loc[:, df.isnull().any()])

#QUESTION 4
print("Unique values in ocean_proximity: ", df["ocean_proximity"].unique()) #5

#QUESTION 5
print('Average of median_house_value (near the bay): ', df.groupby('ocean_proximity').mean()[['median_house_value']]) #'NEAR_BAY' 259212.311790

#QUESTION 6
mean_total_bedrooms = df['total_bedrooms'].mean()
print('Average of total_bedrooms', mean_total_bedrooms)
df2 = df.fillna(value={'total_bedrooms': mean_total_bedrooms})
print('New average of total_bedrooms', df2['total_bedrooms'].mean()) #No

#QUESTION 7
df2 = df.loc[df['ocean_proximity']=='ISLAND'] #5 rows
df3 = df2[["housing_median_age","total_rooms","total_bedrooms"]]
X = df3.to_numpy()
XTX = numpy.dot(X.T,X)
inverse = numpy.linalg.inv(XTX)
y = [[950, 1300, 800, 1000, 1300]]
w = numpy.dot(inverse, XTX)
#w=numpy.dot(w,y)
print('The result of w is: ', w)
