"""
Please, note that some parts of the code are commented out, that is due to the fact that not everything is supposed to run at once.  
"""


#Importing libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import zscore

#Importing and reading the dataset
df = pd.read_csv('C:/Users/bogoi/Desktop/Computing/Year 3/Emerging Technologies - SWE6206/Assignments/Assignment 2/GoldUP.csv')

#Analysing the dataset 
#print(df.head())
#print(df.info())
#print(df.shape)
#print(df.columns)
#print(df.describe())



#Check for duplicates, if any and remove them
#temp_df = df
#print("Number of rows and columns before removing the duplicates: " , temp_df.shape)
#temp_df.drop_duplicates(inplace=True, keep='first')
#print("Number of rows and columns after removing the duplicates: " , temp_df.shape)

#Check and deal with missing values
#print(df.isnull().sum())

#Visualise data
#sns.pairplot(data=df, diag_kind='kde')
#plt.show()


#Outlier Analysis
#temp_df = df
#plt.boxplot(temp_df['Gold_Price'])
#plt.show()



#Check the correlation of columns
#plt.figure(figsize=(12,10))
#sns.heatmap(df.corr(), annot=True)
#plt.show()
#corr_matrix=df.corr()
#print(corr_matrix["Gold_Price"].sort_values(ascending=False))



#Feature Selection 
y=df['Gold_Price']
X=df[['CPI', 'Sensex', 'USD_INR']]


#Split data into test and train
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#Scaling numerical features using sklearn StandardScaler(Preprocessing)
numeric=['CPI', 'Sensex', 'USD_INR']
sc=StandardScaler()
X_train[numeric]=sc.fit_transform(X_train[numeric])
X_test[numeric]=sc.transform(X_test[numeric])

#Create a Multivariate LinearRegression object
lr= LinearRegression()
#Fit X and y 
lr.fit(X_train, y_train)
ypred = lr.predict(X_test)
#Metrics to evaluate the models


print('Mean Absolute Error(LR): ', mean_absolute_error(y_test, ypred))
print('Root Mean Squared Error(LR): ', np.sqrt(mean_squared_error(y_test, ypred)))
acurracyScore=r2_score(y_test, ypred)
acurracyScore="{:.0%}".format(acurracyScore)
print('R2 Score(LR): ' + str(acurracyScore))

plt.title("Distribution of Real vs. Predicted Values (Multivariate Linear Regression)")
ax1=sns.distplot(y_test, hist=False)
sns.distplot(ypred, hist=False, ax=ax1)
plt.show()

