import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/kiran/marketing_campaign.csv",sep='\t')

df.head()

df['Income']=df['Income'].fillna(df['Income'].median())

df.isna().any()

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Kids"
print("Unique catego

df=df.drop(columns=["Z_CostContact", "Z_Revenue"],axis=1)

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Year_Birth"
print("Unique categories present in the Year_Birth:",df["Year_Birth"].value_counts())

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Year_Birth"
print("Unique categories present in the Year_Birth:",df["Year_Birth"].value_counts())

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Year_Birth"
print("Unique categories present in the Year_Birth:",df["Year_Birth"].value_counts())

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Year_Birth"
print("Unique categories present in the Year_Birth:",df["Year_Birth"].value_counts())

df['Marital_Status'].unique()

#REPLACING THE CONFLICT VALUES IN Marital_status..
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Marital_Status"
print("Unique categories present in the Marital_Status:",df['Marital_Status'].value_counts())
print("\n")


#Maximum Income
df['Income'].max()

#Minimum Income
df['Income'].min()

#AverageIncome
df['Income'].mean()

df['Kidhome'].unique()

df['Teenhome'].unique()

# Combining different dataframe into a single column to reduce the number of dimension

df['Kids'] = df['Kidhome'] + df['Teenhome']

print (df['Kids'])

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Kids"
print("Unique categories present in the Kids:",df['Kids'].value_counts())
print("\n")

df['MntWines'].unique()

df['MntFruits'].unique()

df['MntMeatProducts'].unique()

df['MntFishProducts'].unique()

df['MntSweetProducts'].unique()

df['MntGoldProds'].unique()

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Kids"
print("Unique catego

# Combining different dataframe into a single column to reduce the number of dimension

df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['Expenses'].head(10)

#Minimum Expenses
df['Expenses'].min()

#Maximum Expenses
df['Expenses'].max()

#Average Expenses
df['Expenses'].mean()

df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']

#ptedCmp
print("Unique categories present in the TotalAcceptedCmp:",df['TotalAcceptedCmp'].value_counts())
print("\n")

df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']
df['NumTotalPurchases'].unique()

#Minimum NumTotalPurchases
df['NumTotalPurchases'].min()

#Maximum NumTotalPurchases
df['NumTotalPurchases'].max()

#Mean NumTotalPurchases
df['NumTotalPurchases'].mean()

# Deleting some column to reduce dimension and complexity of model

col_del = ["ID","AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df=df.drop(columns=col_del,axis=1)
df.head(10)

x = df.columns 
for i in x:
     print(i)



#NUMBER OF DAYS A CUSTOMER WAS ENGAGED WITH COMPANY...... 
#CHANGING "Dt_customer" into timestamp format......

df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['First_day'] = '01-01-2015'
df['First_day'] = pd.to_datetime(df.First_day)
df['Day_engaged'] = (df['First_day'] - df['Dt_Customer']).dt.days

#ADDING A COLUMN "Age" IN THE DATAFRAME....
df['Age'] = (pd.Timestamp('now').year) - (pd.to_datetime(df['Dt_Customer']).dt.year)

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Age"
print("Unique categories present in the Age:",df['Age'].value_counts())
print("\n")

df.head(5)

df=df.drop(columns=["Dt_Customer", "First_day", "Year_Birth", "Dt_Customer", "Recency", "Complain","Response"],axis=1)
df.head(5).style.set_properties(**{"background-color": "#fffc99","color": "black", "border-color": "black"})

#REARRANGE THE ORDER OF COLUMNS:-
order = [0,1,3,4,6,7,8,2,5]
df = df[[df.columns[i] for i in order]]
df.head(5)

df.shape

pd.crosstab(df['Education'],df['Expenses'],margins=True)





df.head(5)

pd.crosstab(df['TotalAcceptedCmp'],df['Expenses'],margins=True)

pd.crosstab(df['NumTotalPurchases'],df['Expenses'],margins=True)

pd.crosstab(df['Day_engaged'],df['Expenses'],margins=True).head(10)

pd.crosstab(df['Age'],df['Expenses'],margins=True)

df.head()

df.describe(include = 'all')

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,cmap = 'YlGnBu',linewidths=1)

X = df.iloc[:, [7, 3]].values
print(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 18):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 50)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 18), wcss,color = "mediumblue",marker = '*')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 50)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.figure(figsize= (15,8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 25, c = 'mediumblue', label = 'Sensible Customers')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 25, c = 'lawngreen', label = 'Careless Customers')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 55, c = 'black', label = 'Centroids')
plt.title('Clusters of customers',fontsize = 20)
plt.xlabel('Income',fontsize = 15)
plt.ylabel('Expenses',fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

