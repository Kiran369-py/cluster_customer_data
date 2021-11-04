#import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#read data and preprocessing

customer_data = pd.read_csv("C:/kiran/marketing_campaign.csv",sep='\t')

customer_data.head()

customer_data.isna().sum()

#filling the Nan values
customer_data.columns

customer_data.rename(columns={' Income ':'Income'}, inplace=True)
customer_data['Income'].apply(type).value_counts()
def clean_currency(x):
 if isinstance(x, str):
        return(x.replace('$','').replace(',',''))
 return(x)
customer_data['Income'] = customer_data['Income'].apply(clean_currency).astype(float)

customer_data['Income'].apply(type).value_counts()

customer_data.dtypes

customer_data['Income'].head()

customer_data_dropped = customer_data.copy()

customer_data_dropped.dropna(inplace = True)

customer_data_dropped.isna().any()

customer_data.fillna(2240 ,inplace=True)

customer_data['Dt_Customer']=pd.to_datetime(customer_data['Dt_Customer'])

#martial status vs producct type histogram

Martial_product_hist = px.histogram(customer_data, x='Marital_Status', y=['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts'],color_discrete_sequence=px.colors.sequential.Turbo)

Martial_product_hist.update_layout(
    title_x=0.5,
    title_font_color='rgb(42, 1, 52)',
    xaxis_title="Marital Status",
    yaxis_title="Products Count",
    font=dict(
        family="Times New Roman",
        size=18,
        color="purple"
    ))

#education pie chart

Education = px.pie(customer_data,names='Education',color_discrete_sequence=px.colors.sequential.Turbo)

Education.update_layout(
    title_x=0.5,
    title_font_color='rgb(42, 1, 52)',
    margin=dict(l=0, r=0, t=60, b=110),
    font=dict(
        family="Times New Roman",
        size=18,
        color="Purple"
    ))

#years histogram

Years = px.histogram(x = customer_data['Dt_Customer'],color = customer_data.Dt_Customer.dt.year,nbins=50,color_discrete_sequence=px.colors.sequential.Turbo)

Years.update_layout(
    title_x=0.5,
    title_font_color='rgb(42, 1, 52)',
    xaxis_title="Date",
    yaxis_title="Enrollment Count",
    font=dict(
        family="Times New Roman",
        size=18,
        color="Purple"
    ))

#Martial status pie chart

Marital_Status = px.pie(customer_data,names='Marital_Status',color_discrete_sequence=px.colors.sequential.Turbo)
    

Marital_Status.update_layout(
    title_x=0.5,
    title_font_color='rgb(42, 1, 52)',
    margin=dict(l=0, r=0, t=60, b=110),
    font=dict(
        family="Times New Roman",
        size=18,
        color="RebeccaPurple"
    ))

#Number of Kids pie chart

Kidhome = px.pie(customer_data,names='Kidhome',color_discrete_sequence=px.colors.sequential.Turbo)

Kidhome.update_layout(
    title_x=0.5,
    title_font_color='rgb(42, 1, 52)',
    margin=dict(l=0, r=0, t=60, b=110),
    font=dict(
        family="Times New Roman",
        size=18,
    ))

#Checking catogarical unique values


customer_data['Education'].unique()

customer_data['Marital_Status'].unique()

# calculating the age of customers 
customer_data['c_year'] = 2021
customer_data['age'] = customer_data['c_year']-customer_data['Year_Birth']

# calculating the no of days as customers
customer_data['c_date'] = '01-01-2015'
customer_data['Dt_Customer'] =pd.to_datetime(customer_data.Dt_Customer)
customer_data['c_date'] = pd.to_datetime(customer_data.c_date)
customer_data['days_customer'] = (customer_data['c_date']-customer_data['Dt_Customer']).dt.days

# reducing some dimensions 

# defining two categories of expenses as on food ( fish+meat+ fruit) and as leisure_Expense ( wine, sweet, Gold)
customer_data['leisure_Expense'] = customer_data['MntWines']+ customer_data['MntSweetProducts']+ customer_data['MntGoldProds']
customer_data['food'] = customer_data['MntFishProducts'] + customer_data['MntFruits'] + customer_data['MntMeatProducts']


# defining accepted_any_cmp if customer have ever taken any campaign offer ( 1- yes, 0-no)
customer_data['accepted_any_cmp']= customer_data['AcceptedCmp1']+customer_data['AcceptedCmp2']+customer_data['AcceptedCmp3']+customer_data['AcceptedCmp4']+customer_data['AcceptedCmp5']+customer_data['Response']
customer_data['accepted_any_cmp']= np.where(customer_data['accepted_any_cmp'] > 0, 1, 0)

# defining martial status as ( 0- single adult, 1- two adults)
mapping = {'Single' : 0, 'Together': 1, 'Married': 1, 'Divorced': 0, 'Widow': 0, 'Alone': 0,
       'Absurd': 0, 'YOLO': 0, 'Graduation': 1, 'PhD': 2, 'Master': 2, 'Basic': 0, '2n Cycle': 2}
customer_data=customer_data.replace({'Marital_Status': mapping, 'Education': mapping}) 

customer_data['kid_teen'] = customer_data['Kidhome'] + customer_data['Teenhome']
customer_data['kid_teen']= np.where(customer_data['kid_teen'] > 0, 1, 0)

customer_data.columns

#dropping some unwanted columns
customer_data=customer_data[['Education', 'Marital_Status', 'Income', 'kid_teen',
       'Recency','leisure_Expense', 'food', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'accepted_any_cmp', 
       'Complain', 'age', 'days_customer']]

#applying feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(customer_data)

#Finding optimal number of clusters using Elbow method

from sklearn.cluster import KMeans
wcss = []
for i in range(1,16):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    

plt.plot(range(1,16), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

#To plot a 2D graph we will use PCA to convert the dataset in 2 components

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#fitting k means
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#pring seggmented custom values
print(y_kmeans)

#plotting clusters of customers

plt.scatter(X[y_kmeans ==0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label='cluster1')
plt.scatter(X[y_kmeans ==1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label='cluster2')
plt.scatter(X[y_kmeans ==2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label='cluster3')
plt.scatter(X[y_kmeans ==3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label='cluster4')
plt.scatter(X[y_kmeans ==4, 0], X[y_kmeans == 4, 1], s = 100, c = 'gray', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'centroids')
plt.title('The clusters of Customers')
plt.xlabel('Component1')
plt.ylabel('Component2')
plt.legend()
plt.show()