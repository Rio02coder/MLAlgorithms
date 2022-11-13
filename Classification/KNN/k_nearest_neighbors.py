import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Getting dataset and splitting into dependent and independent columns
social_network_ad_dataset = pd.read_csv('Social_Network_Ads.csv')
independent_variables = social_network_ad_dataset.iloc[:,:-1].values
purchased = social_network_ad_dataset.iloc[:,-1].values

# Splitting into training and test sets
x_train, x_test, y_train, y_test = train_test_split(independent_variables, purchased, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model
k_n_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
k_n_model.fit(x_train, y_train)

# Predicting for a specific value
k_n_model.predict(sc.transform([[30, 87000]]))

# Predicting for a list of values
y_pred = k_n_model.predict(x_test)

# Confusion Matrix and Accuracy score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)