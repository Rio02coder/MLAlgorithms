import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

social_network_ads_dataset = pd.read_csv('Social_Network_Ads.csv')
independent_variables = social_network_ads_dataset.iloc[:, :-1].values
purchased = social_network_ads_dataset.iloc[:, -1].values

# Splitting the data into test and train set
x_train, x_test, y_train, y_test = train_test_split(independent_variables, purchased, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting for a list of values
y_pred = classifier.predict(x_test)

# Creating the confusion matrix and the accuracy of the model
matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
