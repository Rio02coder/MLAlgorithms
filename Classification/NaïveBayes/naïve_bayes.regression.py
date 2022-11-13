import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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

# Model Training
gnb_classifier = GaussianNB()
gnb_classifier.fit(x_train, y_train)

# Predicting for a specific value
gnb_classifier.predict(sc.transform([[30, 87000]]))

# Predicting for a list of values
y_pred = gnb_classifier.predict(x_test)

# Confusion matrix and Accuracy score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)