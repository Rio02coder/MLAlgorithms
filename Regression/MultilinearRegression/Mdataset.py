import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Splitting Dataset into dependent and independent variables
company_profit_dataset = pd.read_csv('50_Startups.csv')
x_values = company_profit_dataset.iloc[:,:-1].values
profit = company_profit_dataset.iloc[:,-1].values

# Categorical Data Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x_values = np.array(ct.fit_transform(x_values))

# Splitting Dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_values, profit, test_size=0.2, random_state=0)
