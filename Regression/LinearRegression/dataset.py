import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Splitting Dataset into dependent and independent variables
salary_dataset = pd.read_csv('Salary_Data.csv')
experience = salary_dataset.iloc[:,:-1].values
salary = salary_dataset.iloc[:,-1].values

# Splitting Dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(experience, salary, test_size=0.2, random_state=0)