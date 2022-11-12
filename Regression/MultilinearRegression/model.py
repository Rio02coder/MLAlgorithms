from sklearn.linear_model import LinearRegression
from Mdataset import x_train, y_train

multilinear_regressor = LinearRegression()
multilinear_regressor.fit(x_train, y_train)