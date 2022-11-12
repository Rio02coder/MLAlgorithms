import matplotlib.pyplot as plt
from dataset import x_train, x_test, y_train, y_test
from model import linear_regression_model

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, linear_regression_model.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()