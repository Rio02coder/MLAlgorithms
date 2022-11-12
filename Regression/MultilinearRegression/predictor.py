import numpy as np
from Mdataset import x_test, y_test
from model import multilinear_regressor

y_pred = multilinear_regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))