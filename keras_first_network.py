# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 解决一个二元分类问题(糖尿病的发病为1或者0)

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input(X) and output(y) variables
X = dataset[:,0:8]
print(X)
