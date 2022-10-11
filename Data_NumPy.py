# one-dimensional example
from numpy import array
from numpy import reshape
# list of data
data = [11, 22, 33, 44, 55]
# array of data
data = array(data)
print(data)
print(type(data))

# two-dimensional example
# list of data
data = [[11, 22],
        [33, 44],
        [55, 66]]
# array of data
data = array(data)
print(data)
print(type(data))

# simple indexing
data = array([11, 22, 33, 44, 55])
# index data
# print(data[5])
print(data[-1]) # index -1 will print last item in array
print(data[-5])

# 2d indexing
data = array([[11, 22], [33, 44], [55, 66]])
# index data
print(data[0,0])
print(data)
print(data[0,]) # second parameter set 0 will show all first line result
print("-----")
# simple slicing data[from:to]
data = array([11, 22, 33, 44, 55])
print(data[:]) # access all data in an array dimension
print(data[0:1]) # contains the first index, not contain the second index
print(data[-2:]) # Negative index represent the last number to be counted
print("-----")
# split input and output 按照列拆分
# For the input features, we can select all rows and all columns except the last one by specifying ‘:’ for in the rows
# index, and :-1 in the columns index.
# X = [:,:-1]
# For the output column, we can select all rows again using ‘:’ and index just the last column
# by specifying the -1 index.
# y = [:,-1]
data = array([[11, 22, 33],
              [44, 55, 66],
              [77, 88, 99]])
# separate data
X, y = data[:, :-1], data[:, -1]
print(X)
print(y)
print("-----")
# 拆分数据集为训练集以及测试集一般是按照行去拆分，通过在二维数组指定分割点进行分割
# train = data[:split, :]
# test = data[split:, :]
data = array([[11, 22, 33],
              [44, 55, 66],
              [77, 88, 99]])
# separate data
split = 2
train, test = data[:split, :], data[split:, :]
print(train)
print("-----")
print(test)
print("----------")
# data shape
data = array([11, 22, 33, 44, 55])
print(data.shape)
data = array([[11, 22],
              [33, 44],
              [55, 66]])
print(data.shape) # first index represent Rows, second index represent Cols
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])
print("----------")
# data reshape 将1d数组重塑为为一列多行的2d数组
# data = data.reshape((data.shape[0], 1))
data = array([11, 22, 33, 44, 55])
print(data.shape)
print(data)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)
print(data)
print("----------")
# Reshape 2D to 3D Array
# data.reshape((data.shape[0], data.shape[1], 1))
data = array([[11, 22],
              [33, 44],
              [55, 66]])
print(data.shape)
print(data)
data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)
print(data)









