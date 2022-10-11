# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 解决一个二元分类问题(糖尿病的发病为1或者0)

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input(X) and output(y) variables
X, y = dataset[:, :-1], dataset[:, -1]
print(X.shape, y.shape)

# define the keras model
# 输入层为具有8个变量的数据行
# 第一个隐藏层有12个节点，使用ReLu激活函数
# 第二个隐藏层具有8个节点，使用ReLu激活函数
# 输出层有一个节点，使用sigmoid激活函数
model = Sequential()
# 这里最令人困惑的是模型输入的形状被定义为第一个隐藏层的参数。这意味着添加第一个 Dense 层的代码行做两件事，定义输入层和第一个隐藏层。
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model,针对二元分类问题，采用交叉熵作为损失函数，优化器选择adam-自动梯度下降算法，收集分类准确度的矩阵参数
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

