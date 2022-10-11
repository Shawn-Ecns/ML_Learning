# first neural network with keras tutorial
import numpy

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import os
numpy.random.seed(7)

# 解决一个二元分类问题(糖尿病的发病为1或者0)

# load the dataset
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input(X) and output(y) variables
X, y = dataset[:, :-1], dataset[:, -1]
# print(X.shape, y.shape)

# define the keras model
# 输入层为具有8个变量的数据行
# 第一个隐藏层有12个节点，使用ReLu激活函数
# 第二个隐藏层具有8个节点，使用ReLu激活函数
# 输出层有一个节点，使用sigmoid激活函数
model = Sequential()
# 这里最令人困惑的是模型输入的形状被定义为第一个隐藏层的参数。这意味着添加第一个 Dense 层的代码行做两件事，定义输入层和第一个隐藏层。
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model,针对二元分类问题，采用交叉熵作为损失函数，优化器选择adam-自动梯度下降算法，收集分类准确度的矩阵参数
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
scores = model.evaluate(X, y)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# later
# load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, y)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
