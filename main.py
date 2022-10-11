from sklearn.datasets import make_classification
# define dataset 100列：输入特征，1000行：
# n_samples:样本数 n_features:特征总数
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)
# summarize the dataset
print(X, y)