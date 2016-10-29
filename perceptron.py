import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class Perceptron(object):
	"""
		Perceptron classifier.

		Parameters
		----------
		eta : float - Learning rate (0.0 <= eta <= 1.0)
		n_iter : int - Passes over the training dataset.

		Attributes
		__________
		w_ : 1d-array - Weights after fitting.
		errors_ : list - Number of misclassifications in every epoch
	"""
	def __init__(self, eta=0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		"""
		:param X: array-like, shape = [n_samples, n_features]
				Training vectors, where n_samples is the number of samples
				and n_features is the number of features.
		:param y: array-like, shape = [n_samples] - Target values
		:return: self : object
		"""
		self.w_ = np.zeros(1 + X.shape[1])  # n_features + 1
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):  # for each (sample, result)
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		"""
		Calculate net input
		:param X: array-like - List of features
		:return: array-like - net input sent to perceptron
		"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		"""
		Return class label after unit step
		:param X: array-like - List of features
		:return: int - Class label prediction
		"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)


def import_data():
	"""
	Read data (sepal length, petal length, and classification) from the Iris dataset and extract the first 100 class
	labels that correspond to the 50 Iris-Setosa and 50 Iris-Versicolor flowers. Assign data to a feature matrix X, and
	a results vector, y.
	:return:
	"""
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	y = df.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)
	X = df.iloc[0:100, [0, 2]].values
	return X, y

def plot_decision_regions(X, y, classifier, resolution=0.02):
	"""
	visualize the decision boundaries for 2D datasets
	:param X: array-like, shape = [n_samples, n_features]
				Training vectors, where n_samples is the number of samples
				and n_features is the number of features.
	:param y: array-like - Target values
	:param classifier: agent used to classify data
	:param resolution: Spacing between values (used for numPy arange function)
	:return: None
	"""
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

if __name__ == "__main__":
	X, y = import_data()
	ppn = Perceptron(eta=0.1, n_iter=10)
	ppn.fit(X, y)
	plot_decision_regions(X, y, classifier=ppn)
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.legend(loc='upper left')
	plt.show()