from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
  def __init__(self):
    self.weights = None

  @abstractmethod
  def prepare_data(self, X: np.ndarray):
    pass

  @abstractmethod
  def compute_grad(self, X: np.ndarray, input_value: np.ndarray, input_grad: np.ndarray):
    pass

  @abstractmethod
  def apply(self, X: np.ndarray):
    pass

  def predict(self, X: np.ndarray):
    X = self.prepare_data(X)
    return self.apply(X)
  
class CustomLinearRegression(Model):
  def __init__(self, num_features: int):
    self.weights = np.random.normal(0, 1 / np.sqrt(num_features + 1), num_features + 1)

  def prepare_data(self, X: np.ndarray):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)

  def compute_grad(self, X: np.ndarray, input_value: np.ndarray, input_grad: np.ndarray):
    batch_size = X.shape[0]
    return X.T @ input_grad

  def apply(self, X: np.ndarray):
    return X @ self.weights
  
class LogisticRegressionBC(Model):
  def __init__(self, num_features: int):
    self.weights = np.random.rand(num_features + 1)

  def prepare_data(self, X: np.ndarray):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)

  def _sigmoid(self, x: np.ndarray):
    return 1 / (1 + np.exp(-x))

  def compute_grad(self, X: np.ndarray, input_value: np.ndarray, input_grad: np.ndarray):
    dpred_dxw = input_value * (1 - input_value)
    return X.T @ (dpred_dxw * input_grad)

  def apply(self, X: np.ndarray):
    return self._sigmoid(X @ self.weights)
