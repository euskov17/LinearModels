from abc import ABC, abstractmethod
import numpy as np

EPSILON = 1e-8

class LossFunction(ABC):
  @abstractmethod
  def compute(self, y_predicted, y_target):
    pass

  @abstractmethod
  def compute_grad(self, y_predicted, y_target):
    pass


class MSE(LossFunction):
  def compute(self, y_predicted, y_target):
    error = (y_predicted - y_target)
    return np.mean((y_predicted - y_target) ** 2)

  def compute_grad(self, y_predicted, y_target):
    return 2 / len(y_target) * (y_predicted - y_target)



class CrossEntropyLoss(LossFunction):
  def compute(self, y_predicted, y_target):
    return - np.mean(y_target * np.log(y_predicted + EPSILON) + \
                     (1 - y_target) * np.log(1 + EPSILON - y_predicted))

  def compute_grad(self, y_predicted, y_target):
    return (y_predicted - y_target) / (y_predicted * (1 - y_predicted) * len(y_target) + EPSILON)