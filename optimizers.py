from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
  def __init__(self):
    self.model_params = None

  def set_learning_params(self, params: np.array):
    self.model_params = params

  @abstractmethod
  def step(self, grad: np.ndarray):
    pass

class ConstantLROptimizer(BaseOptimizer):
    def __init__(self, lr:float = 0.001):
        super().__init__()
        self.lr = lr

    def step(self, grad: np.ndarray):
        self.model_params -= self.lr * grad

class Adam(BaseOptimizer):
  def __init__(self, lr: float=0.001, beta1: float = 0.9, beta2: float = 0.999, lmbd: float=0.0, epsilon: float = 1e-8):
    super().__init__()
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.smoothing_mean = 0
    self.smoothing_std = 0
    self.t = 0
    self.beta1_t = self.beta1
    self.beta2_t = self.beta2
    self.lmbd = lmbd
    self.epsilon = epsilon

  def step(self, grad: np.ndarray):
    if self.lmbd > 0:
      grad += self.lmbd * self.model_params

    self.smoothing_mean = self.beta1 * self.smoothing_mean + \
                          (1 - self.beta1) * grad
    self.smoothing_std = self.beta2 * self.smoothing_std + \
                         (1 - self.beta2) * grad ** 2

    m_hat = self.smoothing_mean / (1 - self.beta1_t)
    v_hat = self.smoothing_std / (1 - self.beta2_t)

    self.model_params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    self.beta1_t *= self.beta1
    self.beta2_t *= self.beta2

class AdaGrad(BaseOptimizer):
  def __init__(self, lr: float=0.05, eta: float = 0.01, lmbd: float = 0.0, 
               initial_accumulator_value: float = 0.0, epsilon: float = 1e-8):
    super().__init__()
    self.lr = lr
    self.eta = eta
    self.lmbd = lmbd
    self.state_sum = initial_accumulator_value
    self.epsilon = epsilon
    self.t = 0

  def step(self, grad: np.ndarray):
    if self.lmbd > 0:
      grad += self.lmbd * self.model_params

    lr = self.lr / (1 + self.t * self.eta)

    self.state_sum += grad ** 2

    self.model_params -= lr * grad / (np.sqrt(self.state_sum) + self.epsilon)

    self.t += 1