import numpy as np

from models import Model
from loss import LossFunction, MSE
from optimizers import BaseOptimizer, ConstantLROptimizer

from copy import deepcopy

def SGD(X: np.ndarray, y: np.ndarray, model: Model, loss_function: LossFunction = MSE(),
        opt:BaseOptimizer=ConstantLROptimizer(1e-3), batch_size:int=None,
        n_epochs:int=100):
  X_prepared = model.prepare_data(X)
  y_prepared = deepcopy(y)
  n = len(y)

  loss_history = []

  if batch_size is None:
    batch_size = n

  for _ in range(n_epochs):
    shuffle_ids = np.arange(n)
    if batch_size != n:
      np.random.shuffle(shuffle_ids)
    X_prepared, y_prepared = X_prepared[shuffle_ids], y_prepared[shuffle_ids]

    for start_id in range(0, n, batch_size):
      X_batch = X_prepared[start_id : start_id + batch_size]
      y_batch = y_prepared[start_id : start_id + batch_size]

      y_predict = model.apply(X_batch)
      loss = loss_function.compute(y_predict, y_batch)
      loss_grad = loss_function.compute_grad(y_predict, y_batch)
      grad = model.compute_grad(X_batch, y_predict, loss_grad)

      opt.step(grad)

      loss_history.append(loss)

  return loss_history