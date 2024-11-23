# LinearModels
Implementation of Linear Regression model and Logistic Regression and some modern optimizers like Adam. Comparing result with sklearn implementation.


## Installation

To run code from this repository run command:

```bash
    pip install -r requirements.txt
```

## Code structure 

### Optimizers

There is a base abstract class from optimizers in file [optimizers](optimizers.py) with implementations for simple Constantoptimizer, AdaGrad and Adam optimizers

```python
class BaseOptimizer(ABC):
  def __init__(self):
    self.model_params = None

  def set_learning_params(self, params: np.array):
    # set parameters to train
    self.model_params = params

  @abstractmethod
  def step(self, grad: np.ndarray):
    # make step with updating model weights and self state
    pass
```

### Loss function 

There is base abstract class from loss functions in file [loss](loss.py) with option to compute gradient by preditions. Also there are implementations of MeanSquaredLoss and CrossEntropyLoss.

```python
class LossFunction(ABC):
  @abstractmethod
  def compute(self, y_predicted, y_target):
    # computes loss between prediction and target
    pass

  @abstractmethod
  def compute_grad(self, y_predicted, y_target):
    #computes gradient of loss between prediction and target by prediction
    pass
```

### Model 

There is base abstract class from loss functions in file [models](models.py) with option to compute gradient of preditions by weights and combine it from input gradient of loss function. Also there are implementations of LinearRegression and LogisticRegression for binary classification.

```python
class Model(ABC):
  def __init__(self):
    self.weights = None

  @abstractmethod
  def prepare_data(self, X: np.ndarray):
    # prepares data X to apply model
    # needed to reduce compute by training
    pass

  @abstractmethod
  def compute_grad(self, X: np.ndarray, input_value: np.ndarray, input_grad: np.ndarray):
    # computes gradient of f(self(X)) by X  where:
    # input_value: value of self(X)
    # input_grad: gradient of f by self(X)
    pass

  @abstractmethod
  def apply(self, X: np.ndarray):
    # apply model after preparing
    pass

  def predict(self, X: np.ndarray):
    # get prediction
    X = self.prepare_data(X)
    return self.apply(X)
```

### Algorithm

There is an implementation of Stochastic Gradient Descent [sgd](algorithm.py). 

```python 
def SGD(X: np.ndarray, y: np.ndarray, model: Model, loss_function: LossFunction = MSE(),
        opt:BaseOptimizer=ConstantLROptimizer(1e-3), batch_size:int=None,
        n_epochs:int=100)
```

This implementation is flexible. To optimize your on model with your own loss function you only need to implement computation of model and loss evaluating and computing gradient by input.

## Simple Test

In [simple test](simple_test.ipynb) shown easy custom tests to debug. Where models are trying to predict linear function which is using to build target.

## Experiments with real data

In [experiments](experiments.ipynb) shown comparing our implementation with implementation from `sklearn`.

### Our Linear Regression vs Sklearn Linear Regression

Comparing was on Real estate (dataset)[Realestate.csv]. 

Data was splitted on train-test and for preprocessing was used `StandartScaler` from `sklearn.preprocessing`.

| Metric        |  Our lm    | Sklearn lm |
| ------ | ----------- |   -----------|
| MSE       | 64.4    | 70.7 |
| R2 score  | 0.639   | 0.639 |

### Our Linear Regression vs Sklearn Linear Regression

Comparing was on sklearn dataset [breast cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

Data was splitted on train-test and for preprocessing was used `StandartScaler` from `sklearn.preprocessing`.

| Metric        |  Our lm    | Sklearn lm |
| ------ | ----------- |   -----------|
| Acuracy       | 98.25   | 98.25 |

Also in [experiments](experiments.ipynb) there are loss function plots and roc curve from classification.

## Conclusion

According to results of experiments our method in comparable with sklearn methods and not loosing them by metrics.