from operator import Op
import numpy as np

class SimpleBatchNorm(Op):
    def __init__(self, name, istraining=False):
        self._name = name
        self._params = []

        self._mean = 0
        self._var = 1.0
        self._momentum = 0.9
        self._istraining = istraining

    def forward(self, in_data):
        mean = np.mean(in_data, axis=0)
        var = np.var(in_data, axis=0)
        if self._istraining:
            self._mean = self._mean * self._momentum + mean * (1 - self._momentum)
            self._var  = self._var * self._momentum + var * (1 - self._momentum)
            out_data = (in_data - self._mean) / self._var
        else:
            out_data = (in_data - self._mean) / self._var

        return out_data

    def backward(self, out_grad, lr, wd):
        in_grad = out_grad * self._var + self._mean
        return in_grad

    def set_params(self, params):
        self._params = params
        self._mean = params[0]
        self._var = params[1]

    @property
    def params(self):
        return np.array([self._mean, self._var])
