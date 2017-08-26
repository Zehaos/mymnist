from operator import Op
import numpy as np

class Fullyconnected(Op):
    def __init__(self, in_num, out_num, name):
        self._name = name
        self._params = np.random.normal(0, 0.001, (in_num, out_num))

    def forward(self, in_data):
        out_data = np.dot(in_data, self.params)
        self.input = in_data
        return out_data

    def backward(self, out_grad, lr, wd):
        in_grad = np.dot( out_grad, self.params.T)
        self._params -= lr*(np.dot(self.input.T, out_grad) + wd*self._params)
        return in_grad

