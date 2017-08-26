from operator import Op
import numpy as np

class Relu(Op):
    def __init__(self, name):
        self._name = name
        self._params = []

    def forward(self, in_data):
        out_data = np.maximum(0.0, in_data)
        self.in_data = in_data
        return out_data

    def backward(self, out_grad, lr, wd):
        in_grad = np.zeros_like(self.in_data)
        in_grad[np.where(self.in_data>0)] = 1.0
        return in_grad*out_grad
