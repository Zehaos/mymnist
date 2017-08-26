from operator import Op
import numpy as np

class Sigmoid(Op):
    def __init__(self, name):
        self._name = name
        self._params = []

    def forward(self, in_data):
        out_data = 1/(1+np.exp(-in_data))
        self.sx = out_data
        return out_data

    def backward(self, out_grad, lr, wd):
        in_grad = self.sx*(1-self.sx)
        return in_grad*out_grad
