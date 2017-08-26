from operator import Op
import numpy as np

class Dropout(Op):
    def __init__(self, ratio, name):
        self._name = name
        self._params = []
        self._ratio = ratio

    def forward(self, in_data):
        bs, len = in_data.shape
        self.mask = np.ones_like(in_data)
        for b in range(bs):
            self.mask[b][np.random.choice(range(len), len*self._ratio, replace=False)] = 0
        out_data = in_data*self.mask
        return out_data

    def backward(self, out_grad, lr, wd):
        return out_grad*self.mask
