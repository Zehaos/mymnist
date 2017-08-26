class Op(object):
    def __init__(self, name):
        self._name = name

    def forward(self, in_data):
        raise NotImplementedError

    def backward(self, out_grad, lr, wd):
        raise NotImplementedError

    def set_params(self, params):
        self._params = params

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params
