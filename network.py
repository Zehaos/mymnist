import cPickle as pkl

class NN(object):
    def __init__(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.structure = list()

    def add(self, operator):
        self.structure.append(operator)

    def forward(self, inputs):
        mid_data = inputs
        for op in self.structure:
            mid_data = op.forward(mid_data)
        return mid_data

    def backward(self, grads):
        mid_grad = grads
        for op in self.structure[::-1]:
            mid_grad = op.backward(mid_grad, self.learning_rate, self.weight_decay)

    def set_lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_lr(self):
        return self.learning_rate

    def save_model(self, path):
        model = dict()
        for op in self.structure:
            model.update({op.name: op.params})

        with open(path, 'w') as f:
            pkl.dump(model, f)

    def load_model(self, path):
        with open(path, 'r') as f:
            model = pkl.load(f)
        for op in self.structure:
            op.set_params(model[op.name])

    def set_wd(self, weight_decay):
        self.weight_decay = weight_decay
