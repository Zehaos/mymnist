import numpy as np
import os
from PIL import Image
from network import NN
from op import Fullyconnected, Sigmoid, SimpleBatchNorm, Relu

img_dir = "./imgs/"

model_path = './model/mnist_mlp_epoch9.model'

# Construct nn
MLP = NN()
MLP.add(SimpleBatchNorm(name="data_batchnorm", istraining=False))
MLP.add(Fullyconnected(784, 512, name="fc1"))
MLP.add(Relu(name="fc1_relu"))
MLP.add(SimpleBatchNorm(name='fc1_batchnorm'))
MLP.add(Fullyconnected(512, 512, name="fc2"))
MLP.add(Relu(name="fc2_relu"))
MLP.add(Fullyconnected(512, 10, name="fc3"))

# Load model
MLP.load_model(model_path)

for parent, dirnames, filenames in os.walk(img_dir):
    for filename in filenames:
        if filename.endswith("jpg") or filename.endswith("png"):
            img_path = os.path.join(parent, filename)
            pil_img = Image.open(img_path).convert('L')
            pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
            img = np.array(pil_img)
            out_data = MLP.forward(img.reshape((1, 784)))
            # Softmax
            probs = np.exp(out_data - out_data.max(axis=1).reshape((out_data.shape[0], 1)))
            probs /= probs.sum(axis=1).reshape((out_data.shape[0], 1))
            print img_path, 'predict:', np.argmax(probs)
