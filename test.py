from loader import mnist
from network import NN
from op import Fullyconnected, Sigmoid, SimpleBatchNorm, Relu
import numpy as np
import time

# Model
model_path = './model/mnist_mlp_epoch9.model'

# Construct nn
MLP = NN()
MLP.add(SimpleBatchNorm(name="data_batchnorm", istraining=False))
MLP.add(Fullyconnected(784, 512, name="fc1"))
MLP.add(SimpleBatchNorm(name='fc1_batchnorm'))
MLP.add(Relu(name="fc1_relu"))
MLP.add(Fullyconnected(512, 512, name="fc2"))
MLP.add(Relu(name="fc2_relu"))
MLP.add(Fullyconnected(512, 10, name="fc3"))

# Load model
MLP.load_model(model_path)

# Load mnist data
mnist = mnist(path="./data/", batch_size=1, test=True)
num_imgs = mnist.get_num()

pos = 0
total_time = 0
for i in range(num_imgs):
    img, label = mnist.get_batch()

    t1 = time.time()
    out_data = MLP.forward(img)
    t2 = time.time()

    # Softmax
    probs = np.exp(out_data - out_data.max(axis=1).reshape((out_data.shape[0], 1)))
    probs /= probs.sum(axis=1).reshape((out_data.shape[0], 1))

    pos += np.sum(label == probs.argmax(axis=1))
    print "test img%d, time:%.5f"%(i, t2-t1)
    total_time += t2-t1

print "acc:", pos/float(num_imgs)
print "average time:", total_time/float(num_imgs)