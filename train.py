from loader import mnist
from network import NN
from op import Fullyconnected, Sigmoid, SimpleBatchNorm, Relu, Dropout
import numpy as np

# Set hyper-params here
batch_size = 20
learning_rate = 0.01
learning_step = [6]
weight_decay = 0.
total_epoch = 12
model_path = './model/'

# Construct nn
MLP = NN(learning_rate=learning_rate)
MLP.add(SimpleBatchNorm(name='data_batchnorm', istraining=True))
MLP.add(Dropout(ratio=0.3, name='data_dropout'))
MLP.add(Fullyconnected(784, 512, name="fc1"))
MLP.add(SimpleBatchNorm(name='fc1_batchnorm'))
MLP.add(Relu(name="fc1_relu"))
MLP.add(Fullyconnected(512, 512, name="fc2"))
MLP.add(Relu(name="fc2_relu"))
MLP.add(Fullyconnected(512, 10, name="fc3"))
MLP.set_wd(weight_decay)
MLP.set_lr(learning_rate)

# Load mnist data
mnist = mnist(path="./data/", batch_size=batch_size)
num_imgs = mnist.get_num()

epoch = 0
iters = 0
lr_step = 0
pos = 0
learning_step.append(total_epoch)

# Train loop
while epoch < total_epoch:

    if epoch > learning_step[lr_step]:
        lr_step += 1
        MLP.set_lr(MLP.get_lr() * 0.1)

    imgs, labels = mnist.get_batch()

    # Forward and cal log loss
    out_data = MLP.forward(imgs)

    # Softmax with cross entropy loss
    probs = np.exp(out_data - out_data.max(axis=1).reshape((out_data.shape[0], 1)))
    probs /= probs.sum(axis=1).reshape((out_data.shape[0], 1))
    grads = probs.copy()
    grads[np.arange(labels.shape[0]), labels] -= 1.0
    grads /= batch_size

    # Backward and update params
    MLP.backward(grads=grads)

    # Metrics
    # Cal loss
    cls_probs = probs[range(batch_size), labels]
    cls_probs += 1e-14
    log_loss = -1 * np.log(cls_probs)
    log_loss = np.sum(log_loss) / float(batch_size)

    # Cal acc
    pos += np.sum(labels == probs.argmax(axis=1))
    acc = pos/float(batch_size*(iters+1))

    # Info
    if iters%100 == 0:
        print "Epochs:%d, Iters:%d, Acc:%.3f, Logloss:%.5f"%(epoch, iters, acc, log_loss)
    iters += 1

    # Save model
    if mnist.get_epoch() > epoch:
        pos = 0
        iters = 0
        epoch = mnist.get_epoch()
        MLP.save_model(model_path+"mnist_mlp_epoch%s.model"%epoch)
