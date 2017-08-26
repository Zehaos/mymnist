import numpy as np
import struct

# Mnist parser from http://www.csuldw.com/2016/02/25/2016-02-25-machine-learning-MNIST-dataset/
class DataUtils(object):
    def __init__(self, filename=None):
        self._filename = filename

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def getImage(self):
        binfile = open(self._filename, 'rb')
        buf = binfile.read()
        binfile.close()
        index = 0
        numMagic, numImgs, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)

    def getLabel(self):
        binFile = open(self._filename, 'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

class mnist():
    def __init__(self, path, batch_size, test=False):

        img_path = path + "/train-images.idx3-ubyte"
        label_path = path + "/train-labels.idx1-ubyte"
        if test:
            img_path = path + "/t10k-images.idx3-ubyte"
            label_path = path + "/t10k-labels.idx1-ubyte"
        mnist_img = DataUtils(img_path)
        mnist_label = DataUtils(label_path)

        self.imgs = mnist_img.getImage()
        self.labels = mnist_label.getLabel()

        self.idx = 0
        self.bs = batch_size
        self.num = self.labels.shape[0]
        self.epoch = 0

    def shuffle(self):
        inds = np.arange(self.labels.shape[0])
        np.random.shuffle(inds)
        self.imgs = self.imgs[inds]
        self.labels = self.labels[inds]

    def get_batch(self):
        if self.bs + self.idx > self.num:
            self.idx = 0
            self.epoch += 1
            self.shuffle()
        batch_inds = self.idx + np.arange(self.bs)
        batch_imgs = self.imgs[batch_inds]
        batch_labels = self.labels[batch_inds]
        self.idx += self.bs
        return batch_imgs, batch_labels

    def get_num(self):
        return self.num

    def get_epoch(self):
        return self.epoch

