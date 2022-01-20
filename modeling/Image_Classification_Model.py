from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential
from torch.nn import LogSoftmax
from torch import flatten


class LeNet(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(LeNet, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        # fist layer
        layer1 = Conv2d(16, 8, kernel_size=5, stride=1, padding=1)
        layer1.append(BatchNorm2d(4))
        layer1.append(ReLU(inplace=True))
        layer1.append(MaxPool2d(kernel_size=2, stride=2))

        # second layer
        layer2 = Conv2d(8, 8, kernel_size=5, stride=1, padding=1)
        layer2.append(BatchNorm2d(4))
        layer2.append(ReLU(inplace=True))
        layer2.append(MaxPool2d(kernel_size=2, stride=2))

        # third layer
        layer3 = Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        layer3.append(BatchNorm2d(4))
        layer3.append(ReLU(inplace=True))
        layer3.append(MaxPool2d(kernel_size=2, stride=2))

        # fourth layer
        layer4 = Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        layer4.append(BatchNorm2d(4))
        layer4.append(ReLU(inplace=True))
        layer4.append(MaxPool2d(kernel_size=2, stride=2))

        # fifth layer
        layer5 = Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        layer5.append(BatchNorm2d(4))
        layer5.append(ReLU(inplace=True))
        layer5.append(MaxPool2d(kernel_size=2, stride=2))

        # ((W - K + 2P) / S) + 1) Where W = Input size  K = Filter size  S = Stride  P = Padding
        # TODO Osama: Calculate correct dimentions for Linear layer input
        linear_layer = Linear(20, 3)

        self.cnn_layers = Sequential(
            layer1, layer2, layer3, layer4, layer5
        )

        self.linear_layers = Sequential(
            linear_layer
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
            # pass the input through our first set of CONV => RELU =>
            # POOL layers
            x = self.cnn_layers(x)
            x = flatten(x, 1)
            x = self.linear_layers(x)
            # pass the output to our softmax classifier to get our output predictions
            output = self.logSoftmax(x)
            # return the output predictions
            return output
