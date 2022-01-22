from Image_Classification_Model import MyNet
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    batch_size = 256
    #An example for MNIST DATASET- Need to convert to out own dataset
    #TODO Osama: Intergrate given dataset with current dataset
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    #init an instance of our dataset
    my_cnn = MyNet()
    #.parameters is a method of model we're inhereting in mynet
    sgd = SGD(my_cnn.parameters(), lr=1e-1)
    #TODO Osama: ASK Chen which loss to use
    cost = CrossEntropyLoss()
    #TODO Osama: check epochs with dataset
    epoch = 100

    for _epoch in range(epoch):
        #Set the mode for training mode-update params
        my_cnn.train(mode=True)
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            #always zero the grad
            sgd.zero_grad()
            #forward pass - Predict results of train_set
            predict_y = my_cnn(train_x.float())
            #calculate loss over the trainset
            loss = cost(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            #backpropogation of the result
            loss.backward()
            #apply the optimizer step - update the weights/parameters
            sgd.step()

        correct = 0
        _sum = 0
        #Truns off layers that should not be used during eval like batchnorm
        #The parallel of my_cnn.train() to retrun these layers on
        my_cnn.eval()
        #Turn off gradient calc
        with torch.no_grad():
            for idx, (test_x, test_label) in enumerate(test_loader):
                #Using detach to detach tensor from graph - not requiring grad
                predict_y = my_cnn(test_x.float()).detach()
                predict_ys = np.argmax(predict_y, axis=-1)
                label_np = test_label.numpy()
                is_correct = predict_ys == test_label
                correct += np.sum(is_correct.numpy(), axis=-1)
                _sum += is_correct.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))
        torch.save(my_cnn, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))
