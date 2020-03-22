import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

def nb_flattened_features(x):
    nb = 1
    for dimension in x.size()[1:]:
        nb *= dimension
    return nb

class MLP(nn.Module):
    def __init__(self, x):
        super(MLP, self).__init__()

        self.input_dimension = nb_flattened_features(x)

        self.fc1 = nn.Linear(self.input_dimension, 16)
        self.fc1activation = nn.ReLU()

        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, self.input_dimension)

        x = self.fc1(x)
        x = self.fc1activation(x)

        x = self.fc2(x)

        return x

class CNN(nn.Module):
    def __init__(self, nb_classes, *args, **kwargs):
        super(CNN, self).__init__()

        # Number of filters of last convolution = nb features of 1st FC layer
        self.nb_filters = 32

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv1activation = nn.ReLU()
        self.conv1pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(4, self.nb_filters, 3)
        self.conv2activation = nn.ReLU()
        #self.conv2pool = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(self.nb_filters, self.nb_filters//2)
        self.fc1activation = nn.ReLU()

        self.fc2 = nn.Linear(self.nb_filters//2, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1activation(x)
        x = self.conv1pool(x)

        x = self.conv2(x)
        x = self.conv2activation(x)

        x = self.global_pool(x)

        x = x.view(-1, self.nb_filters)

        x = self.fc1(x)
        x = self.fc1activation(x)

        x = self.fc2(x)

        return x


if __name__ == '__main__':
    dummy_inputs = torch.randn(2, 1, 28, 28)
    dummy_targets = torch.randn(2, 1)
    dummy_nb_classes = 10

    model = CNN(dummy_inputs, dummy_nb_classes)
    torchsummary.summary(model, input_size=dummy_inputs.size()[1:])


    dummy_predictions = model(dummy_inputs)
    print(dummy_predictions)
    print(dummy_predictions.size())
