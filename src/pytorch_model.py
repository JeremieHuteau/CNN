import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from tqdm import tqdm
import utils


class PyTorchModel(nn.Module, abc.ABC):
    """Abstract class for pytorch models."""

    def __init__(self, *args, **kwargs):
        # Subclasses should set criterion and optimizer, 
        # and optionally metrics.
        super(PyTorchModel, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.metrics = None

    def fit_generator(self,
            generator,
            steps_per_epoch = None,
            epochs = 1,
            validation_data = None,
            validation_steps = None,
        ):
        history = {}

        # Will raise an error if generator is a real generator (no __len__).
        if steps_per_epoch is None:
            steps_per_epoch = len(generator)

        for epoch in range(epochs):
            self.train()

            for batch_id, (inputs, targets) in tqdm(
                    enumerate(generator),
                    desc="Epoch n°{}/{}".format(epoch+1, epochs),
                    total=steps_per_epoch,
                    ncols=80,
                    position=0,
                ):
                self.optimizer.zero_grad()

                predictions = self(inputs)
                loss = self.criterion(predictions, targets)

                loss.backward()
                self.optimizer.step()

                if batch_id+1 >= steps_per_epoch:
                    break

            # Compute training metrics.
            epoch_metrics = self.evaluate(
                    generator, 
                    self.metrics, 
                    validation_steps
                )
            for metric_name, value in epoch_metrics.items():
                history.setdefault(metric_name, []).append(value)

            if validation_data is not None:
                validation_metrics = self.evaluate(
                        validation_data, 
                        self.metrics, 
                        validation_steps
                    )
                for metric_name, value in validation_metrics.items():
                    history.setdefault('val_'+metric_name, []).append(value)
                    epoch_metrics['val_'+metric_name] = value

            #print(utils.format_metrics(epoch_metrics))
            tqdm.write(utils.format_metrics(epoch_metrics))


        return history

    def evaluate(self, dataloader, metrics=None, steps=None):
        if steps is None:
            steps = len(dataloader)

        metric_values = {
                'loss': 0.0
        }
        if metrics is None:
            metrics = self.metrics
        if metrics is None:
            metrics = {}
        for metric_name, metric_fn in metrics.items():
            metric_values[metric_name] = 0.0

        self.eval()
        with torch.no_grad():
            for batch_id, (inputs, targets) in enumerate(dataloader):
                predictions = self(inputs)

                metric_values['loss'] += self.criterion(predictions, targets)
                for metric_name, metric_fn in metrics.items():
                    metric_values[metric_name] += metric_fn(predictions, targets)

                if batch_id+1 >= steps:
                    break

        for metric_name in metric_values:
            metric_values[metric_name] /= steps

        return metric_values



def nb_flattened_features(x):
    nb = 1
    for dimension in x.size():
        nb *= dimension
    return nb

class MLP(PyTorchModel):
    def __init__(self, x):
        super(MLP, self).__init__()

        self.input_dimension = nb_flattened_features(x)

        self.fc1 = nn.Linear(self.input_dimension, 16)
        self.fc1activation = nn.ReLU()

        self.fc2 = nn.Linear(16, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        self.metrics = {
            'accuracy': lambda y_hat, y: (torch.max(y_hat, 1)[1]==y).float().mean()
        }

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
    dummy_targets = torch.randint(0, 1, (2, 1))
    dummy_nb_classes = 10

    model = MLP(dummy_inputs)
    #model = CNN(dummy_inputs, dummy_nb_classes)
    torchsummary.summary(model, input_size=dummy_inputs.size()[1:])


    dummy_predictions = model(dummy_inputs)
    print(dummy_predictions)
    print(dummy_predictions.size())

