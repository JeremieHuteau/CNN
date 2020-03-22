import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision

from tqdm import tqdm
import matplotlib.pyplot as plt

import data
import models

def main(train_images_path, train_labels_path, model_save_path):
    """ Load data, train model, display metrics, save model."""
    # Load data.
    dataset = data.MnistDataset(
            train_images_path, train_labels_path,
            image_transform=torchvision.transforms.ToTensor(),
            label_transform=None,
        )

    # Split data into train/validation sets.
    train_split_size = int(len(dataset)*0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_split_size, len(dataset)-train_split_size])

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
    validation_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=True
        )

    # Get a single batch from the training data.
    # We use this batch to set the shapes of layers according to the shape
    # of the actual data.
    for inputs, labels in train_dataloader:
        sample = inputs
        break

    # Define model.
    model = models.CNN(nb_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    metrics = {
        'cross_entropy': cross_entropy,
        'accuracy': correct_predictions,
    }

    epochs = 2
    # Fit model.
    model, optimizer, history = fit(
            model, 
            criterion,
            optimizer,
            train_dataloader, 
            epochs=epochs, 
            validation_dataloader = validation_dataloader,
            validation_steps = 100,
            metrics = metrics,
        )

    # Plot learning curves.
    for metric_name, metric_values in history.items():
        plt.plot(metric_values, label=metric_name)
    plt.plot(np.full(epochs, 1.0), 'k--')
    plt.gca().set_ylim(bottom=0.0)
    plt.xlabel('epoch')
    plt.ylabel('metric value')
    plt.legend()
    plt.show(block=False)

    # Metrics computed on whole sets.
    training_metrics = validate(model, train_dataloader, metrics)
    validation_metrics = validate(model, validation_dataloader, metrics)
    print()
    print(
            "{:<12}".format("Training:"), 
            format_metrics(training_metrics))
    print(
            "{:<12}".format("Validation:"), 
            format_metrics(validation_metrics))

    # Save the model and optimizer states.
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)

    plt.show()

def fit(
        model, 
        criterion,
        optimizer,
        train_dataloader, 
        epochs=1, 
        validation_dataloader = None,
        validation_steps = None,
        metrics = None,
    ):

    if validation_steps is None and validation_dataloader is not None:
        validation_steps = 100

    history = {}

    for epoch in range(epochs):
        model.train()

        for batch_id, (inputs, targets) in tqdm(
                enumerate(train_dataloader),
                desc="Epoch n°{}".format(epoch+1),
                total=len(train_dataloader),
                ncols=80,
            ):
            optimizer.zero_grad()

            predictions = model(inputs)
            loss = criterion(predictions, targets.long())

            loss.backward()
            optimizer.step()


        model.eval() 

        training_metrics = validate(
                model, 
                train_dataloader, 
                metrics, 
                validation_steps
            )
        for metric_name, value in training_metrics.items():
            history.setdefault(metric_name, []).append(value)

        if validation_dataloader is not None:
            validation_metrics = validate(
                    model, 
                    validation_dataloader, 
                    metrics, 
                    validation_steps
                )
            for metric_name, value in validation_metrics.items():
                history.setdefault('val_'+metric_name, []).append(value)

        print(", ".join([
                format_metrics(training_metrics), 
                format_metrics(validation_metrics, 'val_')]))


    return model, optimizer, history

def validate(model, dataloader, metrics, steps=None):
    if steps is None:
        steps = len(dataloader)

    metric_values = {
        metric_name: 0.0
        for metric_name in metrics
    }
    total_predictions = 0

    model.eval()
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            predictions = model(inputs)

            nb_predictions = len(targets) 
            total_predictions += nb_predictions

            for metric_name, metric_fn in metrics.items():
                metric_values[metric_name] += metric_fn(predictions, targets)

            if batch_id+1 >= steps:
                break

    for metric_name in metric_values:
        metric_values[metric_name] /= total_predictions

    return metric_values

def format_metrics(metrics, prefix=''):
    return ", ".join([
        "{}: {:.4f}".format(prefix+metric_name, metric_values)
            for metric_name, metric_values in metrics.items()
        ])


def cross_entropy(predictions, targets):
    return torch.nn.functional.cross_entropy(
            predictions, targets.long(), reduction='sum').item()

def correct_predictions(predictions, targets):
    return (torch.max(predictions, 1)[1] == targets.long()).sum().item()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('train_images_path', type=Path)
    argparser.add_argument('train_labels_path', type=Path)
    argparser.add_argument('model_save_path', type=Path)

    args = argparser.parse_args()

    main(args.train_images_path, args.train_labels_path, args.model_save_path)

