import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision

from tqdm import tqdm
import matplotlib.pyplot as plt

import data
from CNN import Model
import utils

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
    training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [train_split_size, len(dataset)-train_split_size])

    batch_size = 16
    training_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True)

    # Get a single batch from the training data.
    # We use this batch to set the shapes of layers according to the shape
    # of the actual data.
    for inputs, labels in training_dataloader:
        sample = inputs
        break
    model = Model(nb_classes=10, inputs=sample[0])


    epochs = 10
    # Fit model.
    history = model.fit_generator(
            training_dataloader,
            steps_per_epoch = (train_split_size // batch_size),
            epochs = epochs,
            validation_data = validation_dataloader,
            validation_steps = 100
        )

    # Metrics computed on whole sets.
    training_metrics = model.evaluate(training_dataloader, model.metrics)
    validation_metrics = model.evaluate(validation_dataloader, model.metrics)
    print()
    print("{:<12}".format("Training:"), utils.format_metrics(training_metrics))
    print("{:<12}".format("Validation:"), utils.format_metrics(validation_metrics))

    # Save the model and optimizer states.
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'learning_history': history,
        }, model_save_path)

    # Plot learning curves.
    for metric_name, metric_values in history.items():
        plt.plot(metric_values, label=metric_name)
    plt.plot(np.full(epochs, 1.0), 'k--')
    plt.gca().set_ylim(bottom=0.0)
    plt.xlabel('epoch')
    plt.ylabel('metric value')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('train_images_path', type=Path)
    argparser.add_argument('train_labels_path', type=Path)
    argparser.add_argument('model_save_path', type=Path)

    args = argparser.parse_args()

    main(args.train_images_path, args.train_labels_path, args.model_save_path)

