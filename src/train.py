import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torchvision
import torchcontrib

from tqdm import tqdm
import matplotlib.pyplot as plt

import data
from CNN import Model
import utils

def main(train_images_path, train_labels_path, model_save_path):
    """ Load data, train model, display metrics, save model."""
    # Load data.
    #dataset = data.MnistDataset(
    #        train_images_path, train_labels_path,
    #        image_transform=None, 
    #        label_transform=None,
    #    )
    dataset = torchvision.datasets.CIFAR10(
        'data/', train=True, 
        transform=None, target_transform=None, 
        download=True)

    if torch.cuda.is_available():
        print("GPU enabled.")
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("CPU only.")

    #dataset = torchvision.datasets.MNIST(
    #    'data/', train=True, 
    #    transform=None, target_transform=None, 
    #    download=True)
    
    # Split data into train/validation sets.
    train_split_size = int(len(dataset)*0.8)
    validation_split_size = len(dataset)-train_split_size
    training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [train_split_size, len(dataset)-train_split_size])

    # These values come from a pytorch github issue, and are
    # computed on the whole dataset.
    #normalization = torchvision.transforms.Normalize(
    #    (0.4914, 0.4822, 0.4465), 
    #    (0.247, 0.243, 0.261)
    #)
    # These values are close enough, and do not leak validation into training
    # set. (I could also compute them myself every split, but laziness)
    normalization = torchvision.transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.25, 0.25, 0.25)
    )
    crop_size = 26 
    augmentation = torchvision.transforms.Compose([
        #torchvision.transforms.RandomRotation(22.5, fill=(0,)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomCrop(crop_size, 0, fill=(0,)),
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        torchvision.transforms.ToTensor(),
        normalization,
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        normalization,
    ])

    training_dataset.dataset.transform = augmentation
    validation_dataset.dataset.transform = test_transform

    batch_size = 2**8
    steps_per_epoch = (train_split_size // batch_size)

    #print(training_dataset.image_transform, validation_dataset.image_transform)
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
    model = Model(nb_classes=10, input_image=sample[0])

    epochs = 15
    print(steps_per_epoch, "steps per epoch.")

    #strategy = None
    #strategy = 'ReduceOnPlateau'
    #strategy = 'CyclicLR'
    strategy = 'OneCycle'
    swa = False

    for group in model.optimizer.param_groups:
        base_lr = group['lr']
    min_lr = base_lr #* 0.1
    max_lr = base_lr * 25


    if input('LR range test ? y/[n]: ') == 'y':
        import torch_lr_finder

        for group in model.optimizer.param_groups:
            group['lr'] = base_lr*1e-2
        lr_finder = torch_lr_finder.LRFinder(model, model.optimizer, model.criterion)
        lr_finder.range_test(training_dataloader, 
                end_lr=base_lr*1e2, num_iter=steps_per_epoch*1)
        lr_finder.plot()

        return

    print("LR (min, base, max) = ({}, {}, {}))".format(
        min_lr, base_lr, max_lr))

    strategy_functions = {
        'before_epoch_fn': None,
        'after_epoch_fn': None,
        'before_batch_fn': None,
        'after_batch_fn': None,
        'before_validation_fn': None,
        'after_validation_fn': None,
    }

    if strategy == 'WarmUpReduceOnPlateau':
        pct_start = max(1/epochs, 1/8)
        div_factor = max_lr / (base_lr * 0.1)
        warmup_steps = pct_start*epochs*steps_per_epoch
        print("OneCycle with {} warmup steps ({:.2f} epochs)".format(
            warmup_steps, warmup_steps / steps_per_epoch
        ))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.optimizer,
            max_lr = max_lr, div_factor = div_factor, 
            final_div_factor = 1,
            epochs = epochs, steps_per_epoch = steps_per_epoch,
            pct_start = pct_start,
            anneal_strategy = 'linear',
        )

        strategy_functions['after_batch_fn'] = lambda *a, **k: scheduler.step()

    if strategy == 'ReduceOnPlateau':
        for group in model.optimizer.param_groups:
            group['lr'] = max_lr

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, 
            mode='min', 
            factor=0.2, patience=0, verbose=True, 
            threshold=1e-2, threshold_mode='rel', 
            cooldown=0, min_lr=base_lr * 1e-2
        )
        strategy_functions['after_epoch_fn'] = lambda *a, **k: scheduler.step(
            k['metrics']['loss'])

    if strategy == 'CyclicLR':
        # Cycle variables
        cycle_length = 0.5 #max(1, epochs // 8)
        steps_per_cycle = int(steps_per_epoch * cycle_length)

        step_up_ratio = 1/8
        step_down_ratio = 1 - step_up_ratio
        step_size_up = int(steps_per_epoch * step_up_ratio * cycle_length)
        step_size_down = steps_per_cycle - step_size_up
        #int(steps_per_epoch * step_down_ratio * cycle_length)

        print("Cyclic LR with steps of sizes:", step_size_up, step_size_down)
        print("Cycling between {} and {}".format(min_lr, max_lr))

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            model.optimizer, 
            min_lr, max_lr, 
            step_size_up=step_size_up, 
            step_size_down=step_size_down, 
            mode='triangular', 
            gamma=1.0, scale_fn=None, 
            scale_mode='iterations', 
            cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, 
            last_epoch=-1
        )

        strategy_functions['after_batch_fn'] = lambda *a, **k: scheduler.step()
        

    if strategy == 'OneCycle':
        pct_start = 3/epochs #max(1/epochs, 3/10)
        max_lr = base_lr * 200
        div_factor = 200 #max_lr / (base_lr * 0.1)
        final_div_factor = 1/100
        warmup_steps = pct_start*epochs*steps_per_epoch
        print("OneCycle with {} warmup steps ({:.2f} epochs)".format(
            warmup_steps, warmup_steps / steps_per_epoch
        ))
        print("start_lr, max_lr, final_lr = {:.4f}, {:.4f}, {:.4f}".format(
            max_lr / div_factor, max_lr, (max_lr / div_factor) / final_div_factor
        ))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.optimizer,
            max_lr = max_lr, div_factor = div_factor, 
            final_div_factor = final_div_factor,
            epochs = epochs, steps_per_epoch = steps_per_epoch,
            pct_start = pct_start,
            anneal_strategy = 'linear',
        )
        strategy_functions['after_batch_fn'] = lambda *a, **k: scheduler.step()

    # Fit model.
    history = model.fit_generator(
        training_dataloader,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = validation_dataloader,
        validation_steps = min(300, validation_split_size // batch_size),
        **strategy_functions,
    )

    strategy_functions = {
        'before_epoch_fn': None,
        'after_epoch_fn': None,
        'before_batch_fn': None,
        'after_batch_fn': None,
        'before_validation_fn': None,
        'after_validation_fn': None,
    }

    # Stochastic Weight Averaging
    if swa:
        min_lr = base_lr
        max_lr = base_lr * 1e0
        model.optimizer = torchcontrib.optim.SWA(
            model.optimizer,
            swa_start = 0,
            swa_freq = steps_per_epoch//2,
            swa_lr = max_lr)

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            model.optimizer, 
            min_lr, max_lr, 
            step_size_up=int(steps_per_epoch*0.3), 
            step_size_down=int(steps_per_epoch*0.7), 
            mode='triangular', 
            gamma=1.0, scale_fn=None, 
            scale_mode='iterations', 
            cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, 
            last_epoch=-1
        )
        #strategy_functions['after_batch_fn'] = lambda *a, **k: scheduler.step()
        #print("SWA with lr: {} {}".format(min_lr, max_lr))
        print("SWA with lr: {}".format(max_lr))

        #strategy_functions['before_validation_fn'] = lambda *a, **k: \
        #    model.optimizer.swap_swa_sgd()
        #strategy_functions['after_validation_fn'] = lambda *a, **k: \
        #    model.optimizer.swap_swa_sgd()

        swa_history = model.fit_generator(
            training_dataloader,
            steps_per_epoch = steps_per_epoch,
            epochs = max(3, int(epochs*0.25)),
            validation_data = validation_dataloader,
            validation_steps = 300,
            **strategy_functions,
        )

        for metric_name, values in swa_history.items():
            history[metric_name] = history[metric_name] + values


        model.optimizer.swap_swa_sgd()
        model.optimizer.bn_update(training_dataloader, model)

    # Disable augmentations.
    training_dataset.dataset.transform = test_transform
    validation_dataset.dataset.transform = test_transform

    training_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True)

    model.bn_update(training_dataloader)
    # Compute metrics on whole sets.
    training_metrics, training_confusion_matrix = model.evaluate(
            training_dataloader, steps=None, confusion=True)
    validation_metrics, validation_confusion_matrix = model.evaluate(
            validation_dataloader, steps=None, confusion=True)
    print()
    print("{:<12}".format("Training:"), utils.format_metrics(training_metrics))
    print("{:<12}".format("Validation:"), utils.format_metrics(validation_metrics))
    print()

    class_labels = training_dataset.dataset.classes
    utils.print_confusion_matrix(
            training_confusion_matrix, class_labels, normalize=True)
    print()
    utils.print_confusion_matrix(
            validation_confusion_matrix, class_labels, normalize=True)
    print()

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

