import abc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from tqdm import tqdm

import utils


class PyTorchModel(nn.Module, abc.ABC):
    """Abstract class for pytorch models."""

    def __init__(self, *args, **kwargs):
        super(PyTorchModel, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = {}

    def train_on_batch(self, inputs, targets):
        self.optimizer.zero_grad()

        predictions = self(inputs)
        loss = self.criterion(predictions, targets)

        loss.backward()
        self.optimizer.step()


    def fit_generator(self,
            generator,
            steps_per_epoch = None,
            epochs = 1,
            validation_data = None,
            validation_steps = None,
            before_batch_fn = None,
            after_batch_fn = None,
            before_epoch_fn = None,
            after_epoch_fn = None,
            before_validation_fn = None,
            after_validation_fn = None,
        ):
        starting_time = time.time()
        history = {}

        previous_loss = np.log(10)
        smoothing = 0.99

        # Will raise an error if generator is a real generator (no __len__).
        if steps_per_epoch is None:
            steps_per_epoch = len(generator)

        for epoch in range(epochs):
            if before_epoch_fn is not None:
                before_epoch_fn(self, epoch=epoch) 
            self.train()

            tqdm_batches = tqdm(
                enumerate(generator),
                desc="Epoch n°{}/{}".format(epoch+1, epochs),
                total=steps_per_epoch,
                unit="batch",
                leave=False,
                ascii=True,
            )

            for batch_id, (inputs, targets) in tqdm_batches:
                if before_batch_fn is not None:
                    before_batch_fn(self, epoch=epoch, step=batch_id)

                self.optimizer.zero_grad()

                predictions = self(inputs)
                loss = self.criterion(predictions, targets)

                loss.backward()
                self.optimizer.step()

                #if self.scheduler is not None:
                #    self.scheduler.step()

                current_loss = loss.item()
                current_loss = (smoothing*previous_loss) + (
                    (1-smoothing)*current_loss)
                previous_loss = current_loss

                tqdm_batches.set_postfix(
                        {"loss": "{:.3f}".format(current_loss)},
                        refresh=False)

                if after_batch_fn is not None:
                    after_batch_fn(self, epoch=epoch, step=batch_id)

                if batch_id+1 >= steps_per_epoch:
                    break


            tqdm_batches.close()


            # Compute training metrics.
            if before_validation_fn is not None:
                before_validation_fn(self, epoch=epoch)

            epoch_metrics = self.evaluate(
                    generator, 
                    self.metrics, 
                    validation_steps,
                )
            for metric_name, value in epoch_metrics.items():
                history.setdefault(metric_name, []).append(value)
            previous_loss = epoch_metrics['loss']

            if validation_data is not None:
                validation_metrics = self.evaluate(
                        validation_data, 
                        self.metrics, 
                        validation_steps
                    )
                for metric_name, value in validation_metrics.items():
                    history.setdefault('val_'+metric_name, []).append(value)
                    epoch_metrics['val_'+metric_name] = value


            if after_validation_fn is not None:
                after_validation_fn(self, epoch=epoch)

            if after_epoch_fn is not None:
                after_epoch_fn(self, epoch=epoch, metrics=epoch_metrics)

            print("Epoch {}/{}:".format(epoch+1, epochs),
                utils.format_metrics(epoch_metrics))
        
        print("Training completed in {:.1f}s.".format(time.time()-starting_time))

        return history

    def evaluate(self, dataloader, metrics=None, steps=None,
            confusion=False):
        if steps is None:
            steps = len(dataloader)

        metric_values = {
                'loss': 0.0
        }
        if metrics is None:
            metrics = self.metrics
        for metric_name, metric_fn in metrics.items():
            metric_values[metric_name] = 0.0

        if confusion:
            confusion_matrix = torch.zeros(self.nb_classes, self.nb_classes)

        ## Warmup the BNs.
        #self.train
        #with torch.no_grad():
        #    for batch_id, (inputs, targets) in enumerate(dataloader):
        #        outputs = self(inputs)
        #        if batch_id+1 >= 50: break

        tqdm_batches = tqdm(
            enumerate(dataloader),
            desc="Evaluation",
            total=steps,
            unit="batch",
            leave=False,
            ascii=True,
        )

        self.eval()
        with torch.no_grad():
            for batch_id, (inputs, targets) in tqdm_batches:
                outputs = self(inputs)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                predictions = torch.max(logits, 1)[1]

                metric_values['loss'] += self.criterion(
                        outputs, targets).item()

                for metric_name, metric_fn in metrics.items():
                    metric_values[metric_name] += metric_fn(
                            outputs,
                            targets).item()

                if confusion:
                    for t, p in zip(targets.view(-1), predictions.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                if batch_id+1 >= steps:
                    break

        tqdm_batches.close()

        for metric_name in metric_values:
            metric_values[metric_name] /= steps

        if confusion:
            return metric_values, confusion_matrix

        return metric_values

    def bn_update(self, dataloader, steps=300):
        self.train()
        with torch.no_grad():
            for batch_id, (inputs, _) in enumerate(dataloader):
                outputs = self(inputs)
                if batch_id+1 >= steps:
                    break

