import argparse
from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, 
            images_path, labels_path, 
            image_transform=None, label_transform=None):
        self.images = idx2np(images_path)
        self.labels = idx2np(labels_path)
        self.image_transform = image_transform
        self.label_transform = label_transform

        if len(self.images) != len(self.labels):
            raise ValueError(
                    "Length of images and labels are different: " + 
                    "{} vs {}".format(len(self.images), len(self.labels)) +
                    "\n\t images loaded from {}".format(images_path) +
                    "\n\t labels loaded from {}".format(labels_path) 
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx,:,:,None], int(self.labels[idx])

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return image, label

# Equivalent to torchvision.transforms.ToTensor() for MNIST images.
# We use the torch version for reliability (and likely performance).
class GrayscaleToTensor(object):
    def __call__(self, image):
        tensor = torch.from_numpy(
                image.transpose((2, 0, 1)) / np.iinfo(image.dtype).max
            )
        return tensor

def idx2np(idx_path):
    """Read a binary file (in the IDX format) and convert it to a numpy array."""
    # 3rd byte encodes the data type: this a a dictionary mapping this byte
    # to a 2-tuple comprising the type string and the number of bytes
    # of the data type. 
    int2typeformat = {
            8: ('B', 1),
            9: ('b', 1),
            10: ('h', 2),
            11: ('i', 4),
            12: ('f', 4),
            13: ('d', 8),
        }

    with open(idx_path, 'rb') as idx:
        # First 2 bytes are always 0.
        idx.read(2)

        # 3rd byte is data type, 4th is the number of dimensions
        data_type, nb_dimensions = struct.unpack('>BB', idx.read(2))

        # Convert the 3rd byte to its format string
        formatstring, nb_bytes = int2typeformat[data_type]
        formatstring = '>' + formatstring

        # The nb_dimensions next 4 bytes are the number of rows/samples, 
        # as unsigned ints.
        shape = struct.unpack(
                '>' + "".join(['I']*nb_dimensions), 
                idx.read(4*nb_dimensions)
            )

        tensor = np.fromfile(idx, dtype=np.dtype(formatstring)).reshape(shape)

    return tensor

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('images_path', type=Path)
    argparser.add_argument('labels_path', type=Path)
    args = argparser.parse_args()

    images_path = args.images_path
    labels_path = args.labels_path

    dataset = MnistDataset(
            images_path, labels_path,
            image_transform=torchvision.transforms.ToTensor(),
            label_transform=None,
        )

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True
        )

    for batch_id, (images, labels) in enumerate(dataloader):
        for i in range(2):
            print(images[i])
            print(labels[i])
            print(images[i].size())

            plt.imshow(
                    images[i].numpy()[0],
                    cmap='gray'
                )
            plt.show()
        break
