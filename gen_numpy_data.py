import torch
import numpy as np
import torchvision.datasets as dset
from utils import _data_transforms_cifar100
import itertools
import time


train_transform, valid_transform = _data_transforms_cifar100()
train_data = dset.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)


path = "./data/bn_data_full.npy"
data = []

for i, (inputs, labels) in enumerate(train_queue):
    data.append(inputs)

data = torch.cat(data, dim=0)
print(data.shape)
data = data.numpy()

np.save(path, data)




test_data = dset.CIFAR100(root='./data', train=False, download=True, transform=valid_transform)
test_queue = torch.utils.data.DataLoader(
    test_data, batch_size=128, shuffle=False, pin_memory=False, num_workers=4)

path1 = "./data/test_inputs.npy"
path2 = "./data/test_labels.npy"
# start = time.time()
data1 = []
data2 = []
for i, (inputs, labels) in enumerate(test_queue):
    data1.append(inputs)
    data2.append(labels)

data1 = torch.cat(data1, dim=0)
data2 = torch.cat(data2, dim=0)

print(data1.shape)
print(data2.shape)

data1 = data1.numpy()
data2 = data2.numpy()

np.save(path1, data1)
np.save(path2, data2)