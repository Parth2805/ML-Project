from pip._vendor.distlib.compat import raw_input
import sys
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cifar10
from cifar10 import CNN


user_response = raw_input("Enter Y for Yes and N for No\n")

if user_response.__eq__('Y'):
    print("You said Yes")
else:
    print("You said No")

cifar10.Cifar10(sys.argv[1], user_response)
