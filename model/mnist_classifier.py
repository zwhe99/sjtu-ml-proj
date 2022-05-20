from typing import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.layers = nn.ModuleDict(OrderedDict({
            f"sub-layer{i}": nn.Sequential(OrderedDict({
                            "conv": nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2),
                            "relu": nn.ReLU(),
                            "bn": nn.BatchNorm2d(num_features=1),
                        }))
            for i in range(4)
        }))
        self.max_pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(144, 64)
        self.fc2 = nn.Linear(64, 10)
        self.tanh = nn.Tanh()

    def forward(self, x, return_feature=False, layer_name="fc2"):
        for name, m in self.layers.items():
            x = m(x)
            if layer_name == name:
                h = torch.flatten(x, 1)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.tanh(x)
        if layer_name == "fc1":
            h = x.clone()

        x = self.fc2(x)
        if layer_name == "fc2":
            h = x.clone()

        if return_feature:
            return x, h
        else:
            return x
