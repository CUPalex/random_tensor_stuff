import torch
from torch import nn
from TTCL import TTCL

class ModelConv(nn.Module):
    def __init__(self, inp_ch=3):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inp_ch, 64, (3, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU()
        )
        self.tcl1 = nn.Sequential(
                nn.Conv2d(64, 64, (3, 3), padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2)
        )
        self.tcl2 = nn.Sequential(
                nn.Conv2d(64, 128, (3, 3), padding='same'),
                nn.BatchNorm2d(128),
                nn.ReLU()
        )
        self.tcl3 = nn.Sequential(
                nn.Conv2d(128, 128, (3, 3), padding='same'),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2)
        )
        self.tcl4 = nn.Sequential(
                nn.Conv2d(128, 128, (3, 3), padding='same'),
                nn.BatchNorm2d(128),
                nn.ReLU()
        )
        self.tcl5 = nn.Sequential(
                nn.Conv2d(128, 128, (3, 3), padding='same'),
                nn.AvgPool2d(4)
        )
        self.linear = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.tcl1(x)
        x = self.tcl2(x)
        x = self.tcl3(x)
        x = self.tcl4(x)
        x = self.tcl5(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class Model(nn.Module):
    def __init__(self, device, inp_ch=3, p=1, ranks=[(20, 20, 20, 1), (27, 22, 22, 1), (23, 23, 23, 1), (23, 23, 23, 1), (23, 23, 23, 1)]):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inp_ch, 64, (3, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU()
        )
        self.tcl1 = nn.Sequential(
                TTCL((4, 4, 4), (4, 4, 4), (3, 3), rank=ranks[0], p=p, padding='same', device=device),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2)
        )
        self.tcl2 = nn.Sequential(
                TTCL((4, 4, 4), (4, 8, 4), (3, 3), rank=ranks[1], p=p, padding='same', device=device),
                nn.BatchNorm2d(128),
                nn.ReLU()
        )
        self.tcl3 = nn.Sequential(
                TTCL((4, 8, 4), (4, 8, 4), (3, 3), rank=ranks[2], p=p, padding='same', device=device),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2)
        )
        self.tcl4 = nn.Sequential(
                TTCL((4, 8, 4), (4, 8, 4), (3, 3), rank=ranks[3], p=p, padding='same', device=device),
                nn.BatchNorm2d(128),
                nn.ReLU()
        )
        self.tcl5 = nn.Sequential(
                TTCL((4, 8, 4), (4, 8, 4), (3, 3), rank=ranks[4], p=p, padding='same', device=device),
                nn.AvgPool2d(4)
        )
        self.linear = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.tcl1(x)
        x = self.tcl2(x)
        x = self.tcl3(x)
        x = self.tcl4(x)
        x = self.tcl5(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x