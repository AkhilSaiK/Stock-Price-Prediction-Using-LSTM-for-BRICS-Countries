import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns

class BasicNN(nn.Module):
    def __init__(self):
        super().__init()