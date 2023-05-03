import torch
import torch.nn as nn


def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    #cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    cr = nn.CrossEntropyLoss()
    return cr(output, target)
