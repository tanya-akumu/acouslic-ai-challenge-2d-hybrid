import torch
import torch.nn as nn
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR,ExponentialLR
import numpy as np

def initialize_model(device,warmup=5,weights=None,base_rate=1e-3,decay=0.02,exp_factor = 1000, model='50'):
    if model == '50':
        model = models.resnet50(weights=weights)
    if model == '18':
        model = models.resnet18(weights=weights)

    #model.fc = nn.Linear(model.fc.in_features, 2)
    # Create a new fully connected layer with 2 output features
    new_fc = nn.Linear(model.fc.out_features, 2)
    original_fc = model.fc
    # Combine the original fully connected layer and the new layer into a Sequential container
    model.fc = nn.Sequential(
        original_fc,
        new_fc
    )
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.to(device)

    optimizer = AdamW(model.parameters(),lr=1)#intial lr multipled by lambda

    
    def lr_lambda(epoch):
        if epoch < warmup:
            return base_rate/exp_factor * np.exp((epoch) / warmup * np.log(exp_factor))
        else:
            return base_rate* np.exp(-decay * (epoch - warmup))


    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, scheduler, criterion
