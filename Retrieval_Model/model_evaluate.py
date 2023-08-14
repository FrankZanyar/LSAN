import torchvision.models
import torch
from thop import profile
from thop import clever_format
from model import TestModel
model = TestModel(word_dim=64,n_blocks=8,n_classes=100,represent_dim=128)
device = torch.device('cpu')
model.to(device)
myinput = torch.zeros((1, 192, 64)).to(device)
flops, params = profile(model.to(device), inputs=(myinput,))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)
