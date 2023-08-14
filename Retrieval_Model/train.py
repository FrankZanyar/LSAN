from model import TestModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import split_dataset, ImageDataSet, RandomIdentitySampler,loadF
import torch.nn as nn
from tqdm import tqdm
from test import test
import torch.optim as optim
from LossFunction import Proxy_Anchor

batch_size=100
num_class=10
epochs=600
IMG_NUM=1000

Features=loadF('../data/features/DCTHistfeats.csv',IMG_NUM)
Labels=np.load('../data/Label/Label_1K.npy')
y=np.zeros(IMG_NUM)
for i in range(IMG_NUM):
    y[i]=Labels[i][0]
Xp_train, yp_train, Xp_test, yp_test = split_dataset(Features,y)

train_data = ImageDataSet(Xp_train, yp_train.tolist())
test_data = ImageDataSet(Xp_test, yp_test)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                          sampler=RandomIdentitySampler(train_data, batch_size, num_class))
valid_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

model=TestModel(word_dim=64,n_blocks=8,n_classes=10,represent_dim=128)

optimizer = optim.AdamW(params=model.parameters(), lr=0.0003)
cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=2e-5, last_epoch=-1)

model = model.cuda()

#test_stage(model, Xp_test, yp_test)

criterion = nn.CrossEntropyLoss()
PAL = Proxy_Anchor(10,128)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
        
    print("第%d个epoch的学习率:%f" % (epoch, optimizer.param_groups[0]['lr']))

    for data, label in tqdm(train_loader):
        model.train()
        data = data.cuda()
        label = label.cuda()

        fea, output = model(data)
        loss = 0.8*criterion(output, label) + PAL(fea, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    cosineScheduler.step()
    print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")
    #if epoch>=400 and epoch%50==0:
    #    torch.save(model.state_dict(), './Model'+str(epoch)+'.pth')

    # save model
torch.save(model.state_dict(), './Model.pth')
test(model,Xp_train,yp_train,Xp_test,yp_test,100)
test(model,Xp_train,yp_train,Xp_test,yp_test,300)