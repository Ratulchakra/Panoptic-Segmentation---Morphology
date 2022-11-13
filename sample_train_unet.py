root_path="/home/sysadm/Morphology/segmentaton-master"
import sys
sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as  nn
torch.cuda.empty_cache()

from data_loader.coco import COCOSegmentation
from unet import UNet, dice_loss, mse_loss, cross_entropy_custom, trial_cross_entropy, mse_ce_combo

import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

epochs = int(input("Epochs:"))
lr = 0.001
batch_size = 4
parser = argparse.ArgumentParser() 
args = parser.parse_args()
args.base_size = 128
args.crop_size = 128

# Train
coco_train=COCOSegmentation(args, split='train', year='2017')
train_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=0) 
# Val
coco_val=COCOSegmentation(args, split='val', year='2017')
val_loader = DataLoader(coco_val, batch_size=batch_size, shuffle=True, num_workers=0) 
# Test


model=UNet(num_class= 21, retain_dim=True, out_sz=(128, 128)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()

print( sum(p.numel() for p in model.parameters() if p.requires_grad))

# sample=next(iter(dataloader)) 
# I=sample['image']
# seg_gt = sample['label_seg']
# # inst_gt = sample['label_inst']
# I=I[:1]
# seg_gt=seg_gt[:1]
# # inst_gt=inst_gt[:1]
    
# print("Training Starts")
# torch.no_grad()
# for i in range(epochs):
#     train_loss, val_loss = 0, 0
#     model.train()
#     for i, sample in enumerate(train_loader):

#         I=sample['image']
#         seg_gt = sample['label_seg']
#         # inst_gt = sample['label_inst']

#         I=I.to(device)
#         seg_gt=seg_gt.to(device)
#         # inst_gt=inst_gt.to(device)

#         optimizer.zero_grad()
#         pred_out=model(I)
#         loss,gt_mask = dice_loss(seg_gt, pred_out)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     model.eval()
#     for i, sample in enumerate(val_loader):
#         I=sample['image']
#         seg_gt = sample['label_seg']

#         I=I.to(device)
#         seg_gt=seg_gt.to(device)

#         pred_out=model(I)
#         loss,gt_mask = dice_loss(seg_gt, pred_out)
#         val_loss += loss.item()

#     print(f"For Epoch {i+1} train loss {train_loss} and validation loss {val_loss}")

# print("Training Ends-----")


sample=next(iter(train_loader)) 
I=sample['image']
seg_gt = sample['label_seg']
# inst_gt = sample['label_inst']
I=I[:1]
seg_gt=seg_gt[:1]
# inst_gt=inst_gt[:1]
I = I.to(device)
seg_gt = seg_gt.to(device)
# One hot encoding
# seg_gt = torch.nn.functional.one_hot(seg_gt.to(torch.int64), 21).transpose(1, 3)

torch.no_grad()
model.train()
train_loss = []
for i in range(epochs):
    optimizer.zero_grad()
    pred_out = model(I)
    loss = mse_ce_combo(pred_out, seg_gt.long())
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    print(f"For epoch {i+1} loss {loss.item()}")

print(pred_out.shape)
plt.plot(range(len(train_loss)), train_loss)
plt.show()
