root_path="/home/sysadm/Morphology/segmentaton-master"
import sys
sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as  nn
torch.cuda.empty_cache()

from data_loader.coco import COCOSegmentation
from unet import UNet, mse_ce_combo

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

print( sum(p.numel() for p in model.parameters() if p.requires_grad))

# sample=next(iter(dataloader)) 
# I=sample['image']
# seg_gt = sample['label_seg']
# # inst_gt = sample['label_inst']
# I=I[:1]
# seg_gt=seg_gt[:1]
# # inst_gt=inst_gt[:1]
    
print("Training Starts")
with torch.no_grad():
    train_loss_rec, val_loss_rec = [], []
    for j in range(epochs):
        train_loss, val_loss = 0, 0
        model.train()
        for i, sample in enumerate(train_loader):
            print(i)

            I=sample['image']
            seg_gt = sample['label_seg']
            # inst_gt = sample['label_inst']

            I=I.to(device)
            seg_gt=seg_gt.to(device)
            # inst_gt=inst_gt.to(device)

            optimizer.zero_grad()
            pred_out=model(I)
            loss = mse_ce_combo(pred_out, seg_gt, alpha=0.5)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"{j} train end")
        model.eval()
        for i, sample in enumerate(val_loader):
            I=sample['image']
            seg_gt = sample['label_seg']

            I=I.to(device)
            seg_gt=seg_gt.to(device)

            pred_out=model(I)
            loss = mse_ce_combo(pred_out, seg_gt, alpha=0.5)
            val_loss += loss.item()

        train_loss_rec.append(train_loss)
        val_loss_rec.append(val_loss)

        print(f"For Epoch {j+1} train loss {train_loss} and validation loss {val_loss}")

print("Training Ends-----")

plt.plot(range(len(train_loss_rec)), train_loss_rec)
plt.plot(range(len(val_loss_rec)), val_loss_rec)
plt.show()