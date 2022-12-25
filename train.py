root_path="/home/sysadm/Documents/ratul_env/Morphology/segmentation_master/"
import sys
sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as  nn
torch.cuda.empty_cache()

from data_loader.coco import COCOSegmentation

# from losses import * 
from unet_model  import * 
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm

from losses.sk_loss import *

#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

epochs = 1000000 
lr = 0.0001
batch_size = 32
parser = argparse.ArgumentParser() 
args = parser.parse_args()
args.base_size = 128
args.crop_size = 128

# Train
coco_train=COCOSegmentation(args, split='train', year='2017')
train_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=0) 

"""
# Val
coco_val=COCOSegmentation(args, split='val', year='2017')
val_loader = DataLoader(coco_val, batch_size=batch_size, shuffle=True, num_workers=0) 
"""

#model=UNet(num_class= 21, retain_dim=True, out_sz=(128, 128)).to(device)
model=UNet_new().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

print( sum(p.numel() for p in model.parameters() if p.requires_grad))


print("Training Starts")
with torch.no_grad():
    train_loss_rec = []
    for j in range(epochs):
        train_loss = 0
        model=model.train()
        n = len(train_loader)
        tbar = tqdm(enumerate(train_loader), total=n)
        for i, sample in tbar:
            I=sample['image']
            seg_gt = sample['label_seg']
            inst_gt = sample['label_inst']

            I=I.to(device)
            seg_gt=seg_gt.to(device)
            inst_gt=inst_gt.to(device)

            optimizer.zero_grad()
            pred_out=model(I)
            loss = lp_norm(pred_out, seg_gt, inst_gt, p = 8)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Training Now Batch: {}/{} , with Batch Loss {}'. \
                                  format(i, n, loss.item()) )
            if (i+1) % 100 == 0:
                torch.save(model, f"/home/sysadm/Documents/ratul_env/Morphology/temp_unet_{i+1}.pt")
                # print(f"Model saved at {i+1}")
        print(f"{j} train end")
        train_loss_rec.append(train_loss/n)


        print(f"For Epoch {j+1} train loss {train_loss_rec[-1]}")
        if (j+1) % 30 == 0:
            torch.save(model, f"/home/sysadm/Documents/ratul_env/Morphology/remodeled_unet_trained_{j+1}_COCO.pt")
            print(f"Model saved at {j+1}")

print("Training Ends-----")

torch.save(model, f"/home/sysadm/Documents/ratul_env/Morphology/remodeled_unet_{epochs}_COCO.pt")
print(f"Model Saved")

plt.plot(range(len(train_loss_rec)), train_loss_rec)
plt.plot(range(len(val_loss_rec)), val_loss_rec)
plt.show()
