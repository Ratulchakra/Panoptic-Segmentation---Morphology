root_path="/home/sysadm/Documents/ratul_env/Morphology/segmentation_master"
import sys
sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as  nn
torch.cuda.empty_cache()

from data_loader.coco import COCOSegmentation

from losses import * 
from unet_model  import * 
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm

#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

epochs = int(input("Epochs:"))
lr = 0.001
batch_size = 32
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

##########################
model = torch.load("/home/sysadm/Documents/ratul_env/Morphology/remodeled_unet_41_COCO.pt")
##########################


optimizer = optim.Adam(model.parameters(), lr=lr)

print( sum(p.numel() for p in model.parameters() if p.requires_grad))
    
print("Training Starts")
with torch.no_grad():
    train_loss_rec, val_loss_rec = [], []
    for j in range(epochs):
        train_loss, val_loss = 0, 0
        model.train()
        n = len(train_loader)
        tbar = tqdm(enumerate(train_loader), total=n)
        for i, sample in tbar:
            I=sample['image']
            seg_gt = sample['label_seg']
            # inst_gt = sample['label_inst']

            I=I.to(device)
            seg_gt=seg_gt.to(device)
            # inst_gt=inst_gt.to(device)

            optimizer.zero_grad()
            pred_out=model(I)
            loss=torch.nn.CrossEntropyLoss()(pred_out,seg_gt.long()) + focal_tv_loss(pred_out, seg_gt)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Training Now Batch: {}/{} , with Batch Loss {}'. \
                        format(i, n, loss.item()) )

        train_loss_rec.append(train_loss/n)
        print(f"For Epoch {j+1} train loss {train_loss_rec[-1]}")
        if (j+1) % 30 == 0:
        	torch.save(model, f"/home/sysadm/Documents/ratul_env/Morphology/remodeled_unet_trained_{j+1}_COCO.pt")

print("Training Ends-----")

torch.save(model, f"/home/sysadm/Documents/ratul_env/Morphology/remodeled_unet_trained_{epochs}_COCO.pt")
print("Model Saved")

plt.plot(range(len(train_loss_rec)), train_loss_rec)
plt.plot(range(len(val_loss_rec)), val_loss_rec)
plt.show()



