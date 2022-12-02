import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


#############################

def mse_loss(true, pred):
    true = true.long()
    true = torch.nn.functional.one_hot(true.to(torch.int64), 21).permute((0, 3, 1, 2))
    x = (true - pred)**2
    loss = torch.mean(x)
    return loss

###############################

def trial_cross_entropy(predicton, target):
    """
    input should be: (batch, class, H, W)
    target should be: (batch, H, W)
    """
    target = target.long()
    loss = torch.nn.CrossEntropyLoss()
    return loss(predicton, target)

#################################

def mse_ce_combo(prediction, target, alpha = 0.7):
    a = mse_loss(target, prediction)
    b = trial_cross_entropy(prediction, target)
    return alpha*a + (1-alpha)*b

###################################

#PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        inputs = F.softmax(inputs,dim=1)       
        targets = torch.nn.functional.one_hot(targets.to(torch.int64), 21).permute((0, 3, 1, 2))

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

#######################################

def focal_tv_loss(pred, target):
    ftv = FocalTverskyLoss()
    return ftv(pred, target)

############################# evaluation #############################


def mIoU (outputs: torch.Tensor, labels: torch.Tensor):
    labels = torch.nn.functional.one_hot(labels.to(torch.int64), 21).permute((0, 3, 1, 2))
    outs = outputs.clone().detach()
    # outs = F.sigmoid(outs)
    outs = torch.argmax(outs, axis = 1)
    # print(outs.shape)
    outs = torch.nn.functional.one_hot(outs.to(torch.int64), 21).permute((0, 3, 1, 2))

    intersection = torch.logical_and(labels, outs)
    union = torch.logical_or(labels, outs)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score


def precision (outputs: torch.Tensor, labels: torch.Tensor):
    labels = torch.nn.functional.one_hot(labels.to(torch.int64), 21).permute((0, 3, 1, 2))
    outs = outputs.clone().detach()
    # outs = F.sigmoid(outs)
    outs = torch.argmax(outs, axis = 1)
    # print(outs.shape)
    outs = torch.nn.functional.one_hot(outs.to(torch.int64), 21).permute((0, 3, 1, 2))

    intersection = torch.logical_and(labels, outs)
    return torch.sum(intersection)/torch.sum(outs)

def recall (outputs: torch.Tensor, labels: torch.Tensor):
    labels = torch.nn.functional.one_hot(labels.to(torch.int64), 21).permute((0, 3, 1, 2))
    outs = outputs.clone().detach()
    # outs = F.sigmoid(outs)
    outs = torch.argmax(outs, axis = 1)
    # print(outs.shape)
    outs = torch.nn.functional.one_hot(outs.to(torch.int64), 21).permute((0, 3, 1, 2))

    intersection = torch.logical_and(labels, outs)
    return torch.sum(intersection)/torch.sum(labels)

def recognition_quality (outputs: torch.Tensor, labels: torch.Tensor):
    labels = torch.nn.functional.one_hot(labels.to(torch.int64), 21).permute((0, 3, 1, 2))
    outs = outputs.clone().detach()
    # outs = F.sigmoid(outs)
    outs = torch.argmax(outs, axis = 1)
    # print(outs.shape)
    outs = torch.nn.functional.one_hot(outs.to(torch.int64), 21).permute((0, 3, 1, 2))

    intersection = torch.logical_and(labels, outs)
    return torch.sum(2*intersection)/torch.sum(labels + outs)

def segmentation_quality (outputs: torch.Tensor, labels: torch.Tensor):
    pass

##########################################


if __name__=="__main__":
    model=UNet(num_class=21)
    I=torch.randn((4,3,128,128))
    out=model(I)
    print(out.shape)

