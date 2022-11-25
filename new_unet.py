import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 2)
        self.norm  = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 2)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.norm(self.conv1(x)))))

##########################

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        # self.norm       = nn.BatchNorm2d()
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            # x = self.norm(x)
            x = self.pool(x)
        return ftrs

#########################

class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 7, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

#########################

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(128, 128)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(64, num_class, 1)
        self.spare_conv  = nn.Conv2d(64, 64, 3)
        self.spare_upconv= nn.ConvTranspose2d(64, 64, 11)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        # print(out.shape)
        # out      = self.spare_conv(out)
        out      = self.spare_upconv(self.spare_conv(self.spare_upconv(out)))
        # print(out.shape)
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        # print(out.shape)
        # out = F.softmax(out, 1)
        # out = torch.gt(out, 0.5)
        out = self.sig(out)
        return out

#############################

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen-Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen-Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


#################################

def mse_loss(true, pred):
    true = true.long()
    true = torch.nn.functional.one_hot(true.to(torch.int64), 21).permute((0, 3, 1, 2))
    # print(true.shape)
    # true_flat = torch.flatten(true, start_dim = 2)
    # pred_flat = torch.flatten(pred, start_dim = 2)
    # print(true.shape , pred.shape)
    # pred = F.softmax(pred, 1)
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

