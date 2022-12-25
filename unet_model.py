""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet_new(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, bilinear=False):
        super(UNet_new, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = logits.view(A.size(0), -1)
        # logits -= logits.min(1, keepdim=True)[0]
        # logits /= logits.max(1, keepdim=True)[0]
        # logits = logits.view(batch_size, height, width)
        out = self.sig(12 * logits - 6)
        return out



if __name__=="__main__":
    model=UNet_new()
    I=torch.randn((4,3,128,128))
    
    
