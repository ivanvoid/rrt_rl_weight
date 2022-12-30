import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
torch.manual_seed(0)
# Actor-critic
# inspired by
# https://github.com/higgsfield/RL-Adventure-2
# 
# UNET taken https://amaarora.github.io/2020/09/13/unet.html

# class Block(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
#     def forward(self, x):
#         return self.relu(self.conv2(self.relu(self.conv1(x))))

# class Encoder(nn.Module):
#     def __init__(self, chs=(3,64,128,256,512,1024)):
#         super().__init__()
#         self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
#         self.pool       = nn.MaxPool2d(2)
    
#     def forward(self, x):
#         ftrs = []
#         for block in self.enc_blocks:
#             x = block(x)
#             ftrs.append(x)
#             x = self.pool(x)
#         return ftrs

# class Decoder(nn.Module):
#     def __init__(self, chs=(1024, 512, 256, 128, 64)):
#         super().__init__()
#         self.chs         = chs
#         self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
#         self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
#     def forward(self, x, encoder_features):
#         for i in range(len(self.chs)-1):
#             x        = self.upconvs[i](x)
#             enc_ftrs = self.crop(encoder_features[i], x)
#             x        = torch.cat([x, enc_ftrs], dim=1)
#             x        = self.dec_blocks[i](x)
#         return x
    
#     def crop(self, enc_ftrs, x):
#         _, _, H, W = x.shape
#         enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
#         return enc_ftrs

# class UNet(nn.Module):
#     def __init__(self, enc_chs=(3,64,128,256,512,1024), 
#                 dec_chs=(1024,512, 256, 128, 64), 
#                 num_class=2, 
#                 retain_dim=True, 
#                 out_sz=(480, 640)):
#         super().__init__()
#         self.encoder     = Encoder(enc_chs)
#         self.decoder     = Decoder(dec_chs)
#         self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
#         self.retain_dim  = retain_dim
#         self.out_sz = out_sz

#     def forward(self, x):
#         enc_ftrs = self.encoder(x)
#         out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
#         out      = self.head(out)
        
#         if self.retain_dim:
#             out = F.interpolate(out, self.out_sz)
#         return out

from unet import *

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()

        # ACTOR
        # self.actor = UNet()
        in_chans = 3 
        out_chans = 2
        chans = 64 #128
        num_pool_layers = 4 
        drop_prob = 0.0
        self.actor = UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)

        # CRITIC
        # feature extractor for critic and it's head.
        feature_extractor = models.efficientnet_b0(pretrained=False)
        self.feature_extractor = nn.Sequential(*list((feature_extractor.children()))[:-1])
        # self.feature_extractor = nn.Conv2d(1024, 3, 3)
        # non-linear critic head
        self.critic_head = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Tanh()
        )


    def forward(self, x):

        probs = self.actor(x)

        c_out = self.feature_extractor(x)
        c_out = c_out.squeeze()
        value = self.critic_head(c_out)

        return probs, value 