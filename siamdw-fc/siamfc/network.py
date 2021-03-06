from __future__ import absolute_import, division, print_function
from torchvision.models import resnet18
import timm
from timm.data.transforms_factory import create_transform
import torch
import torch.nn as nn
import os
from transformer import ViT
import cv2

__all__ = ['SiamFCNet']


class SiamFCNet(nn.Module):

    def __init__(self, backbone, head):
        super(SiamFCNet, self).__init__()
        self.features = backbone
        self.head = head
        self.attentionz=ViT(image_size=(8,8),patch_size=(1,1),in_channels=512,dim=512)





    def forward(self, z, x):

        x=self.features(x)
        z=self.features(z)


        z=self.attentionz(z)


        return self.head(z, x)  

