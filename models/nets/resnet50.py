import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Wrapper, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)
        self.feat_proj = nn.Linear(in_features, 128)

    def forward(self, x, is_tuple=False):
        feat = self.backbone(x)
        logits = self.fc(feat)
        if is_tuple:
            return self.feat_proj(feat), logits
        return logits


class build_ResNet50:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        return ResNet50Wrapper(num_classes)
