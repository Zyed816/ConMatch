import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0,
                 activate_before_residual=False, leaky_slope=0.1,
                 bn_momentum=0.001):
        super(BasicBlock, self).__init__()
        self.equal_in_out = in_planes == out_planes
        self.drop_rate = drop_rate
        self.activate_before_residual = activate_before_residual

        self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv_shortcut = (None if self.equal_in_out else
                              nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                        stride=stride, padding=0, bias=False))

    def forward(self, x):
        if not self.equal_in_out and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))

        residual = x if self.equal_in_out else self.conv_shortcut(x)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        return residual + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 drop_rate=0.0, activate_before_residual=False,
                 leaky_slope=0.1, bn_momentum=0.001):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(
                i == 0 and in_planes or out_planes,
                out_planes,
                i == 0 and stride or 1,
                drop_rate,
                activate_before_residual=(activate_before_residual and i == 0),
                leaky_slope=leaky_slope,
                bn_momentum=bn_momentum,
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0,
                 first_stride=1, leaky_slope=0.1, bn_momentum=0.001):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=first_stride,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], BasicBlock, 1,
                                   drop_rate, activate_before_residual=True,
                                   leaky_slope=leaky_slope,
                                   bn_momentum=bn_momentum)
        self.block2 = NetworkBlock(n, channels[1], channels[2], BasicBlock, 2,
                                   drop_rate, leaky_slope=leaky_slope,
                                   bn_momentum=bn_momentum)
        self.block3 = NetworkBlock(n, channels[2], channels[3], BasicBlock, 2,
                                   drop_rate, leaky_slope=leaky_slope,
                                   bn_momentum=bn_momentum)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=bn_momentum)
        self.relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.feature_dim = channels[3]
        self.embed_dim = 128
        self.feat_proj = (nn.Identity() if self.feature_dim == self.embed_dim
                          else nn.Linear(self.feature_dim, self.embed_dim))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x, is_tuple=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        feat = torch.flatten(out, 1)
        logits = self.fc(feat)
        if is_tuple:
            return self.feat_proj(feat), logits
        return logits


class build_WideResNet:
    first_stride = 1
    depth = 28
    widen_factor = 2
    leaky_slope = 0.1
    bn_momentum = 0.001
    dropRate = 0.0
    use_embed = False
    is_remix = False
    is_con = True
    con_net = 'single'

    def build(self, num_classes):
        return WideResNet(
            num_classes=num_classes,
            depth=self.depth,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            first_stride=self.first_stride,
            leaky_slope=self.leaky_slope,
            bn_momentum=self.bn_momentum,
        )
