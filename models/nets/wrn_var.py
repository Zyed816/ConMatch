from .wrn import WideResNet


class build_WideResNetVar:
    first_stride = 2
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
