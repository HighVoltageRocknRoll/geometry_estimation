import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, batch_normalization=True):
        super(BasicBlock, self).__init__()
        
        conv1 = conv3x3(inplanes, planes)
        if batch_normalization:
            bn1 = nn.BatchNorm2d(planes)
        relu1 = nn.ReLU(inplace=True)

        conv2 = conv3x3(planes, planes)
        if batch_normalization:
            bn2 = nn.BatchNorm2d(planes)
        
        if inplanes != planes:
            if batch_normalization:
                self.downsample = nn.Sequential([conv1x1(inplanes, planes), nn.BatchNorm2d(planes)])
            else:
                self.downsample = conv1x1(inplanes, planes)
        else:
            self.downsample = None
            
        self.relu = nn.ReLU(inplace=True)

        if batch_normalization:
            layers = [conv1, bn1, relu1, conv2, bn2]
        else:
            layers = [conv1, relu1, conv2]
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        out = self.residual(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MERegression4(nn.Module):
    def __init__(self, output_dim=3, use_cuda=True, batch_normalization=True, channels=[4,32,32,64,64,128], zero_init_residual=True):
        super(MERegression4, self).__init__()

        self.conv1 = conv3x3(channels[0], channels[1])
        self.conv2 = conv3x3(channels[1], channels[1])
        self.conv3 = conv3x3(channels[1], channels[1])
        self.relu = nn.ReLU(inplace=True)

        channels = channels[1:]
        nn_modules = []
        num_blocks = len(channels) - 1

        for i in range(num_blocks):
            nn_modules.append(BasicBlock(channels[i], channels[i+1], batch_normalization=batch_normalization))
        self.residual_layers = nn.Sequential(*nn_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels[-1], output_dim)

        if zero_init_residual and batch_normalization:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.residual_layers(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x