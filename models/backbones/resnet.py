import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1  
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)  
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):

        identity = x 

        if self.downsample is not None:
            identity = self.downsample(x)
        
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        out = self.conv2(out)
        out = self.bn2(out)
        

        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4  
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               stride=1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(
            self, 
            block, 
            block_nums, 
            in_channels=3, 
            last_stride = 2
        ):
        super().__init__()
        self.in_planes = 64  
        
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        
        self.layer1 = self._make_layer(block, 64, block_nums[0], stride=1)
        self.layer2 = self._make_layer(block, 128, block_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_nums[3], stride=last_stride)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_planes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, out_channels, stride, downsample))
        
        self.in_planes = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, out_channels))
        
        return nn.Sequential(*layers)
    
    def _extract_stem_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward(self, x):
        output = self._extract_stem_features(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output
    

def resnet50(in_channels=3, last_stride = 2):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, last_stride=last_stride)

