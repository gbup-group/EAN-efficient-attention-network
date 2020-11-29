from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['forward_dia_fbresnet50']

pretrained_settings = {
    'fbresnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/fbresnet152-2e20f6b4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class small_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(small_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 20),
                      nn.ReLU(inplace=True),
                      nn.Linear(input_size // 20, 4 * hidden_size))
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.1):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                ih.append(small_cell(input_size, hidden_size))
                hh.append(small_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * torch.sigmoid(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)  # number of layer * batch * hidden
        return hy, cy


# bottleneck
class Attention(nn.Module):
    def __init__(self, ModuleList, block_idx):
        super(Attention, self).__init__()
        self.ModuleList = ModuleList
        if block_idx == 1:
            self.lstm = LSTMCell(64*4, 64*4, 1)
        elif block_idx == 2:
            self.lstm = LSTMCell(128*4, 128*4, 1)
        elif block_idx == 3:
            self.lstm = LSTMCell(256*4, 256*4, 1)
        elif block_idx == 4:
            self.lstm = LSTMCell(512*4, 512*4, 1)

        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.block_idx = block_idx

    def forward(self, x, config):

        binary_code = bin(config)[2:]
        assert len(binary_code) <= len(self.ModuleList), 'the code length is larger than the network length'
        while len(binary_code) < len(self.ModuleList):
            binary_code = '0' + binary_code
        bool_config = tuple([i == '1' for i in binary_code])

        first = True ## flag for the whether the initial block for lstm input
        for idx, layer in enumerate(self.ModuleList):
            x, org = layer(x)
            if bool_config[idx] and first:
                seq = self.GlobalAvg(x)
                seq = seq.view(seq.size(0), seq.size(1))
                ht = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
                ct = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
                ht, ct = self.lstm(seq, (ht, ct))
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                first = False

            elif bool_config[idx]:
                seq = self.GlobalAvg(x)
                seq = seq.view(seq.size(0), seq.size(1))
                ht, ct = self.lstm(seq, (ht, ct))
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))

            x += org
            x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out, residual


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out, residual

class Bottleneck_org(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_org, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = Attention(self._make_layer(block, 64, layers[0]), 1)
        self.layer2 = Attention(self._make_layer(block, 128, layers[1], stride=2), 2)
        self.layer3 = Attention(self._make_layer(block, 256, layers[2], stride=2), 3)
        self.layer4 = Attention(self._make_layer_2(block, 512, layers[3], stride=2),4)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def _make_layer_2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input, configs):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, configs[0])
        x = self.layer2(x, configs[1])
        x = self.layer3(x, configs[2])
        x = self.layer4(x, configs[3])
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input, configs):
        x = self.features(input, configs)
        x = self.logits(x)
        return x

def forward_dia_fbresnet50(num_classes=1000, pretrained = False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

if __name__ == '__main__':
    model = forward_dia_fbresnet50(1000, False).cuda()
    print('    Total params: %.10fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    a = torch.randn((50, 3, 224, 224)).cuda()
    import time

    start = time.time()
    for i in range(1000):
        # model(a, (6, 3, 50, 3))
        model(a, (7, 15, 63, 7))
    print('total time: ', time.time() - start)