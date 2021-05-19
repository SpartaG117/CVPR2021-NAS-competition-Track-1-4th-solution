import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS_layer1 = [4, 8, 12, 16]
CHANNELS_layer2_7 = [4, 8, 12, 16]
CHANNELS_layer8_13 = [4, 8, 12, 16, 20, 24, 28, 32]
CHANNELS_layer14_19 = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]


class Conv2dBlock(nn.Module):
    def __init__(self, in_choices, out_choices, kernel_size, stride, padding=1, bias=False, norm=True, relu=True):
        super().__init__()
        self.in_choices = in_choices
        self.out_choices = out_choices
        self.norm_flag = norm
        self.relu = relu
        self.max_in = max(in_choices)
        self.convs = nn.ModuleDict()
        if self.norm_flag:
            self.norms = nn.ModuleDict()
        else:
            self.norms = None
            
        for o in self.out_choices:
            self.convs[str(o)] = nn.Conv2d(self.max_in, o, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            if self.norm_flag:
                self.norms[str(o)] = nn.BatchNorm2d(o)

    def forward(self, x, out_channel):
        in_channel = x.size(1)
        assert in_channel in self.in_choices
        assert out_channel in self.out_choices

        conv = self.convs[str(out_channel)]
        x = F.conv2d(x, conv.weight[:, :in_channel, :, :], conv.bias,
                     conv.stride, conv.padding, conv.dilation, conv.groups)
        if self.norm_flag:
            x = self.norms[str(out_channel)](x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_choices, mid_choices, out_choices, stride=1, downsampling=False):
        super(BasicBlock, self).__init__()
        
        self.downsampling = downsampling
        self.adaptive_pooling = False
        self.in_choices = in_choices
        self.out_choices = out_choices
        self.mid_choices = mid_choices
        
        self.conv1 = Conv2dBlock(in_choices=in_choices,
                                 out_choices=mid_choices,
                                 kernel_size=3,
                                 stride=stride,
                                 padding=1,
                                 bias=False,
                                 norm=True,
                                 relu=True)
        
        self.conv2 = Conv2dBlock(in_choices=mid_choices,
                                 out_choices=out_choices,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=False,
                                 norm=True,
                                 relu=False)
        
        self.downsample = Conv2dBlock(in_choices=in_choices,
                                      out_choices=out_choices,
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      bias=False,
                                      norm=True,
                                      relu=False)
    
    def forward(self, x, mid_channel, out_channel):
        in_channel = x.size(1)
        residual = x
        out = self.conv1(x, out_channel=mid_channel)
        out = self.conv2(out, out_channel=out_channel)
        residual = self.downsample(residual, out_channel=out_channel)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, len_list, num_classes=100, pretrain_path=None):
        super(ResNet, self).__init__()
        self.len_list = len_list
        
        self.stem = Conv2dBlock(in_choices=[3],
                                out_choices=len_list[0],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                                norm=True,
                                relu=True)
        
        self.layer1 = self.make_layer(self.len_list[:7], block=blocks[0], stride=1)
        self.layer2 = self.make_layer(self.len_list[6:13], block=blocks[1], stride=2)
        self.layer3 = self.make_layer(self.len_list[12:19], block=blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(max(self.len_list[18]), num_classes, bias=True)
        self.reset_parameters()
        if pretrain_path:
            self.load_from_weight_sharing(pretrain_path)
            
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def load_conv(self, name, module, state_dict, in_choices, out_choices):
        conv_weight = state_dict[name+'.weight']
        with torch.no_grad():
            for o in out_choices:
                op_name = str(o)
                module.convs[op_name].weight.copy_(conv_weight.data[:o, :, :, :])
                if module.norms:
                    norm = module.norms[op_name]
                    state_name = name+".norms."+str(o)+'.'
                    norm.weight.copy_(state_dict[state_name+'weight'])
                    norm.bias.copy_(state_dict[state_name + 'bias'])
                    norm.running_mean.copy_(state_dict[state_name + 'running_mean'])
                    norm.running_var.copy_(state_dict[state_name + 'running_var'])
                    norm.num_batches_tracked.copy_(state_dict[state_name + 'num_batches_tracked'])
    
    def load_layer(self, name, module, state_dict):
        for l, block in enumerate(module):
            state_name = name + '.' + str(l) + '.'
            self.load_conv(state_name+'conv1', block.conv1, state_dict, block.in_choices, block.mid_choices)
            self.load_conv(state_name+'conv2', block.conv2, state_dict, block.mid_choices, block.out_choices)
            self.load_conv(state_name+'downsample', block.downsample, state_dict, block.in_choices, block.out_choices)


    def load_from_weight_sharing(self, path):
        state_dict = torch.load(path, map_location='cpu')['model']
        self.load_conv('stem', self.stem, state_dict, [3], self.len_list[0])
        self.load_layer('layer1', self.layer1, state_dict)
        self.load_layer('layer2', self.layer2, state_dict)
        self.load_layer('layer3', self.layer3, state_dict)
        with torch.no_grad():
            self.fc.weight.copy_(state_dict['fc.weight'])
            self.fc.bias.copy_(state_dict['fc.bias'])
    
    
    def make_layer(self, len_list, block, stride):
        layers = []
        layers.append(BasicBlock(in_choices=len_list[0],
                                 mid_choices=len_list[1],
                                 out_choices=len_list[2],
                                 stride=stride,
                                 downsampling=True))
        for i in range(1, block):
            prev = i * 2
            mid = i * 2 + 1
            rear = i * 2 + 2
            layers.append(BasicBlock(in_choices=len_list[prev], mid_choices=len_list[mid],
                                     out_choices=len_list[rear], stride=1))
        return nn.ModuleList(layers)
    
    def get_bn(self, rng):
        bns = []
        bns.append(self.stem.norms[str(rng[0])])
        idx = 1
        for m in self.modules():
            if isinstance(m, BasicBlock):
                norm1 = m.conv1.norms
                assert isinstance(norm1, nn.ModuleDict)
                bns.append(norm1[str(rng[idx])])
                idx += 1

                norm2 = m.conv2.norms
                assert isinstance(norm2, nn.ModuleDict)
                bns.append(norm2[str(rng[idx])])
                norm_ds = m.downsample.norms
                assert isinstance(norm2, nn.ModuleDict)
                bns.append(norm_ds[str(rng[idx])])
                idx += 1
        assert idx == 19
        return bns
    
    def forward(self, x, rngs):
        assert len(rngs) == len(self.len_list)
        
        x = self.stem(x, out_channel=rngs[0])
        
        for b, i in enumerate(range(1, 7, 2)):
            x = self.layer1[b](x, rngs[i], rngs[i + 1])
        
        for b, i in enumerate(range(7, 13, 2)):
            x = self.layer2[b](x, rngs[i], rngs[i + 1])
        
        for b, i in enumerate(range(13, 19, 2)):
            x = self.layer3[b](x, rngs[i], rngs[i + 1])
        
        x = self.avgpool(x).flatten(1)
        x = F.linear(x, self.fc.weight[:, :rngs[18]])
        return x


len_list = []
len_list.extend([CHANNELS_layer2_7 for _ in range(7)])
len_list.extend([CHANNELS_layer8_13 for _ in range(6)])
len_list.extend([CHANNELS_layer14_19 for _ in range(6)])


