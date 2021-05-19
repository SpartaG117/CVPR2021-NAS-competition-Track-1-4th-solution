import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS_layer1 = [4, 8, 12, 16]
CHANNELS_layer2_7 = [4, 8, 12, 16]
CHANNELS_layer8_13 = [4, 8, 12, 16, 20, 24, 28, 32]
CHANNELS_layer14_19 = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        in_choices = kwargs.pop("in_choices", None)
        out_choices = kwargs.pop("out_choices", None)
        norm = kwargs.pop("norm", False)
        relu = kwargs.pop("relu", False)
        super().__init__(*args, **kwargs)
        
        self.in_choices = in_choices
        self.out_choices = out_choices
        self.norm_flag = norm
        self.relu = relu
        
        if self.norm_flag:
            if self.out_choices is not None:
                self.norms = nn.ModuleDict()
                for channel in self.out_choices:
                    self.norms[str(channel)] = nn.BatchNorm2d(channel, track_running_stats=True)
            else:
                self.norms = nn.BatchNorm2d(self.out_channels)
                
    def forward(self, x, out_channel=None):
        in_channel = x.size(1)
        assert in_channel in self.in_choices
        weight = self.weight[:, :in_channel, :, :]
            
        x = F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        
        if out_channel is not None:
            assert out_channel in self.out_choices
            x = x[:, :out_channel, :, :]
            if self.norm_flag:
                x = self.norms[str(out_channel)](x)
            if self.relu:
                x = F.relu(x)
            return x
        
        output = {}
        for o in self.out_choices:
            out_o = x[:, :o, :, :].clone()
            if self.norms is not None:
                out_o = self.norms[str(o)](out_o)
            if self.relu:
                out_o = F.relu(out_o)
            output[o] = out_o
        return output


class BasicBlock(nn.Module):
    def __init__(self, in_choices, mid_choices, out_choices, stride=1, downsampling=False):
        super(BasicBlock, self).__init__()
        
        self.downsampling = downsampling
        self.adaptive_pooling = False
        self.in_choices = in_choices
        self.out_choices = out_choices
        self.mid_choices = mid_choices
        self.max_in_channel = max(self.in_choices)
        self.max_mid_channel = max(self.mid_choices)
        self.max_out_channel = max(self.out_choices)

        self.conv1 = Conv2d(in_choices=in_choices,
                            out_choices=mid_choices,
                            in_channels=self.max_in_channel,
                            out_channels=self.max_mid_channel,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False,
                            norm=True,
                            relu=True)
        
        self.conv2 = Conv2d(in_choices=mid_choices,
                            out_choices=out_choices,
                            in_channels=self.max_mid_channel,
                            out_channels=self.max_out_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                            norm=True,
                            relu=False)
        
        self.downsample = Conv2d(in_choices=in_choices,
                                 out_choices=out_choices,
                                 in_channels=self.max_in_channel,
                                 out_channels=self.max_out_channel,
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
    def __init__(self, blocks, len_list, num_classes=100):
        super(ResNet, self).__init__()
        self.len_list = len_list
        
        self.stem = Conv2d(in_choices=[3],
                           out_choices=len_list[0],
                           in_channels=3,
                           out_channels=max(len_list[0]),
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
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
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
    
    def get_bn(self, rng, calibration=True):
        if not calibration:
            return None
        
        bns = []
        norm = self.stem.norms[str(rng[0])]
        if not norm.track_running_stats:
            norm.track_running_stats = True
            delattr(norm, 'running_mean')
            delattr(norm, 'running_var')
            delattr(norm, 'num_batches_tracked')
    
            norm.register_buffer('running_mean', torch.zeros(norm.num_features).to(norm.weight.device))
            norm.register_buffer('running_var', torch.ones(norm.num_features).to(norm.weight.device))
            norm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).to(norm.weight.device))
            
        bns.append(norm)
        idx = 1
        for m in self.modules():
            if isinstance(m, BasicBlock):
                norm1 = m.conv1.norms
                assert isinstance(norm1, nn.ModuleDict)
                norm = norm1[str(rng[idx])]
                if not norm.track_running_stats:
                    norm.track_running_stats = True
                    delattr(norm, 'running_mean')
                    delattr(norm, 'running_var')
                    delattr(norm, 'num_batches_tracked')

                    norm.register_buffer('running_mean', torch.zeros(norm.num_features).to(norm.weight.device))
                    norm.register_buffer('running_var', torch.ones(norm.num_features).to(norm.weight.device))
                    norm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).to(norm.weight.device))
                bns.append(norm)
                idx += 1
            
                norm2 = m.conv2.norms
                assert isinstance(norm2, nn.ModuleDict)
                norm = norm2[str(rng[idx])]
                if not norm.track_running_stats:
                    norm.track_running_stats = True
                    delattr(norm, 'running_mean')
                    delattr(norm, 'running_var')
                    delattr(norm, 'num_batches_tracked')
                    # norm.running_mean = torch.zeros(norm.num_features)
                    # norm.running_var = torch.ones(norm.num_features)
                    # norm.num_batches_tracked = torch.tensor(0, dtype=torch.long)
                    norm.register_buffer('running_mean', torch.zeros(norm.num_features).to(norm.weight.device))
                    norm.register_buffer('running_var', torch.ones(norm.num_features).to(norm.weight.device))
                    norm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).to(norm.weight.device))
                bns.append(norm)
                
                norm_ds = m.downsample.norms
                assert isinstance(norm2, nn.ModuleDict)
                norm = norm_ds[str(rng[idx])]
                if not norm.track_running_stats:
                    norm.track_running_stats = True
                    delattr(norm, 'running_mean')
                    delattr(norm, 'running_var')
                    delattr(norm, 'num_batches_tracked')
                    norm.register_buffer('running_mean', torch.zeros(norm.num_features).to(norm.weight.device))
                    norm.register_buffer('running_var', torch.ones(norm.num_features).to(norm.weight.device))
                    norm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).to(norm.weight.device))
                bns.append(norm)
                idx += 1
        assert idx == 19
        return bns
    
    def forward(self, x, rngs):
        assert len(rngs) == len(self.len_list)
        
        x = self.stem(x, out_channel=rngs[0])
        
        for b, i in enumerate(range(1, 7, 2)):
            x = self.layer1[b](x, rngs[i], rngs[i+1])
            
        for b, i in enumerate(range(7, 13, 2)):
            x = self.layer2[b](x, rngs[i], rngs[i+1])

        for b, i in enumerate(range(13, 19, 2)):
            x = self.layer3[b](x, rngs[i], rngs[i+1])

        x = self.avgpool(x).flatten(1)
        x = F.linear(x, self.fc.weight[:, :rngs[18]])
        return x
    
len_list = []
len_list.extend([CHANNELS_layer2_7 for _ in range(7)])
len_list.extend([CHANNELS_layer8_13 for _ in range(6)])
len_list.extend([CHANNELS_layer14_19 for _ in range(6)])
