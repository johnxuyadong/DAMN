import torch
import torch.nn as nn
import torch.nn.functional as F



def gn(planes, channel_per_group=4, max_groups=32):
    groups = planes // channel_per_group
    return nn.GroupNorm(min(groups, max_groups), planes)

class TAM(nn.Module): # TAM
    reduction = 4
    def __init__(self, k=48):
        super(TAM, self).__init__()
        k_mid = int(k // self.reduction)
        self.attention = nn.Sequential(
            nn.Conv1d(k, k_mid, 1, 1, bias=False),
            gn(k_mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(k_mid, k, 1, 1, bias=False),
            gn(k),
            nn.Sigmoid(),
        )
        self.block = nn.Sequential(nn.Conv1d(k, k, 3, 1, 1, bias=False), gn(k), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.attention(x)
        out = torch.add(x, torch.mul(x, out))
        out = self.block(out)
        return out



########################################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x




##########################################################################################################################



class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv1d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm1d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)



class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, out_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel)
                )

    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class Spatialatt(nn.Module):
    def __init__(self,channels_in):
        super(Spatialatt, self).__init__()
        kernel_size = 3
        self.spatial = BasicConv(channels_in, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_out = self.spatial(x)
        scale = torch.sigmoid(x_out) # broadcasting

        return scale

class VAMM(nn.Module):
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4, channels_in=16):
        super(VAMM, self).__init__()
        self.se = SELayer(channels_in, reduction=4)
        self.inplanes1 = 16
        self.spaitla = Spatialatt(channels_in=channels_in)
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.ModuleList([
                DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
                ])
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv1d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
                convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
                nn.Conv1d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
                )


    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)  # gather torch.Size([2, 16, 256])

        ### ChannelGate
        d = self.gap(gather) # d torch.Size([2, 16, 1])
        d = self.fc2(self.fc1(d)) # d torch.Size([2, 64, 1])
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1) # d torch.Size([2, 4, 16, 1])

        s = self.convs(gather).unsqueeze(1) # [2, 1, 1, 256])

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)]))	+ x


####################################################################################################################################################
class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel, kernel):
        super(RFB, self).__init__()
        self.myNet = nn.Sequential(
            nn.Linear(out_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid()
        )
        # self.maxpooling = nn.MaxPool1d(kernel_size=kernel, stride=4, padding=0)
        self.maxpooling = nn.MaxPool1d(kernel_size=kernel, stride=1, padding=0)
        self.gap = nn.AvgPool1d(kernel)
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv1d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, kernel_size=5, padding=2),
            BasicConv1d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),
            BasicConv1d(out_channel, out_channel, kernel_size=7, padding=3),
            BasicConv1d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv1d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv1d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        con1 = self.relu(x_cat + self.conv_res(x))

        # calculate global means
        x1 = con1
        x_raw = x1
        x1 = torch.abs(x1)
        x_abs = x1
        x1 = self.gap(x1)
        x1 = torch.flatten(x1, 1)
        average = x1
        x1 = self.myNet(x1)
        x1 = torch.mul(average, x1)
        x1 = x1.unsqueeze(2)
        # 软阈值
        sub = x_abs - x1
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x1 = torch.mul(torch.sign(x_raw), n_sub)

        res1 = self.conv_res(x)

        x1_add = torch.add(x1, res1)
        x1_add = self.relu(x1_add)
        return x1_add



############################################################################################################


class VAMM_backbone(nn.Module):
    def __init__(self, pretrained=None, num_classes=7):
        super(VAMM_backbone, self).__init__()
        ####################   AFIM
        merge_convs, fcs, convs = [], [], []
        k = 16
        self.len = 3
        for m in range(1):
            merge_convs.append(nn.Sequential(
                        nn.Conv1d(k, k//4, 1, 1, bias=False),
                        gn(k//4),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(k//4, k, 1, 1, bias=False),
                        gn(k),
                    ))
            fcs.append(nn.Sequential(
                    nn.Linear(k, k // 4, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(k // 4, self.len, bias=False),
                ))
            convs.append(nn.Sequential(nn.Conv2d(k, k, 3, 1, 1, bias=False), gn(k), nn.ReLU(inplace=True)))
        self.merge_convs = nn.ModuleList(merge_convs)

        self.merge_convs = nn.ModuleList(merge_convs)
        self.fcs = nn.ModuleList(fcs)
        self.convs = nn.ModuleList(convs)
        self.gapAFI = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu =nn.ReLU(inplace=True)

        self.conv_res = BasicConv1d(48, 48, 1)


        self.maxpool = nn.AvgPool1d(kernel_size=64, stride=1, padding=0)
        self.inplanes1 = 48      ###########输入
        self.inplanes2 = 16      ###########输入
        self.inplanes3 = 16      ###########输入
        self.AFR1 = self._make_layer1(SEBasicBlock, 48, blocks=1)
        self.AFR2 = self._make_layer2(SEBasicBlock, 16, blocks=1)
        self.AFR3 = self._make_layer3(SEBasicBlock, 16, blocks=1)
        self.tam = TAM()


        channel = 16
        self.rfb1_1 = RFB(16, channel, 1024)
        self.rfb2_1 = RFB(32, channel, 512)
        self.rfb3_1 = RFB(64, channel, 256)
        self.rfb4_1 = RFB(128, channel, 128)



        self.fc = nn.Linear(48, num_classes)

        self.fc1 = nn.Linear(16, num_classes)
        self.fc2 = nn.Linear(32, num_classes)
        self.fc3 = nn.Linear(48, num_classes)
        self.fc4 = nn.Linear(64, num_classes)



        self.avg1 = nn.AvgPool1d(kernel_size=256, stride=1, padding=0)
        self.avg2 = nn.AvgPool1d(kernel_size=128, stride=1, padding=0)
        self.avg3 = nn.AvgPool1d(kernel_size=448, stride=1, padding=0)
        self.avg4 = nn.AvgPool1d(kernel_size=32, stride=1, padding=0)



        self.maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.maxpool = nn.AvgPool1d(kernel_size=256, stride=1, padding=0)
        self.maxpool11 = nn.AvgPool1d(kernel_size=1024, stride=1, padding=0)
        self.maxpool22 = nn.AvgPool1d(kernel_size=512, stride=1, padding=0)
        self.maxpool33 = nn.AvgPool1d(kernel_size=256, stride=1, padding=0)
        self.maxpool44 = nn.AvgPool1d(kernel_size=128, stride=1, padding=0)

        self.gap = nn.AvgPool1d(kernel_size=32, stride=1, padding=0)
        # self.maxpool3 = nn.MaxPool1d(kernel_size=64, stride=1, padding=0)

        self.layer1 = nn.Sequential(
                convbnrelu(1, 16, k=3, s=2, p=1),
                VAMM(16, dilation_level=[1,2,4,8], channels_in=16)
                )
        self.layer2 = nn.Sequential(
                convbnrelu(16, 32, k=3, s=2, p=1),
                # DSConv3x3(16, 32, stride=2),
                VAMM(32, dilation_level=[1,2,4,8],channels_in=32)
                )
        self.layer3 = nn.Sequential(
                convbnrelu(32, 64, k=3, s=2, p=1),
                # DSConv3x3(32, 64, stride=2),
                VAMM(64, dilation_level=[1,2,4,8],channels_in=64),
                )

        self.layer4 = nn.Sequential(
                convbnrelu(64, 128, k=3, s=2, p=1),
                # DSConv3x3(32, 64, stride=2),
                VAMM(128, dilation_level=[1,2,4,8],channels_in=128),
                )


        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained model loaded!')

    def _make_layer1(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes1, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes2, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample))
        self.inplanes2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes2, planes))

        return nn.Sequential(*layers)

    def _make_layer3(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        out1 = self.rfb1_1(out1)
        out2 = self.rfb2_1(out2)
        out3 = self.rfb3_1(out3)

        out1 = self.maxpool2(out1)
        out2 = self.maxpool3(out2)
        out3 = out3
        out4 = torch.cat([out1, out2, out3], dim=1)

        out4= self.maxpool33(out4)
        out4 = out4.squeeze()
        out6 = self.fc3(out4)

        return out6, out4

