import os
import sys
import pathlib
import copy
import logging
from collections import OrderedDict
from addict import Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import transforms
from ocr_utils import ResizeFixedSize
from postprocess import build_post_process
# from torchocr.networks.backbones.DetMobilenetV3 import MobileNetV3
# from torchocr.networks import build_model


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.resize = ResizeFixedSize(736, False) # resize:736*736
        self.post_process = build_post_process(cfg['post_process'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['train']['dataset']['mean'], std=cfg['dataset']['train']['dataset']['std'])
        ])

    def predict(self, img):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.cpu().numpy() #
        box_list, score_list = self.post_process(out, data['shape'])
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            box_list = expand_box(box_list)
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list

# def expand_box(box_list):
#     for b in box_list:
#         b_cx = (b[0][0]+b[1][0]+b[2][0]+b[3][0])/4
#         b_cy = (b[0][1]+b[1][1]+b[2][1]+b[3][1])/4
#         for p in b:
#             p[0] = 1.15*(p[0]-b_cx)+b_cx
#             p[1] = 1.15*(p[1]-b_cy)+b_cy
#     return box_list

def expand_box(box_list):
    # print(box_list)

    for b in box_list:
        # tan = (b[0][1]-b[1][1])/(b[1][0]-b[0][0])
        # print(tan)
        dx = max(0.1*(abs(b[1][0]-b[0][0])), 10)  
        dy = max(0.15*(abs(b[2][1]-b[1][1])), 4)
        b[0][0] = int(b[0][0] - dx)
        b[1][0] = int(b[1][0] + dx)
        b[2][0] = int(b[2][0] + dx)
        b[3][0] = int(b[3][0] - dx)
        b[0][1] = int(b[0][1] -dy)
        b[1][1] = int(b[1][1] -dy)
        b[2][1] = int(b[2][1] +dy)
        b[3][1] = int(b[3][1] +dy)
    return box_list

class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class HardSigmoid(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'paddle':
            x = torch.add((1.2 * x), (3.))
            x = torch.clamp(x, 0., 6.)
            x = torch.div(x, 6.) # _
            # x = (1.2 * x).add(3.).clamp(0., 6.).div(6.) 
        else:
            x = F.relu6(x + 3, inplace=True) / 6
            F.hardsigmoid()
        return x

class HSigmoid(nn.Module):
    def forward(self, x):
        # x = (1.2 * x).add(3.).clamp(0., 6.).div(6.) 
        x = torch.add((1.2 * x), (3.))
        x = torch.clamp(x, 0., 6.)
        x = torch.div(x, 6.) # _
        return x
    
class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        num_mid_filter = in_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=in_channels, bias=True)
        # self.relu2 = HardSigmoid(hsigmoid_type)
        self.relu2 = HardSigmoid(type = 'paddle')

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn
    
class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                               padding=0, act=act)

        self.conv1 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                               stride=stride,
                               padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter)
        else:
            self.se = None

        self.conv2 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                               padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)
        if not self.not_add:
            y = x + y
        return y
    
class MobileNetV3(nn.Module):
    def __init__(self, in_channels, pretrained=False, **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super().__init__()
        self.scale = kwargs.get('scale', 0.5)
        model_name = kwargs.get('model_name', 'large')
        self.disable_se = kwargs.get('disable_se', True)
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, self.scale)

        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        self.stages = nn.ModuleList()
        block_list = []
        self.out_channels = []
        for layer_cfg in cfg:
            se = layer_cfg[3] and not self.disable_se
            if layer_cfg[5] == 2 and i > 2:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=se)
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1
        block_list.append(ConvBNACT(
            in_channels=inplanes,
            out_channels=self.make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act='hard_swish'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(self.make_divisible(scale * cls_ch_squeeze))

        if pretrained:
            ckpt_path = f'./weights/MobileNetV3_{model_name}_x{str(scale).replace(".", "_")}.pth'
            logger = logging.getLogger('torchocr')
            if os.path.exists(ckpt_path):
                logger.info('load imagenet weights')
                dic_ckpt = torch.load(ckpt_path)
                filtered_dict = OrderedDict()
                for key in dic_ckpt.keys():
                    flag = key.find('se') != -1
                    if self.disable_se and flag:
                        continue
                    filtered_dict[key] = dic_ckpt[key]

                self.load_state_dict(filtered_dict)
            else:
                logger.info(f'{ckpt_path} not exists')

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x = self.conv1(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

class DB_fpn(nn.Module):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.out_channels = out_channels
        # reduce layers
        self.in2_conv = nn.Conv2d(in_channels[0], self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels[1], self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels[2], self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels[3], self.out_channels, kernel_size=1, bias=False)
        # Smooth layers
        self.p5_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)

        self.in2_conv.apply(weights_init)
        self.in3_conv.apply(weights_init)
        self.in4_conv.apply(weights_init)
        self.in5_conv.apply(weights_init)
        self.p5_conv.apply(weights_init)
        self.p4_conv.apply(weights_init)
        self.p3_conv.apply(weights_init)
        self.p2_conv.apply(weights_init)

    def _interpolate_add(self, x, y):
        return F.interpolate(x, scale_factor=2) + y

    def _interpolate_cat(self, p2, p3, p4, p5):
        p3 = F.interpolate(p3, scale_factor=2)
        p4 = F.interpolate(p4, scale_factor=4)
        p5 = F.interpolate(p5, scale_factor=8)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = self._interpolate_add(in5, in4)
        out3 = self._interpolate_add(out4, in3)
        out2 = self._interpolate_add(out3, in2)

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        x = self._interpolate_cat(p2, p3, p4, p5)
        return x

class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1,
                               bias=False)
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=5, padding=2,
        #                        bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=2,
                                        stride=2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x
    
class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)
        self.binarize.apply(weights_init)
        self.thresh.apply(weights_init)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return y
    
backbone_dict = {'MobileNetV3': MobileNetV3}
neck_dict = {'DB_fpn': DB_fpn}
head_dict = {'DBHead': DBHead}

class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert 'in_channels' in config, 'in_channels must in model config'
        backbone_type = config.backbone.pop('type')
        assert backbone_type in backbone_dict, f'backbone.type must in {backbone_dict}'
        self.backbone = backbone_dict[backbone_type](config.in_channels, **config.backbone)

        neck_type = config.neck.pop('type')
        assert neck_type in neck_dict, f'neck.type must in {neck_dict}'
        self.neck = neck_dict[neck_type](self.backbone.out_channels, **config.neck)

        head_type = config.head.pop('type')
        assert head_type in head_dict, f'head.type must in {head_dict}'
        self.head = head_dict[head_type](self.neck.out_channels, **config.head)

        self.name = f'DetModel_{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

support_model = ['RecModel', 'DetModel']

def build_model(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    arch_model = eval(arch_type)(Dict(copy_config))
    return arch_model
