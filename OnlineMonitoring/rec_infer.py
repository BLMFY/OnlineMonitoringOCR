# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import copy
import sys
import pathlib
import time
import cv2
# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))

import numpy as np

sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from ocr_utils import RecDataProcess
from ocr_utils import CTCLabelConverter
from PIL import Image, ImageDraw, ImageFont
from addict import Dict

class RecInfer:
    def __init__(self, model_path, dict_path, batch_size=16):
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

        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(dict_path)
        self.batch_size = batch_size

    def predict(self, imgs):

        
        # 预处理根据训练来
        if not isinstance(imgs,list):
            imgs = [imgs]
        imgs = [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths) # np.argsort按照元素的大小排序其序号
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            batch_idxs = idxs[idx:min(len(imgs), idx+self.batch_size)]
            batch_imgs = [self.process.width_pad_img(imgs[idx], 
                                                    imgs[batch_idxs[-1]].shape[1]) for idx in batch_idxs] # 每一批里最大的width
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0,3, 1, 2])).float()
            # tensor = torch.from_numpy(np.ones((1,3,32,120),dtype=np.float32)).float()
            tensor = tensor.to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                # print(out)
                out = out.softmax(dim=2)
            out = out.cpu().numpy() #
            # print(out.shape)
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        #按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts

def get_font_file():
    searchs = ["./simfang.ttf", "./demo/simfang.ttf", "/home/ying/OnlineMonitoring3.0/demo/simfang.ttf"]
    for path in searchs:
        if os.path.exists(path):
            return path
    assert False,"can't find .ttf"

def draw_ocr(image, txt = None):
    if isinstance(image,np.ndarray):
        image = Image.fromarray(image)
    # h, w = image.height, image.width
    img_up = image.copy()
    draw_up = ImageDraw.Draw(img_up)
    font_size = 25
    font = ImageFont.truetype(get_font_file(), font_size, encoding="utf-8")
    if txt is not None:
        for i in range(len(txt)):
            draw_up.text([3, 2+26*i], str(i+1)+'.'+txt[i], fill=(255,0,0), font=font)
    return np.array(img_up)

def number_ocr(image, rec_box, mode, edit_id = 0, al_num = 0):
    font_size = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    if mode == 0: # 红色框选区域
        if rec_box is not None:
            for i, box in enumerate(rec_box):
                if box[1] > font_size:
                    cv2.putText(image, str(i+1), (box[0], box[1]-3), font, 0.6, (255,0,0), 2, 8)
                else:
                    cv2.putText(image, str(i+1), (box[0], box[1]+font_size), font, 0.6, (255,0,0), 2, 8)
        return image
    elif mode == 1: # 蓝色待删除区域
        if rec_box is not None:
            if rec_box[1] > (font_size +1):
                cv2.putText(image, str(edit_id), (rec_box[0], rec_box[1]-3), font, 0.6, (0,0,255), 2, 8)
            else:
                cv2.putText(image, str(edit_id), (rec_box[0], rec_box[1]+font_size), font, 0.6, (0,0,255), 1, 8)
        return image
    elif mode == 2: # 绿色
        for i, box in enumerate(rec_box):
            if box[1] > font_size:
                cv2.putText(image, str(i+1+al_num), (box[0], box[1]-3), font, 0.6, (0,255,0), 2, 8)
            else:
                cv2.putText(image, str(i+1+al_num), (box[0], box[1]+font_size), font, 0.6, (0,255,0), 2, 8)
        return image
    elif mode == 3: # 棕色
        if rec_box[1] > font_size:
            cv2.putText(image, str(1+al_num), (rec_box[0], rec_box[1]-3), font, 0.6, (0,0,0), 2, 8)
        else:
            cv2.putText(image, str(1+al_num), (rec_box[0], rec_box[1]+font_size), font, 0.6, (0,0,0), 2, 8)
        return image
    
def Add_text(image, txt, xy, color=(0,0,255)):
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, txt, xy, font, 0.6, color, 2, 8)
    return image

def old_point_sort(area):
    area = np.array(area)
    area = area.reshape(-1,2)
    area = area.tolist()
    min_id = 0
    distance2 = area[0][0]**2 + area[0][1]**2
    for i, point in enumerate(area):
        if distance2 > (point[0]**2+point[1]**2):
            min_id = i
            distance2 = point[0]**2+point[1]**2
    new_sort = []
    for i in range(4):
        if min_id + i < 4:
            new_sort.append(area[i+min_id])
        else:
            new_sort.append(area[i+min_id-4])
    if (new_sort[1][0] - new_sort[0][0])and(new_sort[2][0] - new_sort[0][0]):
        k01 = (new_sort[1][1] - new_sort[0][1])/(new_sort[1][0] - new_sort[0][0])
        k02 = (new_sort[2][1] - new_sort[0][1])/(new_sort[2][0] - new_sort[0][0])
        if k01 > k02:
            c = new_sort[1]
            new_sort[1] = new_sort[3]
            new_sort[3] = c
    new_sort = np.array(new_sort)
    return new_sort

def point_sort(area):
    area = np.array(area)
    area = area.reshape(-1,2)
    area = area.tolist()
    area0 = copy.deepcopy(area)
    for i in range(4):
        area0[i].append(i)
    for j in range(4):
        sign = False
        for i in range(4-1-j):
            if area0[i][0] > area0[i+1][0]:
                area0[i], area0[i+1] = area0[i+1], area0[i]
                sign = True
        if not sign:
            break
    if area0[0][1] < area0[1][1]:
        min_id = area0[0][2]
    else:
        min_id = area0[1][2]
    new_sort = []
    for i in range(4):
        if min_id + i < 4:
            new_sort.append(area[i+min_id])
        else:
            new_sort.append(area[i+min_id-4])
    new_sort = np.array(new_sort)
    return new_sort

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    points = np.array(points)
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

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
        self.expand_conv = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                                     padding=0, act=act)

        self.bottleneck_conv = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter)
        else:
            self.se = None

        self.linear_conv = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                                     padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1

    def forward(self, x):
        y = self.expand_conv(x)
        y = self.bottleneck_conv(y)
        if self.se is not None:
            y = self.se(y)
        y = self.linear_conv(y)
        if not self.not_add:
            y = x + y
        return y
    
class MobileNetV3(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.5)
        model_name = kwargs.get('model_name', 'small')
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', (2, 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (2, 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (2, 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (1, 1)],
                [3, 72, 24, False, 'relu', (2, 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (2, 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', (2, 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, "supported scale are {} but input scale is {}".format(supported_scale,
                                                                                                    self.scale)

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
        inplanes = self.make_divisible(inplanes * scale)
        block_list = []
        for layer_cfg in cfg:
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=layer_cfg[3])
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])

        self.blocks = nn.Sequential(*block_list)
        self.conv2 = ConvBNACT(in_channels=inplanes,
                               out_channels=self.make_divisible(scale * cls_ch_squeeze),
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act='hard_swish')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = self.make_divisible(scale * cls_ch_squeeze)

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
    
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

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))
        return x

class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels,**kwargs):
        super(EncoderWithRNN, self).__init__()
        hidden_size = kwargs.get('hidden_size', 256)
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2,batch_first=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type='rnn',  **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'rnn': EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels,**kwargs)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)
        return x

class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        return self.fc(x)
    
backbone_dict = {'MobileNetV3': MobileNetV3}
neck_dict = {'PPaddleRNN': SequenceEncoder, 'None': Im2Seq}
head_dict = {'CTC': CTC}

class RecModel(nn.Module):
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

        self.name = f'RecModel_{backbone_type}_{neck_type}_{head_type}'

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