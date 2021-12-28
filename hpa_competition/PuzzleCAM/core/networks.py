# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from torchvision import models
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model

from .deeplab_utils import ASPP, Decoder
from .aff_utils import PathIndex
from .puzzle_utils import tile_features, merge_features

from hpa_competition.PuzzleCAM.tools.ai.torch_utils import resize_for_tensors

#######################################################################
# Normalization
#######################################################################
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class EfficModelWrapper(nn.Module):
    def __init__(self, eff_model):
        super().__init__()

        self.model = eff_model

    def forward(self, x):
        result = self.model(x)
        result = result[0]
        return result

# class SegmEfficModelWrapper(nn.Module):
#     def __init__(self, eff_model):
#         super().__init__()
#
#         self.model = eff_model
#
#     def forward(self, x):
#         mask, features = self.model(x)
#         # result = result[0]
#         return mask,

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

def change_first_conv(model, in_channels=3):
    if in_channels == 3:
        return model
    else:
        name, weights = model.named_parameters().__next__()
        layer_names = name.split('.')[:-1]
        layer = model
        for l in layer_names:
            layer = getattr(layer, l)
            layer = getattr(layer, l)

        out_channels = layer.out_channels
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        bias = layer.bias
        groups = layer.groups
        dilation = layer.dilation

        trained_kernel = layer.weight
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
            dilation=dilation
        )
        with torch.no_grad():
            new_conv.weight[:, :] = torch.stack(
                [torch.mean(trained_kernel, 1)] * in_channels, dim=1
            )
        cmd = ".".join([x if not x.isdigit() else f'[{x}]' for x in layer_names]).replace('.[', '[')
        exec(f'model.{cmd} = new_conv')
        return model


# if config['model']['model_name'].startswith('eff'):
#         return

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_config, segmentation=False):

        self.model_config = model_config
        self.segmentation = self.model_config['segmentation']
        self.model_name = self.model_config['arch']
        self.num_classes = self.model_config['classes']
        self.segm_classes = self.model_config['segm_classes']
        self.in_channels = len(model_config['in_channels']) + model_config['cell_input'] + model_config['nuclei_input']
        super().__init__()
        self.effnet = False
        self.mode = model_config['mode']

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in self.model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[self.model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(resnet.urls_dic[self.model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)

        elif 'efficient' in self.model_name:
            depth = 3
            dds = (256, 128, 64, 32, 16)[:3]
            if self.model_config['stride2dilation']:
                # depth -= 1
                dds = dds[:depth-1]
            print(dds)
            self.model = smp.Unet(
                encoder_name=self.model_name,
                encoder_weights=None,
                in_channels=self.in_channels,
                classes=self.segm_classes,
                encoder_depth=depth,
                decoder_channels=dds,
                dilated_encoder=self.model_config['stride2dilation'],
                # activation='sigmoid'
                )
            # print(self.model)
            if self.model_config['stride2dilation']:
                self.model.encoder.blocks[2][0].conv_dw.dilation = (2, 2)
                self.model.encoder.blocks[2][0].conv_dw.stride = (1, 1)
                self.model.encoder.blocks[2][0].conv_dw.padding = (4, 4)
                # self.model.encoder.blocks[-5][0].conv_dw.kernel_size = (5, 5)


            # else:
            #     eff_model = timm.create_model(
            #         self.model_name,
            #         num_classes=self.num_classes,
            #         in_chans=self.in_channels ,
            #         pretrained=self.model_config['pretrained'],
            #         features_only=True,
            #         out_indices=(2,))
            #
            #     self.model = EfficModelWrapper(eff_model)






            self.effnet = True
            return None
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + self.model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class Classifier(Backbone):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_config = model_config
        self.segmentation = self.model_config['segmentation']
        self.model_name = self.model_config['arch']
        self.num_classes = self.model_config['classes']
        self.classifier_param = self.model_config['classifier_param']

        if self.effnet:
            # if not self.segmentation:
            #     f_size = self.model.model.feature_info.channels()[-1]
            # else:
            #     # f_size = 32
            #     # f_size = 40
            f_size = 56
            assert f_size > self.num_classes

            self.classifier = nn.Sequential(nn.ReLU(), nn.Conv2d(f_size, self.num_classes, 1, bias=False))
        else:
            self.classifier = nn.Conv2d(2048, self.num_classes, 1, bias=False)

        self.initialize([self.classifier])

    def get_effnet_classifier(self, in_channels, str_param=None):

        if str_param == 'batchnorm':

            return nn.Sequential(nn.Conv2d(in_channels, self.num_classes, 1, bias=False), nn.BatchNorm2d(self.num_classes))

        if str_param == 'instanorm':

            return nn.Sequential(nn.Conv2d(in_channels, self.num_classes, 1, bias=False), nn.InstanceNorm2d(self.num_classes))
        else:
            return nn.Sequential(nn.ReLU(), nn.Conv2d(in_channels, self.num_classes, 1, bias=False))

    def forward(self, x, with_cam=False, no_decoder=False):
        mask = None
        if self.effnet:
            mask, features = self.model(x, no_decoder=not self.segmentation)
            x = features[3]
                # print(x.shape)

        else:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features, mask
        else:
            x = self.global_average_pooling_2d(x, keepdims=True)
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits
    def get_groups(self):
        return [self.model.encoder.parameters(), self.classifier.parameters(), self.model.decoder.parameters(), self.model.segmentation_head.parameters()]



class Classifier_For_Positive_Pooling(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes
        
        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            # logits = self.global_average_pooling_2d(features)
            logits = self.global_max_avg_pooling_2d(features)
            return logits, features
        else:
            x = self.global_max_avg_pooling_2d(x, keepdims=True)
            # x = self.global_average_pooling_2d(x, keepdims=True)

            logits = self.classifier(x).view(-1, self.num_classes)
            return logits

class Classifier_For_Puzzle(Classifier):
    def __init__(self, config, model_name, num_classes=20, mode='fix'):
        super().__init__(config)
        
    def forward(self, x, num_pieces=1, level=-1):
        batch_size = x.size()[0]
        
        output_dic = {}
        layers = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.classifier]

        for l, layer in enumerate(layers):
            l += 1
            if level == l:
                x = tile_features(x, num_pieces)

            x = layer(x)
            output_dic['stage%d'%l] = x
        
        output_dic['logits'] = self.global_average_pooling_2d(output_dic['stage6'])

        for l in range(len(layers)):
            l += 1
            if l >= level:
                output_dic['stage%d'%l] = merge_features(output_dic['stage%d'%l], num_pieces, batch_size)

        if level is not None:
            output_dic['merged_logits'] = self.global_average_pooling_2d(output_dic['stage6'])

        return output_dic
        
class AffinityNet(Backbone):

    def __init__(self, config, path_index=None):

        super().__init__(config)
        self.model_name = config['arch']
        self.num_classes = self.model_config['classes']
        self.classifier_param = self.model_config['classifier_param']


        if '50' in self.model_name:
            fc_edge1_features = 64
        else:
            fc_edge1_features = 128

        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

        if path_index is not None:
            self.path_index = path_index
            self.n_path_lengths = len(self.path_index.path_indices)
            for i, pi in enumerate(self.path_index.path_indices):
                self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x, with_affinity=False):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

        edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        if with_affinity:
            return edge, self.to_affinity(torch.sigmoid(edge))
        else:
            return edge

    def get_edge(self, x, image_size=512, stride=4):
        feat_size = (x.size(2)-1)//stride+1, (x.size(3)-1)//stride+1

        x = F.pad(x, [0, image_size-x.size(3), 0, image_size-x.size(2)])
        edge_out = self.forward(x)
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        
        return edge_out
    
    """
    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)
    """
    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)
        
        for i in range(self.n_path_lengths):
            ind = self._buffers["path_indices_" + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat

class DeepLabv3_Plus(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)
        
        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn
        
        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)
        
    def forward(self, x, with_cam=False):
        inputs = x

        x = self.stage1(x)
        x = self.stage2(x)
        x_low_level = x
        
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = self.aspp(x)
        x = self.decoder(x, x_low_level)
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

        return x

class Seg_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class CSeg_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, 'fix')

        if '50' in model_name:
            fc_edge1_features = 64
        else:
            fc_edge1_features = 128

        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, num_classes, 1, bias=True)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

        logits = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        # logits = resize_for_tensors(logits, x.size()[2:], align_corners=True)
        
        return logits

#coment
