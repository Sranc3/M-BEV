# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from xml.etree.ElementPath import prepare_descendant
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import numpy as np
from mmcv.cnn import xavier_init, constant_init, kaiming_init
import math
from mmdet.models.utils import NormedLinear
import copy
import sys
from timm.models.vision_transformer import PatchEmbed, Block
from PIL import Image
import torchvision.transforms as transforms


def tensor2img(tensor, path='/data1/nuScenes/visualization'):
    tensor = tensor.squeeze()
    x_ = tensor.reshape(12,20,50,16,16,-1)
    x_ = torch.einsum('nhwpqc->nchpwq', x_)
    x_ = x_.reshape(12,3,320,800)
    
    n = tensor.shape[0]
    for i in range(n):
        x = x_[i]
        #tensor = tensor.cpu().detach().numpy()
        toPIL = transforms.ToPILImage()
        img = toPIL(x)
        img.save(path+ str(i)+'vis.jpg')


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    #print(pos_x.shape)  (900,128)
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    #print(pos_x.shape)   (900,128)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    #print(posemb.shape)   (900,384)
    return posemb

def calculate_sim(x1, x2, shift=50):
    C, H, W = x1.shape[-3:]
    w1 = W -shift
    x1 = x1.squeeze().permute(1,2,0)[:,w1:,...]
    x2 = x2.squeeze().permute(1,2,0)[:,:shift,...]
    x1_sum = (x1*x1).sum(dim=-1)
    x2_sum = (x2*x2).sum(dim=-1)
    sim_ma = (x1*x2).sum(dim=-1)/(torch.sqrt(x1_sum*x2_sum))
    print(sim_ma.mean())

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    (h, w) = grid_size
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class SELayer(nn.Module):                                     ########  feature guided position encoder: fpe
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class RegLayer(nn.Module):
    def __init__(self,  embed_dims=256, 
                        shared_reg_fcs=2, 
                        group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                        act_layer=nn.ReLU, 
                        drop=0.0):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            reg_branch.append(Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                Linear(embed_dims, embed_dims),
                act_layer(),
                Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        reg_feat = self.reg_branch(x)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs

@HEADS.register_module()
class PETRv2MAE(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                #  loss_cls=dict(
                #      type='CrossEntropyLoss',
                #      bg_cls_weight=0.1,
                #      use_sigmoid=False,
                #      loss_weight=1.0,
                #      class_weight=1.0),
                #  loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                #  loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_level = 0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                 init_cfg=None,
                 normedlinear=False,
                 with_fpe=False,
                 with_time=False,
                 with_multi=False,
                 mvr = True,      
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        #class_weight = loss_cls.get('class_weight', None)
        # if class_weight is not None and (self.__class__ is PETRv2MAE):
        #     assert isinstance(class_weight, float), 'Expected ' \
        #         'class_weight to have type float. Found ' \
        #         f'{type(class_weight)}.'
        #     # NOTE following the official DETR rep0, bg_cls_weight means
        #     # relative classification weight of the no-object class.
        #     bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
        #     assert isinstance(bg_cls_weight, float), 'Expected ' \
        #         'bg_cls_weight to have type float. Found ' \
        #         f'{type(bg_cls_weight)}.'
        #     class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            # class_weight[num_classes] = bg_cls_weight
            # loss_cls.update({'class_weight': class_weight})
            # if 'bg_cls_weight' in loss_cls:
            #     loss_cls.pop('bg_cls_weight')
            # self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            # assert 'assigner' in train_cfg, 'assigner should be provided '\
            #     'when train_cfg is set.'
            # assigner = train_cfg['assigner']
            # assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            #     'The classification weight for loss and matcher should be' \
            #     'exactly the same.'
            # assert loss_bbox['loss_weight'] == assigner['reg_cost'][
            #     'weight'], 'The regression L1 weight for loss and matcher ' \
            #     'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            #self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        #print('in chanels', in_channels)   256
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step                                  ############ ?
        #print('depth_step',depth_step)  0.8
        self.depth_num = depth_num                                    ##########  �?        
        #print('depth_num',depth_num)    64
        self.position_dim = 3 * self.depth_num    ### 192，对应的�?06行的192
        self.position_range = position_range
        self.LID = LID                                                 ######### �?        
        self.depth_start = depth_start  #1
        self.position_level = position_level    # 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.mvr = mvr                              ###################### 决定是否用mae
        self.multi_mvr = False
        # assert 'num_feats' in positional_encoding
        # num_feats = positional_encoding['num_feats']
        # assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
        #     f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
        #     f' and {num_feats}.'
        # self.act_cfg = transformer.get('act_cfg',
        #                                dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.with_fpe = with_fpe
        self.with_time = with_time
        self.with_multi = with_multi
        self.group_reg_dims = group_reg_dims
        super(PETRv2MAE, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.positional_encoding = build_positional_encoding(
                positional_encoding)
        #self.transformer = build_transformer(transformer)   ##### 大体框架
        #self.DETRTransformer = DETRTransformer(256,2048)              ################ 额外加的图像交互层        
        if self.mvr:
            self.mae_embed_dim = 256
            mae_pos_embed = torch.zeros(1, 1000, self.mae_embed_dim)
            mae_pos_embed = get_2d_sincos_pos_embed(mae_pos_embed.shape[-1], (20,50), cls_token=False)
            mae_pos_embed = torch.from_numpy(mae_pos_embed).float().unsqueeze(0)
            self.mae_pos_embed = nn.Parameter(mae_pos_embed, requires_grad=False)
            ####  decoder ####
            self.decoder_embed_dim = 256
            #self.decoder_embed = nn.Linear(256, 256, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.mae_embed_dim))
            self.decoder_blocks = nn.ModuleList([
                Block(256, 8, 4., qkv_bias=True , norm_layer=nn.LayerNorm)
                for i in range(6)])
            self.decoder_norm = nn.LayerNorm(256)
            self.decoder_pred = nn.Linear(256, 256, bias=True)
            #### init weight #####
            #self.attn = nn.MultiheadAttention(256, 8, 0.4)
            self.a = nn.Parameter(torch.ones(1))
            
        self._init_layers()

    def _init_layers(self):
        pass
        
    #     """Initialize layers of the transformer head."""
        # if self.with_position:
        #     self.input_proj = Conv2d(
        #         self.in_channels, self.embed_dims, kernel_size=1)
        # else:
        #     self.input_proj = Conv2d(
        #         self.in_channels, self.embed_dims, kernel_size=1)

    #     cls_branch = []
    #     for _ in range(self.num_reg_fcs):
    #         cls_branch.append(Linear(self.embed_dims, self.embed_dims))
    #         cls_branch.append(nn.LayerNorm(self.embed_dims))
    #         cls_branch.append(nn.ReLU(inplace=True))
    #     if self.normedlinear:
    #         cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
    #     else:
    #         cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
    #     fc_cls = nn.Sequential(*cls_branch)

    #     if self.with_multi:
    #         reg_branch = RegLayer(self.embed_dims, self.num_reg_fcs, self.group_reg_dims)
    #     else:
    #         reg_branch = []
    #         for _ in range(self.num_reg_fcs):
    #             reg_branch.append(Linear(self.embed_dims, self.embed_dims))
    #             reg_branch.append(nn.ReLU())
    #         reg_branch.append(Linear(self.embed_dims, self.code_size))
    #         reg_branch = nn.Sequential(*reg_branch)
        
    #     self.cls_branches = nn.ModuleList(
    #         [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
    #     self.reg_branches = nn.ModuleList(
    #         [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])

    #     if self.with_multiview:
    #         self.adapt_pos3d = nn.Sequential(
    #             nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
    #             nn.ReLU(),
    #             nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
    #         )
    #     else:
    #         self.adapt_pos3d = nn.Sequential(
    #             nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
    #             nn.ReLU(),
    #             nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
    #         )

    #     if self.with_position:
    #         self.position_encoder = nn.Sequential(
    #             nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
    #             nn.ReLU(),
    #             nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
    #         )

    # #     self.reference_points = nn.Embedding(self.num_query, 3)
    # #     self.query_embedding = nn.Sequential(
    # #         nn.Linear(self.embed_dims*3//2, self.embed_dims),
    # #         nn.ReLU(),
    # #         nn.Linear(self.embed_dims, self.embed_dims),
    # #     )
    # #     self.tem_embedding = nn.Parameter(torch.randn(2,256))  #### 自己瞎写�?
    #     if self.with_fpe:
    #         self.fpe = SELayer(self.embed_dims)

    def init_weights(self):
        pass
    #     """Initialize weights of the transformer head."""
    #     # The initialization for transformer is important
    #     self.transformer.init_weights()
    #     nn.init.uniform_(self.reference_points.weight.data, 0, 1)
    #     if self.loss_cls.use_sigmoid:
    #         bias_init = bias_init_with_prob(0.01)
    #         for m in self.cls_branches:
    #             nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]  ## 320 800
        
        B, N, C, H, W = img_feats[self.position_level].shape
        #print('B,N,C,H,W' ,B,N, C, H, W)  (1,12,256,20,50)
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            #print(index)    0�?3                  depth_num=64
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            #print(bin_size,self.depth_start)   0.014471153846153847  1
            coords_d = self.depth_start + bin_size * index * index_1
            #print(coords_d)
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index
        ### 到这里，构建camera的frustum space
        D = coords_d.shape[0]  ###  64
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        #print(coords.shape)  (50,20,64,3)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        #print(coords.shape)  (50,20,64,4)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)  ### ? 什么操�?        ##建立3维坐标系


        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):     #### lidar2img 是雷达到图像的转换矩�?                
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))  ## 矩阵求逆运�?img2lidar是图像到3维世界坐标的转化矩阵
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars) # (B, N, 4, 4)
        #print(img2lidars.shape)   (1,12,4,4)
        #print(img2lidars[0,0,1],img2lidars[0,6,1])  这个矩阵每时每刻都在变啊，不是固定的

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        #print(coords.shape)  (1,12,50,20,64,4,1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        #print(img2lidars.shape,'lidar') (1,12,50,20,64,4,4)  
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]  ### 矩阵转化，由frustum空间�?Dspace, 取到3是因为最后一个是常数�?        #print(coords3d.shape)  (1,12,50,20,64,3)
        coords3d[..., 0] = (coords3d[..., 0] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1] = (coords3d[..., 1] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2] = (coords3d[..., 2] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
        ### 归一化        #对齐了没有，要检查一下，到这里都是物理上的操作        

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)   ## | 按位或运�?大于1或小�?�?(超出边界范围，应该遮�?，否则为0�?        #print(coords_mask)  
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)    ### (12,192,20,50)

    

        coords3d = inverse_sigmoid(coords3d)  ### sigmoid的反函数
        coords_position_embeding = self.position_encoder(coords3d)  ### 线性变换，192变为256
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.
        # state_dict_keys = list(state_dict.keys())
        # my_state_dict = torch.load('/home/csr/PETR/work_dirs/petrv2_frame_mae/epoch_15.pth')['state_dict']
        # my_state_dict_keys = list(my_state_dict.keys())
        # for k in my_state_dict_keys:
        #     if k not in state_dict_keys:
        #         state_dict[k] = my_state_dict[k]
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRv2MAE:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.size(0), x.size(1)
        b, n ,c, h, w = x.shape
        x_ori = x.clone().detach()
        x[:,0] = 0
        x[:,6] = 0
        #print(batch_size, num_cams)   1 12 本来6个camera,但是加入了时序信息，拼接在一起，变成12个camera
        #print(x.shape)   (1,12,256,20,50) 

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0] 
        masks = x.new_ones(
                 (batch_size, num_cams, input_img_h, input_img_w)) 
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]   ### 320 800
                masks[img_id, cam_id, :img_h, :img_w] = 0
        # x = self.input_proj(x.flatten(0,1)) 
        # x = x.view(batch_size, num_cams, *x.shape[-3:])

        masks = F.interpolate(
                 masks, size=x.shape[-2:]).to(torch.bool)

        # if self.with_position:
        #     coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
        #     #print(coords_position_embeding.shape)
        #     if self.with_fpe:
        #         coords_position_embeding = self.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())

        #     pos_embed = coords_position_embeding
        #     #print(coords_position_embeding.shape)  (1,12,256,20,50)和x的形状一样
        #     if self.with_multiview:
        #         sin_embed = self.positional_encoding(masks)
        #         #print(sin_embed.shape)  (1,12,384,20,50)  
        #         sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        #         pos_embed = pos_embed + sin_embed
        #     else:
        #         pos_embeds = []
        #         for i in range(num_cams):
        #             xy_embed = self.positional_encoding(masks[:, i, :, :])
        #             pos_embeds.append(xy_embed.unsqueeze(1))
        #         sin_embed = torch.cat(pos_embeds, 1)
        #         sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        #         pos_embed = pos_embed + sin_embed
        # else:
        #     if self.with_multiview:
        #         pos_embed = self.positional_encoding(masks)
        #         pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
        #     else:
        #         pos_embeds = []
        #         for i in range(num_cams):
        #             pos_embed = self.positional_encoding(masks[:, i, :, :])
        #             pos_embeds.append(pos_embed.unsqueeze(1))
        #         pos_embed = torch.cat(pos_embeds, 1)
        #x = x + pos_embed
         
#########################################Transformer Encoder#######################################################
        
        # mask_tokens = self.mask_token.repeat(20,50,1).unsqueeze(0).permute(0,3,1,2)
        # x[:,0] += mask_tokens
        
        # x_info = torch.cat([x[:,1],x[:,2],x[:,7],x[:,8]], dim =1)
        # # x_info = x_info.reshape(4, h*w, c)
        # # x_info += self.mae_pos_embed
        # x_info = x_info.reshape(1, 4*h*w, c)
        # query = pos_embed[:,6].reshape(1,h*w,c)
        # key = torch.cat([pos_embed[:,1],pos_embed[:,2],pos_embed[:,7],pos_embed[:,8]], dim =1).reshape(1,4*h*w, c)
        # x_input,_ = self.attn(query = query,key=key,value=x_info )
        # #print(x_input.shape)  (1,1000,256)
        # x_input = x_input + self.mae_pos_embed
#######################################################################################################################
        patches = x.permute(1,3,4,0,2).reshape(n,h*w,c)
        left_index = torch.cat([torch.arange(0,10)+_*50 for _ in range(20)]).type(torch.long)
        right_index = torch.cat([torch.arange(40,50)+_*50 for _ in range(20)]).type(torch.long)
        mid_index = torch.cat([torch.arange(10,40)+ _*50 for _ in range(20)]).type(torch.long)
        x_ref_token = torch.zeros((2,1000,256),device=x.device)
        a = self.a
        x_ref_token[:,left_index] = (patches[1][right_index] + a * patches[7][right_index]) / (1+a)
        x_ref_token[:,right_index] = (patches[2][left_index] + a * patches[8][left_index]) /(1+a)
        #print(x_ref_token[:,mid_index].shape)
        x_ref_token[:,mid_index] += self.mask_token

        
        x_input = x_ref_token + self.mae_pos_embed
        # pos_embed = pos_embed.reshape(n, h*w, c)
        # x_input = x_input + torch.stack((pos_embed[0],pos_embed[6]))
        x_r = x_input
########################################################################################################################
        # num_patches = h * w
        # ratio = 0.6
        # masked_num = int(num_patches * ratio)
        # patches = x.permute(1,3,4,0,2).reshape(n, h*w, c)
        # #shuffle_indices = torch.rand(n, num_patches, device=x.device).argsort()   ## 随机mask
        # shuffle_indices = torch.zeros((12,1000), device=x.device)
        # shuffle_indices[:,:masked_num] = torch.cat([torch.range(10,39)+ _*50 for _ in range(20)]).type(torch.long)
        # shuffle_indices[:,masked_num:] = torch.cat((torch.cat([torch.range(0,9)+_*50 for _ in range(20)]), torch.cat([torch.range(40,49)+_*50 for _ in range(20)]))).type(torch.long)
        
        # mask_ind, unmask_ind = shuffle_indices[:, :masked_num].type(torch.long), shuffle_indices[:, masked_num:].type(torch.long)
        # batch_ind = torch.arange(n, device=x.device).unsqueeze(-1)
        # mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
        

        # mask = torch.zeros(n, h*w, device=x.device)
        # mask[batch_ind, unmask_ind] = 1
        # mask_tokens = self.mask_token.repeat(n, masked_num, 1)
        # enc_to_dec_tokens = unmask_patches
        # #enc_to_dec_tokens = self.decoder_embed(unmask_patches)
        # concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # dec_input_tokens = torch.empty_like(concat_tokens, device=concat_tokens.device)
        # dec_input_tokens[batch_ind.type(torch.long), shuffle_indices.type(torch.long)] = concat_tokens
        # #pos_embed = pos_embed.reshape(n, h*w, c)
        # dec_input_tokens += self.mae_pos_embed
        # #dec_input_tokens += pos_embed    ### 3d
        # x_r = dec_input_tokens
#######################################################################################################################

        
        for blk in self.decoder_blocks:
            x_r = blk(x_r)
        #x = x.reshape(12, h*w, c)   #### 改
        x_r = self.decoder_norm(x_r)
        x_r = self.decoder_pred(x_r)
        x_r = x_r.permute(0,2,1).reshape(2,c,h,w)  #### 改
        
     
######################################  front ###############################################
        unmask_ind_1 = [1,2,3,4,5]
        unmask_ind_2 = [7,8,9,10,11]
        x = torch.cat((x_r[0].unsqueeze(0), x[:,unmask_ind_1,...].squeeze(), x_r[1].unsqueeze(0), x[:,unmask_ind_2,...].squeeze()), dim=0).unsqueeze(0)
#####################################   back #################################################
        # unmask_ind_1 = [0,1,2,]
        # unmask_ind_2 = [4,5,6,7,8]
        # unmask_ind_3 = [10,11]
        # x = torch.cat((x[:,unmask_ind_1,...].squeeze(),x_r[0].unsqueeze(0), x[:,unmask_ind_2,...].squeeze(), x_r[1].unsqueeze(0), 
        # x[:,unmask_ind_3,...].squeeze()), dim=0).unsqueeze(0)
        
################################## MAE ##########################################################
        if self.multi_mae:
            #encoder = self.maeconder
            t = 2
            x = x.reshape(2,6,c,h,w).permute(0,2,3,4,1).reshape(t,c,h,-1)
            pos_embed = pos_embed.reshape(2,6,c,h,w).permute(0,2,3,4,1).reshape(t,c,h,-1)
            w = w * 6
            num_patches = h * w
            

            ratio = 0.5
            masked_num = int(num_patches * ratio)
            patches = x.reshape(t, h*w, c)
            shuffle_indices = torch.rand(t, num_patches, device=x.device).argsort()
            mask_ind, unmask_ind = shuffle_indices[:, :masked_num], shuffle_indices[:, masked_num:]
            batch_ind = torch.arange(t, device=x.device).unsqueeze(-1)
            mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
            #mask = torch.ones(b*n, h*w, device=x.device)
            

            mask = torch.zeros(t, h*w, device=x.device)
            mask[batch_ind, unmask_ind] = 1
            mask_tokens = self.mask_token.repeat(t, masked_num, 1)
            enc_to_dec_tokens = self.decoder_embed(unmask_patches)
            concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
            dec_input_tokens = torch.empty_like(concat_tokens, device=concat_tokens.device)
            dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
            pos_embed = pos_embed.reshape(t, h*w, c)
            dec_input_tokens += pos_embed
            x = dec_input_tokens
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            x = x.reshape(12,1000,-1)
            mask = mask.reshape(12,-1)
            #print(x.shape)
 ################################################################################################################         
     
        mask = [0,6]     ########### 改
        return x, mask, x_ori
##########################################可视化##########################################################

          
###################################################################################################
        # else:
        #     x0 = x[:,0,...]
        #     x1 = x[:,1,...]
        #     x2 = x[:,2,...]
        #     #calculate_sim(x2,x1,5)
        #     #calculate_sim(x[:,0,...],x[:,1,...])
        #     input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]   ####(320,800,3) 输入图像尺寸
        #     #print(img_metas[0]['filename'])
        #     #print(img_metas)  图像的各种信息，包括12张图像的路径，ori_shape(900,1600,3,6), img_shape(320,800,3)
        #     #lidar2img (4×4的array�?2�?,看起来像是旋转矩�?  intrinsics (array: 4×4, 12个， 看起来像是旋转矩阵，其中不少�?)
        #     # extrinsics (array: 4×4, 12�?  pad_shape(320,800,3)  scale_factor:1  box_mode_3d  ........

        #     masks = x.new_ones(
        #         (batch_size, num_cams, input_img_h, input_img_w))   ### new_ones 保持和x相同的数据类型和device
        #     #print(masks.shape)  (1,12,320,800)
        #     for img_id in range(batch_size):
        #         for cam_id in range(num_cams):
        #             img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]   ### 320 800
        #             masks[img_id, cam_id, :img_h, :img_w] = 0
                
        #     x = self.input_proj(x.flatten(0,1))    #### 只展开�?和第一个维度，也就�?12,256,20,50)
        #     x = x.view(batch_size, num_cams, *x.shape[-3:])  ### *无需保证值和变量数目相同
        #     #print(x.shape)  (1,12,256,20,50)

        #     # interpolate masks to have the same spatial shape with x
        #     masks = F.interpolate(
        #         masks, size=x.shape[-2:]).to(torch.bool)   ### 数组采样 (1,12,20,50)  全部为False 不对前两个维度处理（上下采样�?            

        #     if self.with_position:
        #         coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
        #         #print(coords_position_embeding.shape)
        #         if self.with_fpe:
        #             coords_position_embeding = self.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())

        #         pos_embed = coords_position_embeding
        #         #print(coords_position_embeding.shape)  (1,12,256,20,50)和x的形状一�?
        #         if self.with_multiview:
        #             sin_embed = self.positional_encoding(masks)
        #             #print(sin_embed.shape)  (1,12,384,20,50)  
        #             sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        #             pos_embed = pos_embed + sin_embed
        #         else:
        #             pos_embeds = []
        #             for i in range(num_cams):
        #                 xy_embed = self.positional_encoding(masks[:, i, :, :])
        #                 pos_embeds.append(xy_embed.unsqueeze(1))
        #             sin_embed = torch.cat(pos_embeds, 1)
        #             sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        #             pos_embed = pos_embed + sin_embed
        #     else:
        #         if self.with_multiview:
        #             pos_embed = self.positional_encoding(masks)
        #             pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
        #         else:
        #             pos_embeds = []
        #             for i in range(num_cams):
        #                 pos_embed = self.positional_encoding(masks[:, i, :, :])
        #                 pos_embeds.append(pos_embed.unsqueeze(1))
        #             pos_embed = torch.cat(pos_embeds, 1)

        #     # x_ = x + pos_embed
        #     # x_ = x_.reshape(b*n, c, h*w)
        #     # x_ = x_.permute(0,2,1)
        #     # x_ =  self.DETRTransformer(x_)
        #     # pos_embed = x_.permute(0,2,1).reshape(b,n,c,h,w)  ### 仅仅用这个当作key

        #     reference_points = self.reference_points.weight
        #     #print(reference_points.shape)  (900,3)
        #     query_embeds = self.query_embedding(pos2posemb3d(reference_points))  ### (900,384)
        #     reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1) #.sigmoid()
        #     #print(reference_points.shape)  (1,900,3)
        #     value_tem = torch.ones_like(x)
        #     value_tem[:,:6,...] = self.tem_embedding[0].repeat(1,6,20,50,1).permute(0,1,4,2,3)
        #     value_tem[:,6:,...] = self.tem_embedding[1].repeat(1,6,20,50,1).permute(0,1,4,2,3)
        #     #print(value_tem)

        #     outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, value_tem ,self.reg_branches)
        #     ##     x:作为value，从2D图像线性映射得�?(1,12,256,20,50)
        #     ##     maskS: (1,12,20,50) 全部为False  用于2维正弦编码？
        #     ##     query_embeds: (900,384) 经过线性变换得到的900个query
        #     ##     pos_embed: (1,12,256,20,50)  带有3D位置编码信息的，�?D图像特征大小相同的embedding, 代码中还加入了正弦编码？


        #     outs_dec = torch.nan_to_num(outs_dec)
        #     #print(outs_dec.shape)  (6,1,900,256)  这里�?不是6个相机，而是中间层的
            
        #     ### 计算时间，不重要
        #     if self.with_time:
        #         time_stamps = []
        #         for img_meta in img_metas:    
        #             time_stamps.append(np.asarray(img_meta['timestamp']))
        #         time_stamp = x.new_tensor(time_stamps)
        #         time_stamp = time_stamp.view(batch_size, -1, 6)
        #         mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
            
        #     outputs_classes = []
        #     outputs_coords = []
        #     for lvl in range(outs_dec.shape[0]):
        #         reference = inverse_sigmoid(reference_points.clone())
        #         #print('reference',reference.shape)  (1,900,3) 形状没变，进行了一个inverse_sigmoid,不知道干啥的
        #         assert reference.shape[-1] == 3
        #         outputs_class = self.cls_branches[lvl](outs_dec[lvl])  ### 分类�? classfication
        #         # print('outputs_class',outputs_class.shape)   (1,900,10),对每个框预测10个种类的概率
        #         tmp = self.reg_branches[lvl](outs_dec[lvl])    #### 回归�?regression
        #         #print(tmp.shape)    (1,900,10)
        #         tmp[..., 0:2] += reference[..., 0:2]
        #         #print(reference[..., 0:2].shape)  �?0个中的前两个加上ref的前两个维度
        #         tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        #         tmp[..., 4:5] += reference[..., 2:3]
        #         tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

        #         if self.with_time:
        #             tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

        #         outputs_coord = tmp
        #         outputs_classes.append(outputs_class)
        #         outputs_coords.append(outputs_coord)

        #     all_cls_scores = torch.stack(outputs_classes)
        #     #print(all_cls_scores.shape)  (6,1,900,10)
        #     all_bbox_preds = torch.stack(outputs_coords)
        #     #print(all_bbox_preds.shape)  #(6,1,900,10)

        #     all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]) # x
        #     all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]) # y
        #     all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]) # z 真实坐标
        #     #print(all_bbox_preds[0,0,1])
        #     outs = {
        #         'all_cls_scores': all_cls_scores,
        #         'all_bbox_preds': all_bbox_preds,
        #         'enc_cls_scores': None,
        #         'enc_bbox_preds': None, 
        #     }
        #     return outs

    def get_targets(self,):
        return None

    def get_bboxes(self,):
        return None
    
    def loss(self, x, patches, mask):

        
        # mean = patches.mean(dim=-1, keepdim=True)
        # var = patches.var(dim=-1, keepdim=True)
        # patches = (patches - mean) / (var + 1.e-6)**.5
        #print(x.shape)
        #patches = patches[:,mask]      ######## 改
        # tensor2img(x,path='/data1/nuScenes/visualization/pre')
        # tensor2img(patches,path='/data1/nuScenes/visualization/ground')
        loss = (x - patches) ** 2  # [N, L], mean loss per patch
        #print(loss.shape)
        #print(loss[:,1])
        loss = (x - patches) ** 2
        #print(loss.shape)  (2,1000)
        loss = loss.sum()/2000   ############## 改
        #loss = (loss * mask).sum() / mask.sum()
        loss_dict = {'loss_mae':loss
            }
        #print(loss)
        return loss_dict

    # def _get_target_single(self,
    #                        cls_score,
    #                        bbox_pred,
    #                        gt_labels,
    #                        gt_bboxes,
    #                        gt_bboxes_ignore=None):
    #     """"Compute regression and classification targets for one image.
    #     Outputs from a single decoder layer of a single feature level are used.
    #     Args:
    #         cls_score (Tensor): Box score logits from a single decoder layer
    #             for one image. Shape [num_query, cls_out_channels].
    #         bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
    #             for one image, with normalized coordinate (cx, cy, w, h) and
    #             shape [num_query, 4].
    #         gt_bboxes (Tensor): Ground truth bboxes for one image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (Tensor): Ground truth class indices for one image
    #             with shape (num_gts, ).
    #         gt_bboxes_ignore (Tensor, optional): Bounding boxes
    #             which can be ignored. Default None.
    #     Returns:
    #         tuple[Tensor]: a tuple containing the following for one image.
    #             - labels (Tensor): Labels of each image.
    #             - label_weights (Tensor]): Label weights of each image.
    #             - bbox_targets (Tensor): BBox targets of each image.
    #             - bbox_weights (Tensor): BBox weights of each image.
    #             - pos_inds (Tensor): Sampled positive indices for each image.
    #             - neg_inds (Tensor): Sampled negative indices for each image.
    #     """

    #     num_bboxes = bbox_pred.size(0)
    #     # assigner and sampler
    #     assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
    #                                          gt_labels, gt_bboxes_ignore)
    #     sampling_result = self.sampler.sample(assign_result, bbox_pred,
    #                                           gt_bboxes)
    #     pos_inds = sampling_result.pos_inds
    #     neg_inds = sampling_result.neg_inds

    #     # label targets
    #     labels = gt_bboxes.new_full((num_bboxes, ),
    #                                 self.num_classes,
    #                                 dtype=torch.long)
    #     labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
    #     label_weights = gt_bboxes.new_ones(num_bboxes)

    #     # bbox targets
    #     code_size = gt_bboxes.size(1)
    #     bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
    #     bbox_weights = torch.zeros_like(bbox_pred)
    #     bbox_weights[pos_inds] = 1.0
    #     #print('gt_bboxes size, bbox_pred size',gt_bboxes.size(), bbox_pred.size()) ## (n, 9) (900,10)

    #     # DETR
    #     if sampling_result.pos_gt_bboxes.shape[1] == 4:
    #         bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes.reshape(sampling_result.pos_gt_bboxes.shape[0], self.code_size - 1)
    #     else:
    #         bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

    #     return (labels, label_weights, bbox_targets, bbox_weights, 
    #             pos_inds, neg_inds)

    # def get_targets(self,
    #                 cls_scores_list,
    #                 bbox_preds_list,
    #                 gt_bboxes_list,
    #                 gt_labels_list,
    #                 gt_bboxes_ignore_list=None):
    #     """"Compute regression and classification targets for a batch image.
    #     Outputs from a single decoder layer of a single feature level are used.
    #     Args:
    #         cls_scores_list (list[Tensor]): Box score logits from a single
    #             decoder layer for each image with shape [num_query,
    #             cls_out_channels].
    #         bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
    #             decoder layer for each image, with normalized coordinate
    #             (cx, cy, w, h) and shape [num_query, 4].
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         gt_bboxes_ignore_list (list[Tensor], optional): Bounding
    #             boxes which can be ignored for each image. Default None.
    #     Returns:
    #         tuple: a tuple containing the following targets.
    #             - labels_list (list[Tensor]): Labels for all images.
    #             - label_weights_list (list[Tensor]): Label weights for all \
    #                 images.
    #             - bbox_targets_list (list[Tensor]): BBox targets for all \
    #                 images.
    #             - bbox_weights_list (list[Tensor]): BBox weights for all \
    #                 images.
    #             - num_total_pos (int): Number of positive samples in all \
    #                 images.
    #             - num_total_neg (int): Number of negative samples in all \
    #                 images.
    #     """
    #     assert gt_bboxes_ignore_list is None, \
    #         'Only supports for gt_bboxes_ignore setting to None.'
    #     num_imgs = len(cls_scores_list)
    #     gt_bboxes_ignore_list = [
    #         gt_bboxes_ignore_list for _ in range(num_imgs)
    #     ]

    #     (labels_list, label_weights_list, bbox_targets_list,
    #      bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
    #          self._get_target_single, cls_scores_list, bbox_preds_list,
    #          gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
    #     num_total_pos = sum((inds.numel() for inds in pos_inds_list))
    #     num_total_neg = sum((inds.numel() for inds in neg_inds_list))
    #     return (labels_list, label_weights_list, bbox_targets_list,
    #             bbox_weights_list, num_total_pos, num_total_neg)

    # def loss_single(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 gt_bboxes_list,
    #                 gt_labels_list,
    #                 gt_bboxes_ignore_list=None):
    #     """"Loss function for outputs from a single decoder layer of a single
    #     feature level.
    #     Args:
    #         cls_scores (Tensor): Box score logits from a single decoder layer
    #             for all images. Shape [bs, num_query, cls_out_channels].
    #         bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
    #             for all images, with normalized coordinate (cx, cy, w, h) and
    #             shape [bs, num_query, 4].
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         gt_bboxes_ignore_list (list[Tensor], optional): Bounding
    #             boxes which can be ignored for each image. Default None.
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components for outputs from
    #             a single decoder layer.
    #     """
    #     num_imgs = cls_scores.size(0)
    #     cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    #     bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    #     cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
    #                                        gt_bboxes_list, gt_labels_list, 
    #                                        gt_bboxes_ignore_list)
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     labels = torch.cat(labels_list, 0)
    #     label_weights = torch.cat(label_weights_list, 0)
    #     bbox_targets = torch.cat(bbox_targets_list, 0)
    #     bbox_weights = torch.cat(bbox_weights_list, 0)

    #     # classification loss
    #     cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    #     # construct weighted avg_factor to match with the official DETR repo
    #     cls_avg_factor = num_total_pos * 1.0 + \
    #         num_total_neg * self.bg_cls_weight
    #     if self.sync_cls_avg_factor:
    #         cls_avg_factor = reduce_mean(
    #             cls_scores.new_tensor([cls_avg_factor]))

    #     cls_avg_factor = max(cls_avg_factor, 1)
    #     loss_cls = self.loss_cls(
    #         cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    #     # Compute the average number of gt boxes accross all gpus, for
    #     # normalization purposes
    #     num_total_pos = loss_cls.new_tensor([num_total_pos])
    #     num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    #     # regression L1 loss
    #     bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
    #     normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
    #     isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)  ### 
    #     bbox_weights = bbox_weights * self.code_weights

    #     loss_bbox = self.loss_bbox(
    #             bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

    #     loss_cls = torch.nan_to_num(loss_cls)
    #     loss_bbox = torch.nan_to_num(loss_bbox)
    #     return loss_cls, loss_bbox
    
    # @force_fp32(apply_to=('preds_dicts'))
    # def loss(self,
    #          gt_bboxes_list,
    #          gt_labels_list,
    #          preds_dicts,
    #          gt_bboxes_ignore=None):
    #     """"Loss function.
    #     Args:
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         preds_dicts:
    #             all_cls_scores (Tensor): Classification score of all
    #                 decoder layers, has shape
    #                 [nb_dec, bs, num_query, cls_out_channels].
    #             all_bbox_preds (Tensor): Sigmoid regression
    #                 outputs of all decode layers. Each is a 4D-tensor with
    #                 normalized coordinate format (cx, cy, w, h) and shape
    #                 [nb_dec, bs, num_query, 4].
    #             enc_cls_scores (Tensor): Classification scores of
    #                 points on encode feature map , has shape
    #                 (N, h*w, num_classes). Only be passed when as_two_stage is
    #                 True, otherwise is None.
    #             enc_bbox_preds (Tensor): Regression results of each points
    #                 on the encode feature map, has shape (N, h*w, 4). Only be
    #                 passed when as_two_stage is True, otherwise is None.
    #         gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
    #             which can be ignored for each image. Default None.
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     assert gt_bboxes_ignore is None, \
    #         f'{self.__class__.__name__} only supports ' \
    #         f'for gt_bboxes_ignore setting to None.'

    #     all_cls_scores = preds_dicts['all_cls_scores']
    #     all_bbox_preds = preds_dicts['all_bbox_preds']
    #     enc_cls_scores = preds_dicts['enc_cls_scores']
    #     enc_bbox_preds = preds_dicts['enc_bbox_preds']
    #     # print(gt_labels_list)
    #     num_dec_layers = len(all_cls_scores)
    #     device = gt_labels_list[0].device
    #     gt_bboxes_list = [torch.cat(
    #         (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
    #         dim=1).to(device) for gt_bboxes in gt_bboxes_list]

    #     all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
    #     all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
    #     all_gt_bboxes_ignore_list = [
    #         gt_bboxes_ignore for _ in range(num_dec_layers)
    #     ]

    #     losses_cls, losses_bbox = multi_apply(
    #         self.loss_single, all_cls_scores, all_bbox_preds,
    #         all_gt_bboxes_list, all_gt_labels_list, 
    #         all_gt_bboxes_ignore_list)

    #     loss_dict = dict()
    #     # loss of proposal generated from encode feature map.
    #     if enc_cls_scores is not None:
    #         binary_labels_list = [
    #             torch.zeros_like(gt_labels_list[i])
    #             for i in range(len(all_gt_labels_list))
    #         ]
    #         enc_loss_cls, enc_losses_bbox = \
    #             self.loss_single(enc_cls_scores, enc_bbox_preds,
    #                              gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
    #         loss_dict['enc_loss_cls'] = enc_loss_cls
    #         loss_dict['enc_loss_bbox'] = enc_losses_bbox

    #     # loss from the last decoder layer
    #     loss_dict['loss_cls'] = losses_cls[-1]
    #     loss_dict['loss_bbox'] = losses_bbox[-1]

    #     # loss from other decoder layers
    #     num_dec_layer = 0
    #     for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
    #                                        losses_bbox[:-1]):
    #         loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
    #         loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
    #         num_dec_layer += 1
    #     return loss_dict

    # @force_fp32(apply_to=('preds_dicts'))
    # def get_bboxes(self, preds_dicts, img_metas, rescale=False):
    #     """Generate bboxes from bbox head predictions.
    #     Args:
    #         preds_dicts (tuple[list[dict]]): Prediction results.
    #         img_metas (list[dict]): Point cloud and image's meta info.
    #     Returns:
    #         list[dict]: Decoded bbox, scores and labels after nms.
    #     """
    #     preds_dicts = self.bbox_coder.decode(preds_dicts)
    #     num_samples = len(preds_dicts)

    #     ret_list = []
    #     for i in range(num_samples):
    #         preds = preds_dicts[i]
    #         bboxes = preds['bboxes']
    #         bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
    #         bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
    #         scores = preds['scores']
    #         labels = preds['labels']
    #         ret_list.append([bboxes, scores, labels])
    #     return ret_list