#  -*- coding: utf-8 -*-
#  @Time    : 2021/1/8
#  @Author  : Piddnad
#  @Email   : piddnad@gmail.com

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from torch import nn
from tqdm import tqdm
import torch
import timm
import cv2
import pandas as pd
import numpy as np
from utils import utils
# from imp import reload
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
import os

CFG = {
    'rand_seed': 729,
    'fold_num': 5,
    'model_arch': 'vit_base_patch16_384',
    'img_size': 384,
    'epochs': 10,
    'T_0': 10,
    'train_bs': 8,  # x2
    'valid_bs': 8,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    # 'verbose_step': 1,
    'device': 'cuda:3',
    'exp_name': 'vit_base_patch16_384_bs8x2_mix19data',
}

# reload(utils)
utils.seed_everything(CFG['rand_seed'])

train_img_path = r'../data/train_images'  # 训练集路径
train_csv_path = r'../data/train_mixed.csv'  # 训练集样本列表CSV

# 定义训练集数据增强
def get_train_transforms():
    return Compose([
        Resize(CFG['img_size'], CFG['img_size'], p=1.0),
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)

# 定义验证集数据增强
def get_valid_transforms():
    return Compose([
        Resize(CFG['img_size'], CFG['img_size'], p=1.0),
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Resize(CFG['img_size'], CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

# 分类器定义
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes=5)
        # if 'efficientnet' in model_arch:  # efficientnet
        #     n_features = self.model.classifier.in_features
        #     self.model.classifier = nn.Linear(n_features, n_class)
        # else:  # resnest
        #     n_features = self.model.fc.in_features
        #     self.model.fc = nn.Linear(n_features, n_class)
    def forward(self, x):
        x = self.model(x)
        return x

# 构建每个 fold 的数据
train = pd.read_csv(train_csv_path)
folds = StratifiedKFold(n_splits=CFG['fold_num'],
                        shuffle=True,
                        random_state=CFG['rand_seed']).split(
                            np.arange(train.shape[0]), train.label.values)
trn_transform = get_train_transforms()
val_transform = get_valid_transforms()

# 训练 main loop
for fold, (trn_idx, val_idx) in enumerate(folds):
    # if fold == 0:  # try one fold
    print('Training with fold {} started'.format(fold))
    print('Train : {}, Val : {}'.format(len(trn_idx), len(val_idx)))
    train_loader, val_loader = utils.prepare_dataloader(train,
                                                      trn_idx,
                                                      val_idx,
                                                      data_root = train_img_path,
                                                      trn_transform = trn_transform,
                                                      val_transform = val_transform,
                                                      bs = CFG['train_bs'],
                                                      n_job = CFG['num_workers'])

    device = torch.device(CFG['device'])

    model = CassvaImgClassifier(CFG['model_arch'],
                                train.label.nunique(),
                                pretrained=True).to(device)

    # model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG['lr'],
                                 weight_decay=CFG['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CFG['T_0'],
        T_mult=1,
        eta_min=CFG['min_lr'],
        last_epoch=-1)

    loss_tr = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(CFG['epochs']):
        utils.train_one_epoch(epoch,
                            model,
                            loss_tr,
                            optimizer,
                            train_loader,
                            device,
                            scaler,
                            scheduler=scheduler,
                            schd_batch_update=False,
                            accum_iter=CFG['accum_iter'])

        with torch.no_grad():
            utils.valid_one_epoch(epoch,
                                model,
                                loss_fn,
                                val_loader,
                                device)

        # save model in every epoch
        save_path = '../model/{}'.format(CFG['exp_name'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(
            model.state_dict(),
            '../model/{}/{}_fold_{}_{}'.format(CFG['exp_name'], CFG['model_arch'], fold, epoch))

    del model, optimizer, train_loader, val_loader, scaler, scheduler
    with torch.cuda.device(CFG['device']):  # 如果不添加，则默认在 gpu0 写 ~500M ，此时若 gpu0 被占用会报 out of memory 错误
        torch.cuda.empty_cache()
