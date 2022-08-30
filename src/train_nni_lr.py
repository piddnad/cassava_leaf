#  -*- coding: utf-8 -*-
#  @Time    : 2021/1/15
#  @Author  : Piddnad
#  @Email   : piddnad@gmail.com

import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from torch import nn
from tqdm import tqdm
import torch
import timm
import cv2
import pandas as pd
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
import nni
# from tempered_loss import *

CFG = {
    'rand_seed': 729,
    'fold_num': 5,
    'model_arch': 'vit_base_patch16_384',
    'img_size': 384,
    'epochs': 10,
    'T_0': 10,
    'train_bs': 8,  # x2
    'valid_bs': 8,  # x2
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    # 'verbose_step': 1,
    'device': 'cuda:0',
    'exp_name': 'vit_base_patch16_384_nni',
}

# reload(utils)
utils.seed_everything(CFG['rand_seed'])

train_img_path = r'../data/train_images'  # 训练集路径
train_csv_path = r'../data/train.csv'  # 训练集样本列表CSV

# 定义训练集数据增强
def get_train_transforms():
    return Compose([
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
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Resize(CFG['img_size'], CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def train_one_epoch(epoch,
                    model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    device,
                    scaler,
                    scheduler=None,
                    schd_batch_update=False,
                    accum_iter=2,
                    t1=1.0,
                    t2=1.0,
                    label_smoothing=0.0):
    '''训练集每个epoch训练函数
    Args:
        epoch : int , 训练到第几个 epoch
        model : object, 需要训练的模型
        loss_fn : object, 损失函数
        optimizer : object, 优化方法
        train_loader : object, 训练集数据生成器
        scaler : object, 梯度放大器
        device : str , 使用的训练设备 e.g 'cuda:0'
        scheduler : object , 学习率调整策略
        schd_batch_update : bool, 如果是 true 则每一个 batch 都调整，否则等一个 epoch 结束后再调整
        accum_iter : int , 梯度累加
    '''

    model.train()  # 开启训练模式

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # 构造进度条

    for step, (imgs, image_labels) in pbar:  # 遍历每个 batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        # with autocast():  # 开启自动混精度
        image_preds = model(imgs)  # 前向传播，计算预测值
        loss = loss_fn(image_preds, image_labels)  # 计算 loss

        # tempered loss
        # image_labels = torch.nn.functional.one_hot(image_labels, 5).float().to(device)
        # loss = torch.mean(bi_tempered_logistic_loss(activations=image_preds, labels=image_labels,
        #                                             t1=t1, t2=t2, label_smoothing=label_smoothing))

        scaler.scale(loss).backward()  # 对 loss scale, scale梯度

        # loss 正则,使用指数平均
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % accum_iter == 0) or ((step + 1) == len(train_loader)):
            scaler.step(
                optimizer)  # unscale 梯度, 如果梯度没有 overflow, 使用 opt 更新梯度, 否则不更新
            scaler.update()  # 等着下次 scale 梯度
            optimizer.zero_grad()  # 梯度清空

            if scheduler is not None and schd_batch_update:  # 学习率调整策略
                scheduler.step()

        # 打印 loss 值
        description = f'epoch {epoch} loss: {running_loss:.4f}'
        pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:  # 学习率调整策略
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    '''验证集 inference
    Args:
        epoch : int, 第几个 epoch
        model : object, 模型
        loss_fn : object, 损失函数
        val_loader ： object, 验证集数据生成器
        device : str , 使用的训练设备 e.g 'cuda:0'
    '''

    model.eval()  # 开启推断模式

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))  # 构造进度条

    for step, (imgs, image_labels) in pbar:  # 遍历每个 batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # 前向传播，计算预测值
        image_preds_all += [
            torch.argmax(image_preds, 1).detach().cpu().numpy()
        ]  # 获取预测标签
        image_targets_all += [image_labels.detach().cpu().numpy()]  # 获取真实标签

        loss = loss_fn(image_preds, image_labels)  # 计算损失
        # tempered loss
        # image_labels = torch.nn.functional.one_hot(image_labels, 5).float().to(device)
        # loss = torch.mean(bi_tempered_logistic_loss(activations=image_preds, labels=image_labels, t1=0.5, t2=1.5))

        loss_sum += loss.item() * image_labels.shape[0]  # 计算损失和
        sample_num += image_labels.shape[0]  # 样本数

        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'  # 打印平均 loss
        pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    val_acc = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format(val_acc))  # 打印准确率
    return val_acc

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


# NNI
tuner_params = nni.get_next_parameter()
lr_nni = tuner_params['lr']
min_lr_nni = tuner_params['min_lr']


# 训练 main loop
for fold, (trn_idx, val_idx) in enumerate(folds):
    if fold == 4:  # try one fold
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
                                    lr=lr_nni,
                                    weight_decay=CFG['weight_decay'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CFG['T_0'],
            T_mult=1,
            eta_min=min_lr_nni,
            last_epoch=-1)

        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scaler,
                            scheduler=scheduler, schd_batch_update=False, accum_iter=CFG['accum_iter'])

            with torch.no_grad():
                val_acc = valid_one_epoch(epoch, model, loss_fn, val_loader, device)
                nni.report_intermediate_result(val_acc)

            # save model in every epoch
            # save_path = '../model/{}'.format(CFG['exp_name'])
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # torch.save(
            #     model.state_dict(),
            #     '../model/{}/{}_fold_{}_{}'.format(CFG['exp_name'], CFG['model_arch'], fold, epoch))

        nni.report_final_result(val_acc)
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        with torch.cuda.device(CFG['device']):  # 如果不添加，则默认在 gpu0 写 ~500M ，此时若 gpu0 被占用会报 out of memory 错误
            torch.cuda.empty_cache()
