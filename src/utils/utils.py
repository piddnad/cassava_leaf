'''
Some util functions
Part of the code is referenced from Kaggle
'''

import os
import cv2
import torch
import random
import numpy as np
from . import fmix
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
from .tempered_loss import *

def seed_everything(seed):
    '''固定各类随机种子，方便消融实验.
    Args:
        seed :  int
    '''
    # 固定 scipy 的随机种子
    random.seed(seed)  # 固定 random 库的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 python hash 的随机性（并不一定有效）
    np.random.seed(seed)  # 固定 numpy  的随机种子
    torch.manual_seed(seed)  # 固定 torch cpu 计算的随机种子
    torch.cuda.manual_seed(seed)  # 固定 cuda 计算的随机种子
    torch.backends.cudnn.deterministic = True  # 是否将卷积算子的计算实现固定。torch 的底层有不同的库来实现卷积算子
    torch.backends.cudnn.benchmark = True  # 是否开启自动优化，选择最快的卷积计算方法


def get_img(path):
    '''使用 opencv 加载图片.
    由于历史原因，opencv 读取的图片格式是 bgr
    Args:
        path : str  图片文件路径 e.g '../data/train_img/1.jpg'
    '''
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def rand_bbox(size, lam):
    '''cutmix 的 bbox 截取函数
    Args:
        size : tuple 图片尺寸 e.g (256,256)
        lam  : float 截取比例
    Returns:
        bbox 的左上角和右下角坐标
        int,int,int,int
    '''
    W = size[0]  # 截取图片的宽度
    H = size[1]  # 截取图片的高度
    cut_rat = np.sqrt(1. - lam)  # 需要截取的 bbox 比例
    cut_w = np.int(W * cut_rat)  # 需要截取的 bbox 宽度
    cut_h = np.int(H * cut_rat)  # 需要截取的 bbox 高度

    cx = np.random.randint(W)  # 均匀分布采样，随机选择截取的 bbox 的中心点 x 坐标
    cy = np.random.randint(H)  # 均匀分布采样，随机选择截取的 bbox 的中心点 y 坐标

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # 左上角 x 坐标
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # 左上角 y 坐标
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # 右下角 x 坐标
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # 右下角 y 坐标
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    '''木薯叶比赛数据加载类
    Attributes:
        __len__ : 数据的样本个数.
        __getitem__ : 索引函数.
    '''
    def __init__(
            self,
            df,
            data_root,
            transforms=None,
            output_label=True,
            one_hot_label=False,
            do_fmix=False,
            fmix_params={
                'alpha': 1.,
                'decay_power': 3.,
                'shape': (512, 512),
                'max_soft': 0.3,
                'reformulate': False
            },
            do_cutmix=False,
            cutmix_params={
                'alpha': 1,
            }):
        '''
        Args:
            df : DataFrame , 样本图片的文件名和标签
            data_root : str , 图片所在的文件路径，绝对路径
            transforms : object , 图片增强
            output_label : bool , 是否输出标签
            one_hot_label : bool , 是否进行 onehot 编码
            do_fmix : bool , 是否使用 fmix
            fmix_params :dict , fmix 的参数 {'alpha':1.,'decay_power':3.,'shape':(256,256),'max_soft':0.3,'reformulate':False}
            do_cutmix : bool, 是否使用 cutmix
            cutmix_params : dict , cutmix 的参数 {'alpha':1.}
        Raises:

        '''
        super().__init__()
        self.df = df.reset_index(drop=True).copy()  # 重新生成索引
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        if output_label:
            self.labels = self.df['label'].values
            if one_hot_label:
                self.labels = np.eye(self.df['label'].max() +
                                     1)[self.labels]  # 使用单位矩阵生成 onehot 编码

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        '''
        Args:
            index : int , 索引
        Returns:
            img, target(optional)
        '''
        if self.output_label:
            target = self.labels[index]

        img = get_img(
            os.path.join(self.data_root,
                         self.df.loc[index]['image_id']))  # 拼接地址，加载图片

        if self.transforms:  # 使用图片增强
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(
                0., 1., size=1)[0] > 0.5:  # 50% 概率触发 fmix 数据增强

            with torch.no_grad():
                lam, mask = sample_mask(
                    **self.fmix_params)  # 可以考虑魔改，使用 clip 规定上下限制

                fmix_ix = np.random.choice(self.df.index,
                                           size=1)[0]  # 随机选择待 mix 的图片
                fmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                img = mask_torch * img + (1. - mask_torch) * fmix_img  # mix 图片

                rate = mask.sum() / float(img.size)  # 获取 mix 的 rate
                target = rate * target + (
                    1. - rate) * self.labels[fmix_ix]  # target 进行 mix

        if self.do_cutmix and np.random.uniform(
                0., 1., size=1)[0] > 0.5:  # 50% 概率触发 cutmix 数据增强
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(
                    np.random.beta(self.cutmix_params['alpha'],
                                   self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_img.shape[:2], lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2,
                                                        bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) *
                            (bby2 - bby1) / float(img.size))  # 获取 mix 的 rate
                target = rate * target + (
                    1. - rate) * self.labels[cmix_ix]  # target 进行 mix

        if self.output_label:
            return img, target
        else:
            return img


def prepare_dataloader(df, trn_idx, val_idx, data_root, trn_transform,
                       val_transform, bs, n_job):
    '''多进程数据生成器
    Args:
        df : DataFrame , 样本图片的文件名和标签
        trn_idx : ndarray , 训练集索引列表
        val_idx : ndarray , 验证集索引列表
        data_root : str , 图片文件所在路径
        trn_transform : object , 训练集图像增强器
        val_transform : object , 验证集图像增强器
        bs : int , 每次 batchsize 个数
        n_job : int , 使用进程数量
    Returns:
        train_loader, val_loader , 训练集和验证集的数据生成器
    '''
    train_ = df.loc[trn_idx, :].reset_index(drop=True)  # 重新生成索引
    valid_ = df.loc[val_idx, :].reset_index(drop=True)  # 重新生成索引

    train_ds = CassavaDataset(train_,
                              data_root,
                              transforms=trn_transform,
                              output_label=True,
                              one_hot_label=False,
                              do_fmix=False,
                              do_cutmix=False)
    valid_ds = CassavaDataset(valid_,
                              data_root,
                              transforms=val_transform,
                              output_label=True,
                              one_hot_label=False,
                              do_fmix=False,
                              do_cutmix=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=n_job,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=n_job,
    )

    return train_loader, val_loader


def train_one_epoch(epoch,
                    model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    device,
                    scaler,
                    scheduler=None,
                    schd_batch_update=False,
                    accum_iter=2):
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

        # Bi-tempered loss
        # image_labels = torch.nn.functional.one_hot(image_labels, 5).float().to(device)
        # loss = torch.mean(bi_tempered_logistic_loss(activations=image_preds, labels=image_labels, t1=0.6, t2=2.0, label_smoothing=0.3))

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

        # Bi-tempered loss
        # image_labels = torch.nn.functional.one_hot(image_labels, 5).float().to(device)
        # loss = torch.mean(bi_tempered_logistic_loss(activations=image_preds, labels=image_labels, t1=0.5, t2=1.5))

        loss_sum += loss.item() * image_labels.shape[0]  # 计算损失和
        sample_num += image_labels.shape[0]  # 样本数

        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'  # 打印平均 loss
        pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format(
        (image_preds_all == image_targets_all).mean()))  # 打印准确率


if __name__ == '__main__':
    pass
