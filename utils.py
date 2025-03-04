import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import pytorch_ssim
import pytorch_iou
from edgeloss import *



def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            #mode = config.mode,
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()


def bbox_iou(msk, out):
        gt_list = []
        pred_list  = []
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        gts = msk.squeeze(1).cpu().detach().numpy()
        preds = out.squeeze(1).cpu().detach().numpy()
        gt_list.append(gts)
        pred_list.append(preds)
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= 0.5, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        smooth = 1e-5
        intersection = (y_pre & y_true).sum()
        union = (y_pre | y_true).sum()
        miou = (intersection + smooth) / (union + smooth)

        return miou  # IoU



class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (self.bcedice(gt_pre5, target) * 0.1 +
                   self.bcedice(gt_pre4, target) * 0.2 +
                   self.bcedice(gt_pre3, target) * 0.3 +
                   self.bcedice(gt_pre2, target) * 0.4 +
                   self.bcedice(gt_pre1, target) * 0.5)

        return bcediceloss + gt_loss


bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)



class bce_ssim_loss(nn.Module):
    def __init__(self):
        super(bce_ssim_loss, self).__init__()
        self.bce = nn.BCELoss(size_average=True)
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
        self.iou_loss = pytorch_iou.IOU(size_average=True)
        self.dice_loss = DiceLoss()
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        ssimloss = self.ssim_loss(pred, target)
        iouloss = self.iou_loss(pred, target)
        diceloss = self.dice_loss(pred, target)
        loss =  bceloss +  diceloss +(1-ssimloss ) + iouloss

        return loss

class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__()
        self.bce = nn.BCELoss(size_average=True)
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
        self.iou_loss = pytorch_iou.IOU(size_average=True)
        self.dice_loss = DiceLoss()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        ssimloss = self.ssim_loss(pred, target)
        iouloss = self.iou_loss(pred, target)
        mseloss = self.mse(pred, target)
        diceloss = self.dice_loss(pred, target)
        loss =  diceloss  + bceloss  + (1-ssimloss ) #+ mseloss #+ iouloss +  diceloss # #
        return loss

class atten_loss(nn.Module):
    def __init__(self):
        super(atten_loss, self).__init__()
        self.bce = nn.BCELoss(size_average=True)
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
        self.iou_loss = pytorch_iou.IOU(size_average=True)
        self.dice_loss = DiceLoss()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice_loss(pred, target)
        ssimloss = self.ssim_loss(pred, target)
        iouloss =  self.iou_loss(pred, target)#  self.iou_loss(pred, target)
        mseloss = self.mse(pred, target)
        loss =  diceloss  + bceloss  + (1-ssimloss ) + iouloss #+  diceloss # #+ mseloss
        return loss



class muti_bce_loss_fusion(nn.Module):

    def __init__(self):
        super(muti_bce_loss_fusion, self).__init__()
        self.bce_ssim_loss = bce_ssim_loss()
        self.edge_loss = edge_loss()
        self.atten_loss = atten_loss()

        self.w = nn.Parameter(torch.ones(14)).cuda()

        self.bce = nn.BCELoss(size_average=True)
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
        self.iou_loss = pytorch_iou.IOU(size_average=True)
        self.dice_loss = DiceLoss()



    def forward(self, gt_pre, out, aux, edge_gt, target):



            w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
            w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
            w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
            w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
            w5 = torch.exp(self.w[4]) / torch.sum(torch.exp(self.w))
            w6 = torch.exp(self.w[5]) / torch.sum(torch.exp(self.w))
            w7 = torch.exp(self.w[6]) / torch.sum(torch.exp(self.w))
            w8 = torch.exp(self.w[7]) / torch.sum(torch.exp(self.w))
            w9 = torch.exp(self.w[8]) / torch.sum(torch.exp(self.w))
            w10 = torch.exp(self.w[9]) / torch.sum(torch.exp(self.w))
            w11 =torch.exp(self.w[10]) / torch.sum(torch.exp(self.w))
            w12 =  torch.exp(self.w[11]) / torch.sum(torch.exp(self.w))
            w13 = torch.exp(self.w[12]) / torch.sum(torch.exp(self.w))
            w14 = torch.exp(self.w[13]) / torch.sum(torch.exp(self.w))

            gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
            gt_loss = (self.bce_ssim_loss(gt_pre5, target) *  w1 +
                        self.bce_ssim_loss(gt_pre4, target) * w2 +
                        self.bce_ssim_loss(gt_pre3, target) * w3 +
                        self.bce_ssim_loss(gt_pre2, target) * w4 +
                        self.bce_ssim_loss(gt_pre1, target) * w5

                       )


            edge_gt_pre5, edge_gt_pre4, edge_gt_pre3, edge_gt_pre2, edge_gt_pre1 = edge_gt
            edge_gt__loss2 = (
                        self.atten_loss (edge_gt_pre5, target) * w7 +
                        self.atten_loss (edge_gt_pre4, target) * w8 +
                        self.atten_loss (edge_gt_pre3, target) * w9 +
                        self.atten_loss (edge_gt_pre2, target) * w10 +
                        self.atten_loss (edge_gt_pre1, target) * w11
                        )
            out_loss = self.bce_ssim_loss(out, target)  * w14

            return   gt_loss + out_loss +  edge_gt__loss2  # + aux_loss




class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk