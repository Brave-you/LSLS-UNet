import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import matplotlib.pyplot as plt
from edgeloss import *
import os
def save_visualization(input_img, true_mask, pred_mask, save_dir):


    os.makedirs(save_dir, exist_ok=True)

    pred_mask = torch.where(pred_mask >= 0.5, 1, 0)
    true_mask = torch.where(true_mask >= 0.5, 1, 0)


    img_np = input_img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # HWC
    true_np = true_mask.cpu().numpy().squeeze().astype(np.uint8) * 255
    pred_np = pred_mask.cpu().numpy().squeeze().astype(np.uint8) * 255

    smooth = 1e-5
    intersection = (pred_np & true_np).sum()
    union = (pred_np | true_np).sum()
    miou = (intersection + smooth) / (union + smooth)

    plt.figure(figsize=(24, 8))


    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis('off')


    plt.subplot(1, 3, 2)
    plt.imshow(true_np, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')


    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='gray')
    plt.title("Prediction")
    plt.axis('off')



    save_path = os.path.join(save_dir, f"sample_{miou}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def visualize_tensor(tensor, title=''):

    image_np = tensor.squeeze().cpu().detach().numpy()

    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        gt_pre, out ,aux , edge = model(images)
        loss = criterion(gt_pre, out, aux , edge , targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)



    scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    total_miou = 0.0
    total = 0
    gt_list = []
    pred_list = []
    model.eval()
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out ,aux , edge = model(img)

            gts = msk.squeeze(1).cpu().detach().numpy()
            preds = out.squeeze(1).cpu().detach().numpy()
            gt_list.append(gts)
            pred_list.append(preds)
            preds = np.array(preds).reshape(-1)
            gts = np.array(gts).reshape(-1)

            y_pre = np.where(preds >= config.threshold, 1, 0)
            y_true = np.where(gts >= 0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)

            total_miou += miou
            total += 1
    # visualize_tensor(msk, title='Visualization of M_C Loss')
    # visualize_tensor(out, title='Visualization of M_C Loss')
    # visualize_tensor(aux, title='Visualization of M_C Loss')



    total_miou = total_miou / total
    pred_list = np.array(pred_list).reshape(-1)
    gt_list = np.array(gt_list).reshape(-1)
    y_pre = np.where(pred_list >= 0.5, 1, 0)
    y_true = np.where(gt_list >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]



    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch}, totalmiou and miou: {total_miou,miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
            specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    print(log_info)
    logger.info(log_info)



    return  - (total_miou + f1_or_dsc)

save_dir = os.path.join("D:\Python\LSLS-UNet-main\\vision", "visual_results")
def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    gt_list = []
    pred_list = []
    total_miou = 0.0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk0 = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre,  out0 , aux , edge = model(img)

            msk = msk0.squeeze(1).cpu().detach().numpy()
            out = out0.squeeze(1).cpu().detach().numpy()

            gt_list.append(msk)
            pred_list.append(out)

            y_pre = np.where(out >= config.threshold, 1, 0)
            y_true = np.where(msk >= 0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)

            total_miou += miou
            total += 1

            # if i % config.save_interval == 0:
            # kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12 = key_points
            # gt1, gt2, gt3, gt4, gt5 = gt_pre
            # save_imgs(img, msk, out, key_points, gt_pre, i, config.work_dir + 'outputs/' + 'ISIC2017' + '/', config.datasets, config.threshold, test_data_name=test_data_name)

            for i in range(img.shape[0]):
                # 获取当前样本数据
                img = img[i]  # (C, H, W)
                msk = msk[i]  # (H, W)
                out = out[i]  # (H, W)

                # 保存可视化结果
                save_visualization(img, msk0, out0, save_dir)
        total_miou = total_miou / total

        pred_list = np.array(pred_list).reshape(-1)
        gt_list = np.array(gt_list).reshape(-1)

        y_pre = np.where(pred_list >= 0.5, 1, 0)
        y_true = np.where(gt_list >= 0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, total_miou: {total_miou},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return total_miou
