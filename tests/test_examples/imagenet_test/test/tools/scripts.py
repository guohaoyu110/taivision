import collections
import numpy as np
import os
import time
from tqdm import tqdm

from apex import amp
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval

from simpleAICV.classification.common import ClassificationDataPrefetcher, AverageMeter, accuracy
# from simpleAICV.detection.common import DetectionDataPrefetcher
# from simpleAICV.segmentation.common import SegmentationDataPrefetcher


def validate_classification(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for images, targets in tqdm(val_loader):
            if model_on_cuda:
                images, targets = images.cuda(), targets.cuda()

            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            batch_time.update(time.time() - end)

            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # print('acc1', acc1)
            # print('acc5', acc5)
            # acc1 tensor([78.9062], device='cuda:1')
            # acc5 tensor([92.9688], device='cuda:1')
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            end = time.time()

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / config.batch_size * 1000
    per_image_inference_time = batch_time.avg / config.batch_size * 1000

    return top1.avg, top5.avg, losses.avg, per_image_load_time, per_image_inference_time


def train_classification(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train classification model for one epoch
    '''
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank() if config.distributed else None
    if config.distributed:
        gpus_num = torch.cuda.device_count()
        iters = len(train_loader.dataset) // (
            config.batch_size * gpus_num) if config.distributed else len(
                train_loader.dataset) // config.batch_size
    else:
        iters = len(train_loader.dataset) // config.batch_size

    prefetcher = ClassificationDataPrefetcher(train_loader)
    images, targets = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, targets = images.cuda(), targets.cuda()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss = loss / config.accumulation_steps

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if iter_index % config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        images, targets = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, top1: {acc1.item():.2f}%, top5: {acc5.item():.2f}%, loss: {loss.item():.4f}'
            logger.info(log_info) if (config.distributed and local_rank
                                      == 0) or not config.distributed else None

        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg, losses.avg


def validate_KD(val_loader, model, criterion):
    top1 = AverageMeter()
    top5 = AverageMeter()
    total_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for images, targets in tqdm(val_loader):
            if model_on_cuda:
                images, targets = images.cuda(), targets.cuda()

            tea_outputs, stu_outputs = model(images)
            total_loss = 0
            for loss_name in criterion.keys():
                if 'KD' in loss_name:
                    temp_loss = criterion[loss_name](stu_outputs, tea_outputs)
                else:
                    temp_loss = criterion[loss_name](stu_outputs, targets)

                total_loss += temp_loss

            acc1, acc5 = accuracy(stu_outputs, targets, topk=(1, 5))

            total_losses.update(total_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    return top1.avg, top5.avg, total_losses.avg


def train_KD(train_loader, model, criterion, optimizer, scheduler, epoch,
             logger, config):
    '''
    train classification model for one epoch
    '''
    top1 = AverageMeter()
    top5 = AverageMeter()
    total_losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank() if config.distributed else None
    if config.distributed:
        gpus_num = torch.cuda.device_count()
        iters = len(train_loader.dataset) // (
            config.batch_size * gpus_num) if config.distributed else len(
                train_loader.dataset) // config.batch_size
    else:
        iters = len(train_loader.dataset) // config.batch_size

    prefetcher = ClassificationDataPrefetcher(train_loader)
    images, targets = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, targets = images.cuda(), targets.cuda()
        tea_outputs, stu_outputs = model(images)
        loss = 0
        loss_value = {}
        for loss_name in criterion.keys():
            if 'KD' in loss_name:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs)
            else:
                temp_loss = criterion[loss_name](stu_outputs, targets)

            loss_value[loss_name] = temp_loss
            loss += temp_loss

        total_losses.update(loss.item(), images.size(0))
        loss = loss / config.accumulation_steps

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if iter_index % config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(stu_outputs, targets, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        images, targets = prefetcher.next()

        log_info = ''
        if iter_index % config.print_interval == 0:
            log_info += f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, top1: {acc1.item():.2f}%, top5: {acc5.item():.2f}%, total_loss: {loss.item():.4f} '
            for loss_name in criterion.keys():
                log_info += f'{loss_name}: {loss_value[loss_name].item():.4f} '
            logger.info(log_info) if (config.distributed and local_rank
                                      == 0) or not config.distributed else None

        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg, total_losses.avg


def compute_voc_ap(recall, precision, use_07_metric=True):
    if use_07_metric:
        # use voc 2007 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                # get max precision  for recall >= t
                p = np.max(precision[recall >= t])
            # average 11 recall point precision
            ap = ap + p / 11.
    else:
        # use voc>=2010 metric,average all different recall precision as ap
        # recall add first value 0. and last value 1.
        mrecall = np.concatenate(([0.], recall, [1.]))
        # precision add first value 0. and last value 0.
        mprecision = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mprecision.size - 1, 0, -1):
            mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrecall[1:] != mrecall[:-1])[0]

        # sum (\Delta recall) * prec
        ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])

    return ap


def compute_ious(a, b):
    '''
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    '''

    a = np.expand_dims(a, axis=1)  # [N,1,4]
    b = np.expand_dims(b, axis=0)  # [1,M,4]

    overlap = np.maximum(0.0,
                         np.minimum(a[..., 2:], b[..., 2:]) -
                         np.maximum(a[..., :2], b[..., :2]))  # [N,M,(w,h)]

    overlap = np.prod(overlap, axis=-1)  # [N,M]

    area_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    area_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)

    iou = overlap / (area_a + area_b - overlap)

    return iou


