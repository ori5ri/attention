import mmcv
import copy
import torch
import torch.nn as nn
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np


def get_weight(model):
    for p in model.parameters():
        if p.requires_grad and len(p.data) != 4:
            yield p


def weight(model):
    result = []
    for param in model.parameters():
        if len(param.data) != 4:
            result.append({'params': param})
    return result


def get_net(cfg):
    model = build_detector(cfg.model)
    model = model.cuda()
    return model


def get_cfg(config_path):
    cfg = Config.fromfile(config_path)

    return cfg


def get_dataset(cfg):
    datasets = [build_dataset(cfg.data.train)]

    return datasets[0]


# def get_warmup_lr(optimizer, iter, wa)

def warmup_optim(optimizer, iter, warm_up_iter, warmup_ratio, start_lr):
    if iter < warm_up_iter:
        optimizer.param_groups[0]['lr'] += start_lr * (1 - warmup_ratio) / warm_up_iter
    elif iter == warm_up_iter:
        optimizer.param_groups[0]['lr'] = start_lr


def adjust_optim(optimizer, epoch, steps, gamma):
    if epoch in steps:
        optimizer.param_groups[0]['lr'] *= gamma


def compute_loss(model, data):
    x = model.extract_feat(data['img'].data[0].cuda())

    losses = dict()

    img_metas = data['img_metas'].data[0]
    gt_bboxes = [x.cuda() for x in data['gt_bboxes'].data[0]]
    gt_labels = [x.cuda() for x in data['gt_labels'].data[0]]

    # RPN forward and loss
    proposal_cfg = model.train_cfg.get('rpn_proposal',
                                       model.test_cfg.rpn)
    rpn_losses, proposal_list = model.rpn_head.forward_train(
        x,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        proposal_cfg=proposal_cfg)
    losses.update(rpn_losses)

    roi_losses = model.roi_head.forward_train(x, img_metas, proposal_list,
                                              gt_bboxes, gt_labels)
    losses.update(roi_losses)
    return losses


def virtual_step(model, v_model, train_data, xi, w_optim, cfg):
    """
    Compute unrolled weight w' (virtual step)

    Step process:
    1) forward
    2) calc loss
    3) compute gradient (by backprop)
    4) update gradient

    Args:
        xi: learning rate for virtual gradient step (same as weights lr)
        w_optim: weights optimizer
    """
    # forward & calc loss
    losses = compute_loss(model, train_data)  # L_trn(w)

    loss, _ = model._parse_losses(losses)
    loss.requires_grad_(True)

    # compute gradient
    gradients = torch.autograd.grad(loss, get_weight(model))

    # do virtual step (update gradient)
    # below operations do not need gradient tracking
    with torch.no_grad():
        # dict key is not the value, but the pointer. So original network weight have to
        # be iterated also.
        for w, vw, g in zip(get_weight(model), get_weight(v_model), gradients):
            m = w_optim.state[w].get('momentum_buffer', 0.) * cfg.optimizer.momentum
            vw.copy_(w - xi * (m + g + cfg.optimizer.weight_decay * w))

        # synchronize alphas
        for a, va in zip(model.neck.alphas(), v_model.neck.alphas()):
            va.copy_(a)


def compute_hessian(model, dw, train_data):
    """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
    norm = torch.cat([w.view(-1) for w in dw]).norm()
    eps = 0.01 / norm

    # w+ = w + eps*dw`
    with torch.no_grad():
        for p, d in zip(get_weight(model), dw):
            print(type(p))
            print(type(eps))
            print(type(d))
            p += eps * d
    losses = compute_loss(model, train_data)
    loss, _ = model._parse_losses(losses)
    loss.requires_grad_(True)

    dalpha_pos = torch.autograd.grad(loss, model.neck.alphas())  # dalpha { L_trn(w+) }

    # w- = w - eps*dw`
    with torch.no_grad():
        for p, d in zip(get_weight(model), dw):
            p -= 2. * eps * d
    losses = compute_loss(model, train_data)

    loss, _ = model._parse_losses(losses)
    loss.requires_grad_(True)

    dalpha_neg = torch.autograd.grad(loss, model.neck.alphas())  # dalpha { L_trn(w-) }

    # recover w
    with torch.no_grad():
        for p, d in zip(weight(model), dw):
            p += eps * d

    hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
    return hessian


if __name__ == '__main__':
    config_path = '../mmdetection/configs/attention/faster_rcnn_r50_attention_darts.py'
    cfg = get_cfg(config_path)
    dataset = get_dataset(cfg)
    model = get_net(cfg)
    model.train()

    # weights optimizer
    alpha_lr = 3e-4
    alpha_weight_decay = 1e-3

    # weight optimizer
    w_optim = torch.optim.SGD(weight(model),
                              cfg.optimizer.lr * cfg.lr_config.warmup_ratio,
                              momentum=cfg.optimizer.momentum,
                              weight_decay=cfg.optimizer.weight_decay)
    # lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(w_optim, step_size=cfg.lr_config)

    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.neck.alphas(), alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=alpha_weight_decay)

    n_train = len(dataset)
    split = n_train // 2
    # print(dataset)
    # train_dataset = dataset[:split]
    # valid_dataset = dataset[split:]
    # TODO dataset 쪼개기
    train_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu)

    val_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu)

    # train_loader = data_loaders[:split]
    # val_loader = data_loaders[split:]
    v_model = copy.deepcopy(model)
    for epoch in range(cfg.runner.max_epochs):
        for iter, (train_data, val_data) in tqdm(enumerate(zip(train_loader, val_loader))):

            # phase 2. architect step (alpha)
            alpha_optim.zero_grad()
            virtual_step(model, v_model, train_data, 0.01, w_optim, cfg)
            losses = compute_loss(v_model, val_data)
            loss, _ = v_model._parse_losses(losses)
            loss.requires_grad_(True)

            # TODO compute hessian
            # compute gradient
            # v_alphas = tuple(v_model.neck.alphas())
            # v_weights = tuple(get_weight(v_model))
            # v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
            # dalpha = v_grads[:len(v_alphas)]
            # dw = v_grads[len(v_alphas):]
            #
            # hessian = compute_hessian(model, dw, train_data)
            #
            # # update final gradient = dalpha - xi*hessian
            # with torch.no_grad():
            #     for alpha, da, h in zip(model.neck.alphas(), dalpha, hessian):
            #         alpha.grad = da - 0.01 * h
            alpha_optim.step()

            # phase 1. child network step (w)
            w_optim.zero_grad()
            losses = compute_loss(model, train_data)
            loss, _ = model._parse_losses(losses)
            loss.requires_grad_(True)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(get_weight(model), cfg.optimizer_config.grad_clip.max_norm)
            w_optim.step()
            tqdm.write(str(losses))
            if epoch == 0:
                warmup_optim(w_optim, iter, cfg.lr_config.warmup_iters, cfg.lr_config.warmup_ratio, cfg.optimizer.lr)
        adjust_optim(w_optim, epoch + 1, cfg.lr_config.step, 0.1)
