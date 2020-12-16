import argparse
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Ranger Optimizer
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Oof
import eval as eval_script
from data import *
from layers.modules import MultiBoxLoss
from utils import timer
from utils.augmentations import BaseTransform, SSDAugmentation
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from yolact import Discriminator, Discriminator_StandFord, Discriminator_Dcgan, Yolact


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--num_workers', default=os.cpu_count(), type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss, pred_seg:bool=False):
        super().__init__()

        #NOTE
        self.pred_seg = pred_seg
        self.net = net
        self.criterion = criterion
    
    # one can also check the output format of the dataset
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        if self.pred_seg:
            losses, seg_list, pred_list = self.criterion(self.net, preds, targets, masks, num_crowds)
            return losses, seg_list, pred_list
        else:
            losses = self.criterion(self.net, preds, targets, masks, num_crowds)
            return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out
# NOTE
def gan_init(mod):
    # Since the output value (before final activation) of discriminator 
    # with possible negative initial values will easily be became zero while pass though
    # the ReLU. This will make the BCE loss explode
    # By default, while BCE loss explode, torch will clamp the value
    # to 100, in order to keep the gradient calculation vaild. 
    if isinstance(mod, nn.Conv2d):
        # advice for relu activation
        # nn.init.kaiming_uniform_(mod.weight)
        # nn.init.zeros_(mod.bias)
        
        # advice for leakyReLU
        nn.init.normal_(mod.weight, 0.0, 0.02)
    elif isinstance(mod, nn.BatchNorm2d):
        nn.init.normal_(mod.weight, 1.0, 0.02)
        nn.init.constant_(mod.bias, 0)

def freeze_pretrain(model, freeze=True):
    if freeze:
        for par in model.parameters():
            par.requires_grad = False
    else:
        for par in model.parameters():
            par.requires_grad = True

def seg_mask_clas(seg_list, pred_list, datum):
    # need to transform seg_list to [b, classes, h, w]
    seg_list = [v.permute(2,0,1).contiguous() for v in seg_list]
    b = len(seg_list) # batch size
    _, seg_h, seg_w = seg_list[0].size()

    # empty tensors to be pasted on later (neglact the background class)
    seg_clas    = torch.zeros(b, cfg.num_classes-1, seg_h, seg_w)
    mask_clas   = torch.zeros(b, cfg.num_classes-1, seg_h, seg_w)

    # TODO: might change to upsample the prediction map
    # seg_clas = interpolate(seg_clas, size = (550,550), 
    #                        mode='bilinear',align_corners=True)

    # get the gt target label
    target_list = [target for target in datum[1][0]]
    # input ground truth mask is [b, instances, 550, 550]
    # need to interpolate to     [b, classes, seg_h, seg_w]
    mask_list   = [interpolate(mask.unsqueeze(0), size = (seg_h,seg_w),mode='bilinear', \
                                        align_corners=False).squeeze() for mask in datum[1][1]]
    # all prepared, let's paste them all!
    # NOTE
    # === same class will be pasted together ===
    for idx in range(b):
        for i, (pred, i_target) in enumerate(zip(pred_list[idx], target_list[idx])):
            seg_clas[idx,  pred, ...]                += seg_list[idx][i,...]
            mask_clas[idx, i_target[-1].long(), ...] += mask_list[idx][i,...]

    seg_clas = torch.clamp(seg_clas, 0, 1)

    return seg_clas, mask_clas, b, (seg_h, seg_w)


def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()
    print('\n--- Generator created! ---')

    # NOTE
    # I maunally set the original image size and seg size as 138
    # might change in the future, for example 550
    if cfg.pred_seg:
        dis_size = 138
        dis_net  = Discriminator_Dcgan(i_size = dis_size, s_size = dis_size)
        # set the dis net's initial parameter values
        dis_net.apply(gan_init)
        dis_net.train()
        print('--- Discriminator created! ---\n')

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    # optimizer_gen = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.decay)
    # if cfg.pred_seg:
    #     optimizer_dis = optim.SGD(dis_net.parameters(), lr=cfg.dis_lr, momentum=args.momentum,
    #                         weight_decay=args.decay)
    #     schedule_dis  = ReduceLROnPlateau(optimizer_dis, mode = 'min', patience=6, min_lr=1E-6)

    # NOTE: Using the Ranger Optimizer for the generator
    optimizer_gen     = Ranger(net.parameters(), lr = args.lr, weight_decay=args.decay)
    if cfg.pred_seg:
        optimizer_dis = optim.SGD(dis_net.parameters(), lr=cfg.dis_lr)
        schedule_dis  = ReduceLROnPlateau(optimizer_dis, mode = 'min', patience=6, min_lr=1E-6)

    criterion     = MultiBoxLoss(num_classes=cfg.num_classes,
                                pos_threshold=cfg.positive_iou_threshold,
                                neg_threshold=cfg.negative_iou_threshold,
                                negpos_ratio=cfg.ohem_negpos_ratio, pred_seg=cfg.pred_seg)

    criterion_dis = nn.BCELoss()

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion, pred_seg=cfg.pred_seg))
    if args.cuda:
        net     = net.cuda()
        if cfg.pred_seg:
            dis_net = dis_net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())

    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # NOTE
    val_loader  = data.DataLoader(val_dataset, args.batch_size,
                                  num_workers=args.num_workers*2,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
                      # TODO: global command can modify global variable inside of the function.
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue
            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer_gen, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer_gen, args.lr * (args.gamma ** step_index))
                
                # Zero the grad to get ready to compute gradients
                optimizer_gen.zero_grad()
                # NOTE
                if cfg.pred_seg:
                    # ====== GAN Train ======
                    optimizer_dis.zero_grad()
                    dis_net.zero_grad()
                    # train the gen and dis in different iteration count
                    it_alter_period = iteration % (cfg.gen_iter + cfg.dis_iter)
                    if it_alter_period >= cfg.gen_iter:
                        freeze_pretrain(yolact_net, freeze=True)
                        freeze_pretrain(net, freeze=True)
                        freeze_pretrain(dis_net, freeze=False)
                        if it_alter_period == (cfg.gen_iter+1):
                            print('--- Generator     freeze   ---')
                            print('--- Discriminator training ---')

                        # ----- Discriminator part -----
                        # seg_list  is the prediction mask
                        # can be regarded as generated images from YOLACT
                        # pred_list is the prediction label
                        # seg_list  dim: list of (138,138,instances)
                        # pred_list dim: list of (instances)
                        losses, seg_list, pred_list = net(datum)
                        seg_clas, mask_clas, b, seg_size = seg_mask_clas(seg_list, pred_list, datum)
                        
                        # input image size is [b, 3, 550, 550]
                        # downsample to       [b, 3, seg_h, seg_w]
                        image    = interpolate(torch.stack(datum[0]), size = seg_size, 
                                                    mode='bilinear',align_corners=False)

                        # Because in the discriminator training, we do not 
                        # want the gradient flow back to the generator part
                        # we detach seg_clas (mask_clas come the data, does not have grad)
                        seg_input   = seg_clas.clone().detach()
                        output_pred = dis_net(img = image, seg = seg_input)
                        output_grou = dis_net(img = image, seg = mask_clas)

                        if iteration % (cfg.gen_iter + cfg.dis_iter) == 0:
                            print(f'Probability of fake is fake: {output_pred.mean().item():.2f}')
                            print(f'Probability of real is real: {output_grou.mean().item():.2f}')

                        # 0 for Fake/Generated
                        # 1 for True/Ground Truth
                        fake_label = torch.zeros(b)
                        real_label = torch.ones(b)

                        # Advice of practical implementation 
                        # from https://arxiv.org/abs/1611.08408
                        # loss_pred = -criterion_dis(output_pred,target=real_label)
                        loss_pred = criterion_dis(output_pred,target=fake_label)
                        loss_grou = criterion_dis(output_grou,target=real_label)
                        loss_dis  = loss_pred + loss_grou
                        
                        # TODO: Grid Search this one
                        lambda_dis = cfg.lambda_dis
                        loss_dis   = lambda_dis*loss_dis
                        # Backprop of the discriminator
                        loss_dis.backward()
                        optimizer_dis.step()

                    else:
                        freeze_pretrain(yolact_net, freeze=False)
                        freeze_pretrain(net, freeze=False)
                        freeze_pretrain(dis_net, freeze=True)
                        if it_alter_period == 0:
                            print('--- Generator     training ---')
                            print('--- Discriminator freeze   ---')
                        
                        # ----- Generator part -----
                        net.zero_grad()
                        losses, seg_list, pred_list = net(datum)
                        seg_clas, mask_clas, b, seg_size = seg_mask_clas(seg_list, pred_list, datum)
                        image    = interpolate(torch.stack(datum[0]), size = seg_size, 
                                                    mode='bilinear',align_corners=False)
                        # Perform forward pass of all-fake batch through D
                        # NOTE that seg_clas CANNOT detach, in order to flow the 
                        # gradient back to the generator
                        output = dis_net(img = image, seg = seg_clas)
                        # Since the log(1-D(G(x))) not provide sufficient gradients
                        # We want log(D(G(x)) instead, this can be achieve by
                        # use the real_label as target.
                        # This step is crucial for the information of discriminator
                        # to go into the generator.
                        # Calculate G's loss based on this output
                        real_label = torch.ones(b)
                        loss_gen = criterion_dis(output,target=real_label)
                        if not it_alter_period >= cfg.gen_iter:
                            # since the dis is already freeze, the gradients will only
                            # record the YOLACT
                            loss_gen.backward()
                            # Do this to free up vram even if loss is not finite
                            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                            loss = sum([losses[k] for k in losses])
                            if torch.isfinite(loss).item():
                                # since the optimizer_gen is for YOLACT only
                                # only the gen will be updated
                                optimizer_gen.step()       
                    
                else:
                    # ====== Normal YOLACT Train ======
                    # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                    losses = net(datum)
                    losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                    loss = sum([losses[k] for k in losses])
                    # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                    # all_loss = sum([v.mean() for v in losses.values()])

                    # Backprop
                    loss.backward() # Do this to free up vram even if loss is not finite
                    if torch.isfinite(loss).item():
                        optimizer_gen.step()                    
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    if cfg.pred_seg and (it_alter_period >= cfg.gen_iter):
                        print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || Dis: %.2f || ETA: %s || timer: %.3f')
                                % tuple([epoch, iteration] + loss_labels + [total, loss_dis, eta_str, elapsed]), flush=True)
                    else:
                        print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f ||  ETA: %s || timer: %.3f')
                                % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                    # Loss Key:
                    #  - B: Box Localization Loss
                    #  - C: Class Confidence Loss
                    #  - M: Mask Loss
                    #  - P: Prototype Loss
                    #  - D: Coefficient Diversity Loss
                    #  - E: Class Existence Loss
                    #  - S: Semantic Segmentation Loss
                    #  - T: Total loss
                    #  -Dis:Discriminator Loss

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            
            # This is done per epoch
            if args.validation_epoch > 0:
                # NOTE: Validation loss
                # if cfg.pred_seg:
                #     net.eval()
                #     dis_net.eval()
                #     cfg.gan_eval = True
                #     with torch.no_grad():
                #         for datum in tqdm(val_loader, desc='GAN Validation'):
                #             losses, seg_list, pred_list = net(datum)
                #             losses, seg_list, pred_list = net(datum)
                #             # TODO: warp below as a function
                #             seg_list = [v.permute(2,1,0).contiguous() for v in seg_list]
                #             b = len(seg_list) # batch size
                #             _, seg_h, seg_w = seg_list[0].size()

                #             seg_clas    = torch.zeros(b, cfg.num_classes-1, seg_h, seg_w)
                #             mask_clas   = torch.zeros(b, cfg.num_classes-1, seg_h, seg_w)
                #             target_list = [target for target in datum[1][0]]
                #             mask_list   = [interpolate(mask.unsqueeze(0), size = (seg_h,seg_w),mode='bilinear', \
                #                             align_corners=False).squeeze() for mask in datum[1][1]]

                #             for idx in range(b):
                #                 for i, (pred, i_target) in enumerate(zip(pred_list[idx], target_list[idx])):
                #                     seg_clas[idx, pred, ...]                 += seg_list[idx][i,...]
                #                     mask_clas[idx, i_target[-1].long(), ...] += mask_list[idx][i,...]
                               
                #             seg_clas = torch.clamp(seg_clas, 0, 1)
                #             image    = interpolate(torch.stack(datum[0]), size = (seg_h,seg_w), 
                #                                         mode='bilinear',align_corners=False)
                #             real_label  = torch.ones(b)
                #             output_pred = dis_net(img = image, seg = seg_clas)
                #             output_grou = dis_net(img = image, seg = mask_clas)
                #             loss_pred   = -criterion_dis(output_pred,target=real_label)
                #             loss_grou   =  criterion_dis(output_grou,target=real_label)
                #             loss_dis    = loss_pred + loss_grou
                #         losses = { k: (v).mean() for k,v in losses.items() }
                #         loss = sum([losses[k] for k in losses])
                #         val_loss = loss - cfg.lambda_dis*loss_dis
                #         schedule_dis.step(loss_dis)
                #         lr = [group['lr'] for group in optimizer_dis.param_groups]
                #         print(f'Discriminator lr: {lr[0]}')
                #     net.train()
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    cfg.gan_eval = False
                    dis_net.eval()
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer_gen, new_lr):
    for param_group in optimizer_gen.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            # NOTE: 
            # Since the allocation is for training
            # not for GAN validation
            try:
                for _ in range(alloc):
                    images[cur_idx]  = gradinator(images[cur_idx].to(device))
                    targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                    masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                    cur_idx += 1
            except:
                print('GAN spile done')
                break

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

# compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print('\n')
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()

