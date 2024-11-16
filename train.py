
import math
import os
import builtins
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from args import get_args
import models.vision_transformer as vits
import wandb

import time


def main():



    args = get_args()

    model_name = str(args.arch) + '_e' + str(args.epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_wd' + str(args.wd) + '_wre' + str(args.warmup_epochs) + '_wk' + str(args.workers) + '_nc' + str(args.num_classes)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.seed(args.seed)

    wandb.init(
        project="CBWC-exp",
        name=model_name
        notes=str(self.cfg)
        config={
            "model": args.arch,
            
            "epochs": args.epochs,
            "batch_size": args.batch_size,

            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "warmup_epochs": args.warmup_epochs,

            "workers": args.workers,
            "method": str('origin')


            "seed": self.cfg.seed,
        }
    )

    # Data loading code
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, 
        num_workers=args.workers, pin_memory=True)
    print("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    print("creating model '{}'".format(args.arch))

    model = vits.__dict__[args.arch](num_classes=args.num_classes)

    args.lr = args.lr * args.batch_size  / 256

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    print("Building optimizer done.")
    
    # copy model to GPU
    model.cuda()
    print(model)
    print("Building model done.")

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    for epoch in range(args.epochs):

        # train the network for one epoch
        print("============ Starting epoch %i ... ============" % epoch)

        # train the network
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        wandb.log({"learning_rate": scheduler.get_lr()})

        # save checkpoints
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(
            save_dict,
            os.path.join(args.dump_path, "checkpoint.pth.tar"),
        )
        
        acc1, acc5 = validate(val_loader, model, criterion, args)

    wandb.finish()

  

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    fp_time = AverageMeter('FPTime', ':6.3f')
    bp_time = AverageMeter('BPTime', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    torch.cuda.synchronize()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        torch.cuda.synchronize()
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        torch.cuda.synchronize()
        fp_begin = time.time()
        output = model(images)
        torch.cuda.synchronize()
        fp_end = time.time()
        fp_time.update(fp_end - fp_begin)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        torch.cuda.synchronize()
        bp_begin = time.time()
        loss.backward()
        bp_end = time.time()
        bp_time.update(bp_end - bp_begin)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        
        if i % 100 == 0:
            progress.display(i)

        torch.cuda.synchronize()
        end = time.time()


    wandb.log({"train_loss":losses.avg, "train_acc_top1": top1.avg, "train_acc_top5": top5.avg, "train_epoch":epoch, "train_fp_time": fp_time.avg, "train_bp_time": bp_time.avg})

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    val_time = AverageMeter('val_time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if args.device is not None:
                images = images.cuda(args.device, non_blocking=True)
                target = target.cuda(args.device, non_blocking=True)

            # compute output

            torch.cuda.synchronize()
            val_begin = time.time()
            output = model(images)
            torch.cuda.synchronize()
            val_end = time.time()
            val_time.update(val_end - val_begin)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    wandb.log({"test_loss":losses.avg, "test_acc_top1": top1.avg, "test_acc_top5": top5.avg, "val_time": val_time.avg``})
    return top1.avg, top5.avg   

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()