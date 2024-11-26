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
import models.vision_transformer_cbwc as vits_cbwc
import models.vision_transformer_rms as vits_rms
import wandb
import time
from data import CustomDataset


def main():

    args = get_args()

    datapath = args.data_path
    dataset = datapath.split('/')[-1]

    model_name = str(args.arch) + '_' + args.m + '_' + dataset + '_e' + str(args.epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_wd' + str(args.weight_decay) 
    model_name = model_name + '_wre' + str(args.warmup_epochs) + '_wk' + str(args.workers) + '_nc' + str(args.num_classes) + '_s' + str(args.seed) + '_ps' + str(args.patch_size) 

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data loading code
    datadir = args.data_path
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ]) 
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    full_dataset = CustomDataset(root_dir=datadir, transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    full_dataset = CustomDataset(root_dir=datadir, transform=test_transform)
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, shuffle=True)

    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=args.workers, 
        pin_memory=True, shuffle=False)

    print("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    print("creating model '{}'".format(args.arch))

    if args.m == 'ori':
        model = vits.__dict__[args.arch](num_classes=args.num_classes, patch_size=args.patch_size)
    elif args.m == 'cbwc':
        model = vits_cbwc.__dict__[args.arch](num_classes=args.num_classes)
    elif args.m == 'rms':
        model = vits_rms.__dict__[args.arch](num_classes=args.num_classes)


    args.lr = args.lr * args.batch_size  / 256

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    print("Building optimizer done.")
    
    # copy model to GPU
    model.cuda()
    print(model)
    print("Building model done.")

    wandb.init(
        project="CBWC-exp",
        name=model_name,
        notes=str(args),
        config={
            "model": args.arch,
            "method": args.m,
            
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,

            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,

            "workers": args.workers,
            "method": str('origin'),


            "seed": args.seed,
        }
    )

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    dump_path = os.path.join(args.dump_path, model_name)

    if not os.path.exists(dump_path):
        # 如果路径不存在，则创建它
        os.makedirs(dump_path)
        print(f"目录 {dump_path} 已创建。")
    else:
        print(f"目录 {dump_path} 已存在。")

    print("==> Begin Training.")

    for epoch in range(args.epochs):

        # train the network for one epoch
        print("============ Starting epoch %i ... ============" % epoch)

        # train the network
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

        # save checkpoints
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
                
        torch.save(
            save_dict,
            os.path.join(dump_path, "checkpoint.pth.tar"),
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
        fp_time_batch = (fp_end - fp_begin) * 1e6
        fp_time.update(fp_time_batch)

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
        bp_time_batch = (bp_end - bp_begin) * 1e6
        bp_time.update(bp_time_batch)
        wandb.log({"fp_time":fp_time_batch, "bp_time":bp_time_batch})

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        
        if i % 100 == 0:
            progress.display(i)

        torch.cuda.synchronize()
        end = time.time()


    wandb.log({"train_loss":losses.avg, "train_acc_top1": top1.avg, "train_acc_top5": top5.avg, "train_epoch":epoch, "train_fp_avg_time": fp_time.avg, "train_bp_avg_time": bp_time.avg})

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

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output

            torch.cuda.synchronize()
            val_begin = time.time()
            output = model(images)
            torch.cuda.synchronize()
            val_end = time.time()
            val_time_batch = (val_end - val_begin)*1e6
            val_time.update(val_time_batch)
            wandb.log({"val_time":val_time_batch})

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
    wandb.log({"test_loss":losses.avg, "test_acc_top1": top1.avg, "test_acc_top5": top5.avg, "val_avg_time": val_time.avg})
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