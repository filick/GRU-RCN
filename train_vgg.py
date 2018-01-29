import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rcn.vgg import VGGGRU
from torchvision.models.vgg import *
import torchvision.transforms as transforms
from data import UCF101Folder
from data.selector import *
from data.transforms import ScaleJittering
from utils import *
import time
import os


# gloabl setting
use_gpu = torch.cuda.is_available()
use_multi_gpu = torch.cuda.device_count() > 1
mode = 'train'
batch_size = 210
test_batch_size = 21
test_each_epochs = 10
seq_len = 5
epochs = 100
print_freq = 10
try_resume = True
latest_check = 'checkpoint/vgg11_357_latest.pth.tar'
best_check = 'checkpoint/vgg11_357_best.pth.tar'


# model
base_model = vgg11(pretrained=True)
modify_layers = [3,5,7]
model = VGGGRU(base_model, modify_layers, 101, only_last=False, dropout=0.5)
if use_multi_gpu:
    model = nn.DataParallel(model)


# data loader
train_selector = TSNSelector(seq_len)
train_traintrans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(),
        ScaleJittering(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
train_dataset = UCF101Folder('/home/member/fuwang/data/UCF101/UCF-101',
                             '/home/member/fuwang/data/UCF101/ucfTrainTestlist',
                             'train', train_selector, transform=train_traintrans)
train_loader = DataLoader(train_dataset, batch_size, True, num_workers=8, pin_memory=use_gpu)

trans_crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_selector = TSNSelector(seq_len, random=False)
train_traintrans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([trans_crop(crop) for crop in crops]))
    ])
test_dataset = UCF101Folder('/home/member/fuwang/data/UCF101/UCF-101',
                            '/home/member/fuwang/data/UCF101/ucfTrainTestlist',
                            'test', test_selector, transform=train_traintrans)
test_loader = DataLoader(test_dataset, test_batch_size, False, num_workers=8, pin_memory=use_gpu)


# optimizer
weight_decay = 0
lr = 0.001
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
#lr_scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
lr_scheduler = lrs.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))


# to cuda
if use_gpu:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    model = model.cuda()
    criterion = criterion.cuda()


# resume
best_prec1 = 0
start_epoch = 0
if try_resume:
    path = latest_check if mode == 'train' else best_check
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        lr = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(path, checkpoint['epoch']))
        if mode == 'train':
            '''
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            '''
            lr_scheduler.step(start_epoch)
    else:
        print("=> no checkpoint found at '{}'".format(path))


# OK， let's begin
repeats = None
if mode == 'train':
    repeats = range(start_epoch, epochs)
else:
    repeats = range(start_epoch, start_epoch + 1)

for epoch in repeats:

    # train
    if mode == 'train':
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        lr_scheduler.step()

        model.train()
        end = time.time()
        for i, (inp, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if use_gpu:
                inp = inp.cuda(async=True)
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(inp, volatile=False)
            target_var = torch.autograd.Variable(target, volatile=False)

            # compute output
            optimizer.zero_grad()
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data[0], inp.size(0))
            top1.update(prec1[0], inp.size(0))
            top3.update(prec3[0], inp.size(0))

            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top3=top3))

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
                .format(top1=top1, top3=top3))

        #lr_scheduler.step(losses.avg)

    # test
    if (mode == 'test') or ((epoch + 1) % test_each_epochs == 0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        model.eval()
        end = time.time()
        for i, (inp, target) in enumerate(test_loader):
            data_time.update(time.time() - end)
            if use_gpu:
                inp = inp.cuda(async=True)
                target = target.cuda(async=True)
            bs, seqs, ncrops, c, h, w = inp.size()
            inp = inp.permute(0, 2, 1, 3, 4, 5).contiguous()
            input_var = torch.autograd.Variable(inp, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            output = model(input_var.view(-1, seqs, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            top1.update(prec1[0], inp.size(0))
            top3.update(prec3[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if mode == 'test' and i % print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                        i, len(test_loader), batch_time=batch_time,
                        data_time=data_time, top1=top1, top3=top3))

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
                .format(top1=top1, top3=top3))

    # remember best prec@1 and save checkpoint
    if mode == 'train':
        is_best = top1.avg > best_prec1
        best_prec1 = max(top1.avg, best_prec1)
        save_checkpoint(latest_check, best_check,
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1,
                            'lr': optimizer.param_groups[0]['lr']
                        }, is_best)