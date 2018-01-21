import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rcn.vgg import VGGGRU
from torchvision.models.vgg import *
import torchvision.transforms as transforms
from data import UCF101Folder, FixedFrameSelector
from utils import *
import time
import os


# gloabl setting
use_gpu = torch.cuda.is_available()
use_multi_gpu = torch.cuda.device_count() > 1
mode = 'train'
batch_size = 50
seq_len = 8
epochs = 60
print_freq = 10
try_resume = True
latest_check = 'checkpoint/vgg11bn47_latest.pth.tar'
best_check = 'checkpoint/vgg11bn47_best.pth.tar'


# model
base_model = vgg11_bn(pretrained=False)
modify_layers = [4, 7]
model = VGGGRU(base_model, modify_layers, 101)
if use_multi_gpu:
    model._rcn = nn.DataParallel(model._rcn, dim=1)


# data loader
selector = FixedFrameSelector(seq_len)
data_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
dataset = UCF101Folder('/Users/filick/projects/video/data/UCF-101',
                       '/Users/filick/projects/video/data/ucfTrainTestlist',
                       mode, selector, transform=data_trans)
dataloader = DataLoader(dataset, batch_size, True, num_workers=8, pin_memory=use_gpu)


# optimizer
weight_decay = 0
lr = 0.001
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
lr_scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)


# resume
best_prec1 = 0
start_epoch = 0
if try_resume:
    if os.path.isfile(latest_check):
        print("=> loading checkpoint '{}'".format(latest_check))
        checkpoint = torch.load(latest_check)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(latest_check, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(latest_check))



# Train
if use_gpu:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    model = model.cuda()
    criterion = criterion.cuda()

for epoch in range(start_epoch, epochs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    if mode == 'train':
        model.train()
    else:
        model.eval()

    end = time.time()

    for i, (inp, target) in enumerate(dataloader):
        data_time.update(time.time() - end)
        if use_gpu:
            inp = inp.cuda(async=True)
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inp.permute(1,0,2,3,4), volatile=(mode!='train'))
        target_var = torch.autograd.Variable(target, volatile=(mode!='train'))

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
                    epoch, i, len(dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
            .format(top1=top1, top3=top3))

    lr_scheduler.step(losses.avg)

    # remember best prec@1 and save checkpoint
    is_best = prec1[0] > best_prec1
    best_prec1 = max(prec1[0], best_prec1)
    save_checkpoint(latest_check, best_check,
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    }, is_best)
