'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from tqdm import tqdm

# init
Train_acc = 0.0
PublicTest_acc = 0.0
PrivateTest_acc = 0.0
best_PublicTest_acc = 0.0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc, PublicTest_acc, PrivateTest_acc
    net.train()
    train_loss = 0.0
    correct = 0.0
    total = 0
    batch_idx = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))
    try:
        with tqdm(trainloader) as t:
            for inputs, targets in t:
                batch_idx += 1
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()

                # inputs, targets = Variable(inputs), Variable(targets)
                with torch.no_grad():
                    inputs = torch.tensor(inputs)
                    targets = torch.tensor(targets)
                    # print(inputs.shape)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                utils.clip_gradient(optimizer, 0.1)
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                # utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                tqdm.write("Loss: %.3f | Acc: %.3f %%(%d/%d)" % (train_loss/batch_idx, 100.0*float(correct)/total, correct, total))
                # print(100.0*float(correct.numpy())/total, correct.numpy(), total)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    Train_acc = 100.*correct/total
    state = {
        'net': net.state_dict() if use_cuda else net,
        'acc': PublicTest_acc,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
    state = {
        'net': net.state_dict() if use_cuda else net,
        'best_PublicTest_acc': best_PublicTest_acc,
        'best_PrivateTest_acc': PrivateTest_acc,
        'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
        'best_PrivateTest_acc_epoch': epoch,
    }
    torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))


def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0.0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)
            print(inputs.shape)

        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        # PublicTest_loss += loss.data[0]
        PublicTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100.0 * float(correct) / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.0*float(correct)/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch


def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        # PrivateTest_loss += loss.data[0]
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().numpy()

        utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PrivateTest_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
    parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
    parser.add_argument('--bs', default=32, type=int, help='learning rate')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train', type=int, default=3, help='1 to train, 2 to val, 3 to train and val')
    opt = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    best_PublicTest_acc = 0.0  # best PublicTest accuracy
    best_PublicTest_acc_epoch = 0
    best_PrivateTest_acc = 0.0  # best PrivateTest accuracy
    best_PrivateTest_acc_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    learning_rate_decay_start = 80  # 50
    learning_rate_decay_every = 5  # 5
    learning_rate_decay_rate = 0.9  # 0.9

    cut_size = 44
    total_epoch = 250

    path = os.path.join(opt.dataset + '_' + opt.model)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    data_transforms = {
        'train': transform_train,
        'test': transform_test,
        'val': transform_test
    }

    trainset = FER2013(split='Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)

    PublicTestset = FER2013(split='PublicTest', transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=True, num_workers=1)

    PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=True, num_workers=1)

    # data_dir = r"C:\Users\lenovo\Desktop\course\ML\fer_project\data"
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
    #                   ['train', 'test', 'val']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.bs, shuffle=True, num_workers=1)
    #                for x in ['train', 'test', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'val']}
    # class_names = image_datasets['train'].classes
    #
    # trainloader = dataloaders['train']
    # PrivateTestloader = dataloaders['test']
    # PublicTestloader = dataloaders['val']

    # Model
    if opt.model == 'VGG19':
        net = VGG('VGG19')
    elif opt.model == 'Resnet18':
        net = ResNet18()
    elif opt.model == 'Resnet34':
        net = ResNet34()
    elif opt.model == 'Resnet50':
        net = ResNet50()
    elif opt.model == 'Resnet101':
        net = ResNet101()
    elif opt.model == 'Resnet152':
        net = ResNet152()
        print('using Resnet152')

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))

        net.load_state_dict(checkpoint['net'])
        best_PublicTest_acc = checkpoint['best_PublicTest_acc']
        best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
        best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
        best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
        start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
    else:
        print('==> Building model..')

    if use_cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    # 1 to train, 2 to val, 3 to train and val
    if opt.train == 1:
        for epoch in range(start_epoch, total_epoch):
            train(epoch)
    if opt.train == 2:
        print('==>Val the model')
        PublicTest(start_epoch)
        PrivateTest(start_epoch)
    if opt.train == 3:
        for epoch in range(start_epoch, total_epoch):
            train(epoch)
            PublicTest(epoch)
            PrivateTest(epoch)

    print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
    print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
    print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
