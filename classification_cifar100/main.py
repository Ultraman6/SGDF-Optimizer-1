"""Train CIFAR100 with PyTorch."""
from __future__ import print_function

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
import argparse
import time
import csv
from models import ResNet34, DenseNet121,vgg11
from torch.optim import Adam, SGD,RAdam,AdamW
from optimizers import MSVAG
from SGDF import SGDF



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--total_epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vgg'])
    parser.add_argument('--optim', default='adam', type=str, help='optimizer',
                        choices=['sgdf','sgd', 'adam', 'radam', 'adamw', 'msvag'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate decay')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2') 
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')     
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_ckpt_name(model='vgg', optimizer='sgd', lr=0.001, momentum=0.9,
                  beta1=0.9, beta2=0.999,eps=1e-8, weight_decay=5e-4,
                  reset = False, run = 0, weight_decouple = False, rectify = False):
    name = {
        'sgdf': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'sgd': 'lr{}-momentum{}-wdecay{}-run{}'.format(lr, momentum,weight_decay, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'radam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'msvag': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
    }[optimizer]
    return '{}-{}-{}-reset{}'.format(model, optimizer, name, str(reset))


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'vgg':vgg11,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgdf':
        return SGDF(model_params, args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim =='sgd':
        return SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                           eps=args.eps)
    elif args.optim == 'radam':
        return RAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    else:
        print('Optimizer not found')


def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    print('train loss %.3f' % loss.item())
    return {'loss': train_loss / len(data_loader), 'accuracy': accuracy}

def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)
    print('test loss %.3f'% loss.item())
    return {'loss': test_loss / len(data_loader), 'accuracy': accuracy}

def adjust_learning_rate(optimizer, epoch, step_size=50, gamma=0.1, reset = False):
    for param_group in optimizer.param_groups:
        if epoch % step_size==0 and epoch>0:
            param_group['lr'] *= gamma

    if  epoch % step_size==0 and epoch>0 and reset:
        optimizer.reset()


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              momentum=args.momentum, eps=args.eps,
                              beta1=args.beta1, beta2=args.beta2, 
                              reset=args.reset, run=args.run,
                              weight_decay=args.weight_decay)
    print('ckpt_name')
    
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']  

        curve = os.path.join('curve', ckpt_name)
        curve = torch.load(curve)
        train_accuracies = curve['train_acc']
        test_accuracies = curve['test_acc']
        train_losses = curve['train_loss']
        test_losses = curve['test_loss']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_accuracies = []
        test_accuracies = []
        train_losses = []
        test_losses = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        adjust_learning_rate(optimizer, epoch, step_size=args.decay_epoch, gamma=args.lr_gamma, reset = args.reset)
        train_metrics = train(net, epoch, device, train_loader, optimizer, criterion, args)
        test_metrics = test(net, device, test_loader, criterion)
        train_loss = train_metrics['loss']
        train_acc = train_metrics['accuracy']
        test_loss = test_metrics['loss']
        test_acc = test_metrics['accuracy']
        end = time.time()
        print('Time: {}'.format(end - start))

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
       

        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                    os.path.join('curve', ckpt_name))
   
            
if __name__ == '__main__':
    main()
