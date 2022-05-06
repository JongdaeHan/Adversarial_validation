import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, utils
import torchvision.models as models

import os
from tqdm import tqdm

from options.train_options import TrainOptions
from networks.LeNet import LeNet
from utils.Logger import Logger

USE_CUDA = torch.cuda.is_available()

if __name__ == '__main__':
    opt = TrainOptions().parser.parse_args()

    DEVICE = torch.device(
        'cuda' if USE_CUDA and opt.device == 'cuda' else 'cpu')
    print('Training will be activated in', DEVICE)

    NETWORK = opt.model
    DATASET = opt.dataset
    SAVE_DIR = opt.save_dir
    NUM_CLASSES = opt.num_classes

    IS_PRETRAINED = opt.is_pretrained
    IS_TRANSFERED = opt.is_transfered

    EPOCHS = opt.epochs
    BATCH_SIZE = opt.batch_size
    LEARNING_RATE = opt.learning_rate
    MOMENTUM = opt.momentum

    CRITERION = opt.criterion
    OPTIMIZER = opt.optimizer

    logger = Logger(model=NETWORK, dataset=DATASET)
    hyper_parameter_infos = """\
    - is_pretrained : %s
    - is_transfered : %s
    - epochs : %s
    - batch_size : %s
    - learning_rate : %s
    - momentum : %s
    - criterion : %s
    - optimizer : %s\
    """ % (IS_PRETRAINED, IS_TRANSFERED, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, CRITERION, OPTIMIZER)
    logs = ''

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    if DATASET == 'cifar10':
        training_dataset = datasets.CIFAR10(
            root='./datasets/cifar10/',
            train=True,
            # download=True,
            transform=transform
        )
    if DATASET == 'cifar100':
        training_dataset = datasets.CIFAR100(
            root='./datasets/cifar100/',
            train=True,
            # download=True,
            transform=transform
        )
    print(DATASET + ' will be used')

    train_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # models : lenet5, resnet18, vgg16
    if NETWORK == 'lenet':
        model = LeNet(num_classes=NUM_CLASSES)
    if NETWORK == 'resnet18':
        model = models.resnet18(pretrained=IS_PRETRAINED,
                                num_classes=NUM_CLASSES)
    if NETWORK == 'vgg16':
        model = models.vgg16(pretrained=IS_PRETRAINED, num_classes=NUM_CLASSES)

    print(NETWORK + ' will be used')
    print(hyper_parameter_infos)
    model.to(DEVICE)  # 모델을 넣어주지 않을 경우 에러 발생
    print(model)

    # if CRITERION == 'crossentropy':
    #     criterion = nn.CrossEntropyLoss()
    # if CRITERION == 'mseloss':
    #     criterion = nn.MSELoss()

    # if OPTIMIZER == 'sgd':
    #     optimizer = optim.SGD(model.parameters(),
    #                           lr=LEARNING_RATE, momentum=MOMENTUM)
    # if OPTIMIZER == 'adam':
    #     optimizer = optim.Adam(
    #         model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # model.train()

    # for epoch in range(EPOCHS):
    #     # print('%d epoch started' % epoch)
    #     running_loss = .0
    #     for i, (data, target) in enumerate(train_loader):
    #         # print(data)
    #         # print(data.shape)
    #         # print(target)
    #         # print(target.shape)
    #         # break
    #         data, target = data.to(DEVICE), target.to(DEVICE)
    #         optimizer.zero_grad()

    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2_000))
    #     # torch.save(model.state_dict(), SAVE_DIR + NETWORK +
    #     #            '_' + str(epoch + 1) + '_' + DATASET + '_net.pth')
    #     # print('%d epoch loss : %.3f' %
    #     #       (epoch + 1, running_loss / len(train_loader)))
    #     logs += '%d epoch loss : %.3f' % (epoch + 1,
    #                                       running_loss / len(train_loader)) + '\n'
    #     # break
    # # use last epoch only
    # torch.save(model.state_dict(), SAVE_DIR + NETWORK + '_' + DATASET + '_net.pth')
    # logger.create_txt(hyper_parameter_infos, logs)
