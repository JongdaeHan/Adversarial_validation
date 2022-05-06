import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, utils
import torchvision.models as models

import os, sys
from tqdm import tqdm
import numpy as np

from options.train_options import TrainOptions
from networks.LeNet import LeNet
from utils.Logger import Logger
from adversarial_examples.Mixed_Dataset import Mixed_Dataset
from adversarial_examples.Adversary_Dataset import Adversary_Dataset
from GTSRB_Reader import GTSRB_Reader

"""
clean과 adversary가 일정 비율로 혼합된 데이터 셋을 활용하여 학습을 진행한다.
"""
# ==========================================
print('Clean data loading...')
# ==========================================
# """ CIFAR10 CIFAR100
clean_train_data_examples = []
clean_train_label_examples = []

clean_train_dataset = datasets.CIFAR10(
    root='./datasets/cifar10/',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
)

clean_train_loader = torch.utils.data.DataLoader( # 배치 사이즈 바꾸지 말 것 1로 유지
    dataset=clean_train_dataset, batch_size=1, shuffle=True)
# mixed_train_loader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=64, shuffle=True) # clean 데이터만 사용하는 경우

for (data, label) in tqdm(clean_train_loader):
    clean_train_data_examples.append(data[0]) # 배치 정보를 제거
    clean_train_label_examples.append(label[0]) # 배치 정보를 제거

# """
""" GTSRB

g = GTSRB_Reader()

clean_train_data_examples, _ = g.readTrafficSigns('./datasets/GTSRB/training/Images')

tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = Adversary_Dataset(clean_train_data_examples, tf)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

clean_train_data_examples = []
clean_train_label_examples = []
for (data, label) in tqdm(data_loader):
    clean_train_data_examples.append(data[0])
    clean_train_label_examples.append(label[0])

# clean일 때
# clean_train_dataset = Mixed_Dataset(clean_train_data_examples, clean_train_label_examples)
# mixed_train_loader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=64, shuffle=True)
"""
# # 8 : 2 비율
clean_train_data_examples = clean_train_data_examples[:int(len(clean_train_data_examples) * 0.8)]
clean_train_label_examples = clean_train_label_examples[:int(len(clean_train_label_examples) * 0.8)]
# """
# adversarial
# ==========================================
print('Adversary data loading...')
# ==========================================

tf = transforms.ToTensor()

ADV_TRAIN_DIR = './adversarial_examples/CIFAR10/training/'

# label
classes = os.listdir(ADV_TRAIN_DIR + '0.05/')

adv_train_list = []

for _class in classes:
    train_images = os.listdir(ADV_TRAIN_DIR + '0.05/' + _class + '/')
    for image in tqdm(train_images):
        adv_train_list.append(ADV_TRAIN_DIR + '0.05/' + _class + '/' + image)

adv_train_dataset = Adversary_Dataset(adv_train_list, tf)
adv_train_loader = torch.utils.data.DataLoader( # 배치 사이즈 바꾸지 말 것 1로 유지
    dataset=adv_train_dataset, batch_size=1, shuffle=True)

adv_train_data_examples = []
adv_train_label_examples = []

for (data, label) in tqdm(adv_train_loader):
    adv_train_data_examples.append(data[0]) # 배치 정보를 제거
    adv_train_label_examples.append(label[0]) # 배치 정보를 제거

adv_train_data_examples = adv_train_data_examples[:int(len(adv_train_data_examples) * 0.2)]
adv_train_label_examples = adv_train_label_examples[:int(len(adv_train_label_examples) * 0.2)]
# ==========================================
# Clean + Adversary
# ==========================================
mixed_train_data_examples = clean_train_data_examples + adv_train_data_examples
mixed_train_label_examples = clean_train_label_examples + adv_train_label_examples
print('# of used clean data : ' + str(len(clean_train_data_examples)))
print('# of used adversary data : ' + str(len(adv_train_data_examples)))
print('# of total datas : ', len(mixed_train_data_examples))
mixed_train_dataset = Mixed_Dataset(mixed_train_data_examples, mixed_train_label_examples)

mixed_train_loader = torch.utils.data.DataLoader(mixed_train_dataset, batch_size=64, shuffle=True)
# """

# ==========================================
print("Training...")
# ==========================================
USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device('cuda:1' if USE_CUDA else 'cpu') # 'cuda' -> 0번 그래픽 카드, 'cuda:1' -> 1번 그래픽 카드
print('Training will be activated in', DEVICE)

NETWORK = 'resnet18'
DATASET = '0.05_8.2_adversary_cifar10' # (epsilon)_(max-rate)_(dataset-name)
SAVE_DIR = './models/ResNet18/'
NUM_CLASSES = 10 # 10, 100, 43(GTSRB)

IS_PRETRAINED = False
IS_TRANSFERED = False

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9

TARGET_ACC = 90

CRITERION = 'crossentropy'
OPTIMIZER = 'sgd'

logger = Logger(model=NETWORK, dataset=DATASET)
hyper_parameter_infos = """\
- dataset : %s
- is_pretrained : %s
- is_transfered : %s
- epochs : %s
- batch_size : %s
- learning_rate : %s
- momentum : %s
- criterion : %s
- optimizer : %s\
""" % (DATASET, IS_PRETRAINED, IS_TRANSFERED, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, CRITERION, OPTIMIZER)
logs = ''

# models : lenet, resnet18, vgg16
if NETWORK == 'lenet':
    model = LeNet(num_classes=NUM_CLASSES)
if NETWORK == 'resnet18':
    model = models.resnet18(pretrained=IS_PRETRAINED, num_classes=NUM_CLASSES)
if NETWORK == 'vgg16':
    model = models.vgg16(pretrained=IS_PRETRAINED, num_classes=NUM_CLASSES)

print(NETWORK + ' will be used')
print(hyper_parameter_infos)
model.to(DEVICE)

if CRITERION == 'crossentropy':
    criterion = nn.CrossEntropyLoss()
if CRITERION == 'mseloss':
    criterion = nn.MSELoss()

if OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                            lr=LEARNING_RATE, momentum=MOMENTUM)
if OPTIMIZER == 'adam':
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

model.train()

for epoch in range(EPOCHS):
    running_loss = .0
    correct, total = 0, 0

    for (data, target) in mixed_train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()

        output = model(data)
        _, predicted = torch.max(output.data, 1) # for calculate acc
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total += target.size(0)
        correct += (predicted == target).sum().item()

        running_loss += loss.item()

    acc = 100 * correct / total

    print('%d epoch loss : %.3f' %
            (epoch + 1, running_loss / len(mixed_train_loader)))
    print('%d epoch acc : %.2f' % (epoch + 1, acc))
    logs += '%d epoch loss : %.3f' % (epoch + 1,
                                        running_loss / len(mixed_train_loader)) + '\n'
    logs += '%d epoch acc : %.2f' % (epoch + 1, acc) + '\n'
    logs += '=' * 25 + '\n'

    if acc >= TARGET_ACC or epoch == EPOCHS - 1: # 원하는 ACC를 초과할 시점에 학습을 종료
        torch.save(model.state_dict(), SAVE_DIR + NETWORK + '_' + DATASET + '_' + str(TARGET_ACC) + '_' + str(epoch + 1) + '_net.pth')
        logger.create_txt(hyper_parameter_infos, logs)
        break
    
    # sys.exit()