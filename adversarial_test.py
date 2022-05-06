import torch

from torchvision import datasets, transforms
import torchvision.models as models

import os
from tqdm import tqdm

from networks.LeNet import LeNet
from utils.Logger import Logger
from adversarial_examples.Mixed_Dataset import Mixed_Dataset
from adversarial_examples.Adversary_Dataset import Adversary_Dataset
from GTSRB_Reader import GTSRB_Reader
from datasets.GTSRB_Dataset import GTSRB_Dataset

# ==========================================
# Test
# ==========================================

USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device('cuda:1' if USE_CUDA else 'cpu')
print('Test will be activated in', DEVICE)

NETWORK = 'lenet' # lenet, resnet18, vgg16
SAVE_DIR = './models/LeNet/'
DATASET = 'GTSRB' # cifar10, adversary_cifar10, mix_cifar10, ... 학습에 사용할 데이터
NUM_CLASSES = 43

BATCH_SIZE = 4 # ResNet18을 위해 고정
TRAINED_DATA = 'adversary_0.1_8.2_GTSRB_90_26' # cifar10, adversary_cifar10, mix_cifar10, ... 학습에 사용된 데이터
SAVED_MODEL = SAVE_DIR + NETWORK + '_' + TRAINED_DATA + '_net.pth'

logger = Logger(code_type='test', model=NETWORK, dataset=DATASET)
hyper_parameter_infos = """\
- dataset : %s 
- batch_size : %s 
- saved_model : %s \
""" % (DATASET, BATCH_SIZE, SAVED_MODEL)
logs = ''

"""
# ==========================================
# Clean data
# ==========================================
print('Clean data loading...')

clean_test_dataset = datasets.CIFAR100(
    root='./datasets/cifar100/',
    train=False,
    # download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
)

test_loader = torch.utils.data.DataLoader(dataset=clean_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
"""
# """ GTSRB

g = GTSRB_Reader()

clean_test_data_examples, clean_test_label_examples = g.readTestTrafficSigns('./datasets/GTSRB/test/Images/')

tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = GTSRB_Dataset(clean_test_data_examples, clean_test_label_examples, tf)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
# """


# ==========================================
# Adversary data
# ==========================================
# if DATASET == 'adversary_cifar10':
"""
print('Adversary data loading...')
ADV_TEST_DIR = './adversarial_examples/GTSRB/test/' # change the path

# label
classes = os.listdir(ADV_TEST_DIR + '0.1/') # change the epsilon 0, 0.05, 0.1
adv_test_list = []

for _class in classes:
    # print(_class)

    test_images = os.listdir(ADV_TEST_DIR + '0.1/' + _class + '/')
    for image in tqdm(test_images):
        adv_test_list.append(ADV_TEST_DIR + '0.1/' + _class + '/' + image)

adv_test_dataset = Adversary_Dataset(file_list=adv_test_list, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=adv_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
"""

# ==========================================
# Mix data(clean + adversary)
# ==========================================
# if DATASET == 'mix_cifar10':
"""
# clean
# clean_test_data_examples = []
# clean_test_label_examples = []

# clean_test_dataset = datasets.CIFAR10(
#     root='./datasets/cifar10/',
#     train=False,
#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
#     ])
# )

# # 배치 사이즈 바꾸지 말 것 1로 유지
# clean_test_loader = torch.utils.data.DataLoader(dataset=clean_test_dataset, batch_size=1, shuffle=False)

# for (data, label) in tqdm(clean_test_loader):
#     clean_test_data_examples.append(data[0]) # 배치 정보를 제거
#     clean_test_label_examples.append(label[0]) # 배치 정보를 제거

# # 비율에 따라 조정 clean:adversary = [9:1, 8:2, 7:3, 6:4]
# clean_test_data_examples = clean_test_data_examples[:int(len(clean_test_data_examples) * 0.8)]
# clean_test_label_examples = clean_test_label_examples[:int(len(clean_test_label_examples) * 0.8)]

# # adversary
# ADV_TEST_DIR = './adversarial_examples/CIFAR10/test/'

# # label
# classes = os.listdir(ADV_TEST_DIR + '0.05/')

# adv_test_list = []

# for _class in classes:
#     test_images = os.listdir(ADV_TEST_DIR + '0.05/' + _class + '/')
#     for image in tqdm(test_images):
#         adv_test_list.append(ADV_TEST_DIR + '0.05/' + _class + '/' + image)

# adv_test_dataset = Adversary_Dataset(file_list=adv_test_list, transform=transforms.ToTensor())
# adv_test_loader = torch.utils.data.DataLoader( # 배치 사이즈 바꾸지 말 것 1로 유지
#     dataset=adv_test_dataset, batch_size=1, shuffle=True)

# adv_test_data_examples = []
# adv_test_label_examples = []

# for (data, label) in tqdm(adv_test_loader):
#     adv_test_data_examples.append(data[0]) # 배치 정보를 제거
#     adv_test_label_examples.append(label[0]) # 배치 정보를 제거

# # 비율에 따라 조정 clean:adversary = [9:1, 8:2, 7:3, 6:4]
# adv_test_data_examples = adv_test_data_examples[:int(len(adv_test_data_examples) * 0.2)]
# adv_test_label_examples = adv_test_label_examples[:int(len(adv_test_label_examples) * 0.2)]

# # clean 8_000, adv 2_000
# mixed_test_data_examples = clean_test_data_examples + adv_test_data_examples
# mixed_test_label_examples = clean_test_label_examples + adv_test_label_examples
# mixed_test_dataset = Mixed_Dataset(mixed_test_data_examples, mixed_test_label_examples)

# test_loader = torch.utils.data.DataLoader(mixed_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
"""

print(DATASET + ' will be used')

# ==========================================
# Model settings
# ==========================================
if NETWORK == 'lenet':
    model = LeNet(num_classes=NUM_CLASSES)
if NETWORK == 'resnet18':
    model = models.resnet18(pretrained=False, num_classes=NUM_CLASSES)
if NETWORK == 'vgg16':
    model = models.vgg16(pretrained=False, num_classes=NUM_CLASSES)

print(NETWORK + ' will be used')
model.load_state_dict(torch.load(SAVED_MODEL))
model.to(DEVICE)

# ==========================================
# Testing
# ==========================================

all_labels = []
all_preds = []

with torch.no_grad():
    for (data, target) in tqdm(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        target_list = target.tolist()
        predicted_list = predicted.tolist()
        for target_item, predicted_item in zip(target_list, predicted_list):
            all_labels.append(target_item)
            all_preds.append(predicted_item)

# # confusion matrix를 활용한 evaluating 구현
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# all_preds = all_preds.argmax(dim=1).tolist()
target_names = [str(i) for i in range(NUM_CLASSES)]

logs += "Confusion matrix\n"
logs += str(confusion_matrix(all_labels, all_preds)) + ('\n' * 2) # ndarray로 반환, 이를 문자열로 바꿔서 넣으면 전체를 출력할 수 있을 것 같다. -> 찬찬히 생각해보기
logs += classification_report(all_labels, all_preds, target_names=target_names) # 얘는 그냥 str이라 붙여넣기만 하면 된다.
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=target_names))

logger.create_txt(hyper_parameter_infos, logs)