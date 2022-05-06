import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, utils
import torchvision.models as models

from tqdm import tqdm

from options.test_options import TestOptions
from networks.LeNet import LeNet
from utils.Evaluator import Evaluator
from utils.Logger import Logger

USE_CUDA = torch.cuda.is_available()

if __name__ =='__main__':
    opt = TestOptions().parser.parse_args()

    DEVICE = torch.device('cuda' if USE_CUDA and opt.device == 'cuda' else 'cpu')
    print('Training will be activated in', DEVICE)

    NETWORK = opt.model
    DATASET = opt.dataset
    SAVE_DIR = opt.save_dir
    NUM_CLASSES = opt.num_classes

    BATCH_SIZE = opt.batch_size
    SAVED_MODEL = opt.saved_model

    logger = Logger(code_type='test', model=NETWORK, dataset=DATASET)
    hyper_parameter_infos = """\
    - batch_size : %s
    - saved_model : %s \
    """ % (BATCH_SIZE, SAVED_MODEL)
    logs = ''

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    if DATASET == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root='./datasets/cifar10/',
            train=False,
            transform=transform
        )
    if DATASET == 'cifar100':
        test_dataset = datasets.CIFAR100(
            root='./datasets/cifar100/',
            train=False,
            transform=transform
        )
    print(DATASET + ' will be used')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # models : lenet, resnet18, vgg16
    if NETWORK == 'lenet':
        model = LeNet(num_classes=NUM_CLASSES)
    if NETWORK == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=NUM_CLASSES)
    if NETWORK == 'vgg16':
        model = models.vgg16(pretrained=False, num_classes=NUM_CLASSES)

    print(NETWORK + ' will be used')
    model.load_state_dict(torch.load(SAVE_DIR + NETWORK + '_' + DATASET + '_net.pth'))
    model.to(DEVICE)

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
    logs += str(confusion_matrix(all_labels, all_preds)) + ('\n' * 2)
    logs += str(classification_report(all_labels, all_preds, target_names=target_names))
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=target_names))

    logger.create_txt(hyper_parameter_infos, logs)

    # get data details for calculate evaluate factors
    # # total = len(test_loader)

    # # predicts = [0] * NUM_CLASSES
    # # positives = [0] * NUM_CLASSES

    # # true_positives = [0] * NUM_CLASSES 
    # # false_negatives = [0] * NUM_CLASSES 

    # with torch.no_grad():
    #     for (data, target) in tqdm(test_loader):
    #         data, target = data.to(DEVICE), target.to(DEVICE)
    #         outputs = model(data)
    #         _, predicted = torch.max(outputs.data, 1)
    #         for i in range(BATCH_SIZE):
    #             positives[target[i].item()] += 1
    #             predicts[predicted[i].item()] += 1
    #             true_positives[target[i].item()] += (predicted[i] == target[i]).sum().item()
    #             false_negatives[target[i].item()] += (predicted[i] != target[i]).sum().item()

    # accuracies = [0.] * NUM_CLASSES
    # error_rates = [0.] * NUM_CLASSES
    # sensitivities = [0.] * NUM_CLASSES
    # precisions = [0.] * NUM_CLASSES
    # specificities = [0.] * NUM_CLASSES
    # false_positive_rates = [0.] * NUM_CLASSES

    # break_point = NUM_CLASSES

    # for i in range(NUM_CLASSES):
    #     true_positive = true_positives[i]
    #     false_positive = predicts[i] - true_positive
    #     true_negative = (total - positives[i]) - false_positive
    #     false_negative = false_negatives[i]
        
    #     # positive가 0일 경우에 ? -> 라벨이 많아지면 존재할 수도 있으니 일단 남겨두기
    #     try:
    #         evaluator = Evaluator(i, positives[i], total - positives[i], true_positive, false_negative, true_negative, false_positive)
    #         accuracies[i] = evaluator.get_accuracy()
    #         error_rates[i] = evaluator.get_error_rate()
    #         sensitivities[i] = evaluator.get_sensitivity()
    #         precisions[i] = evaluator.get_precision()
    #         specificities[i] = evaluator.get_specificity()
    #         false_positive_rates[i] = evaluator.get_false_positive_rate()
    #         print('-' * 30)
    #     except:
    #         print(str(i) + ' is not meaningful label')
    #         break_point = i
    #         break

    # for i in range(break_point):
    #     logs += '%d \'s Results' % i + ('-' * 30) + '\n'
    #     logs += 'accuracy : %.2f' % accuracies[i] + '%\n'
    #     logs += 'error rate : %.2f' % error_rates[i] + '%\n'
    #     logs += 'sensitivity : %.2f' % sensitivities[i] + '%\n'
    #     logs += 'precision : %.2f' % precisions[i] + '%\n'
    #     logs += 'specificity : %.2f' % specificities[i] + '%\n'
    #     logs += 'false positive rate : %.2f' % false_positive_rates[i] + '%\n'

    # logger.create_txt(hyper_parameter_infos, logs)