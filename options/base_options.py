import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--device', type=str, default='cuda', help='어떤 기기를 학습 또는 평가에 활용할 것인지')
        self.parser.add_argument('--model', type=str, default='lenet', help='어떤 모델을 학습 또는 평가에 활용할 것인지(lenet, vgg16, resnet18)')
        self.parser.add_argument('--dataset', type=str, default='cifar10', help='학습에 사용할 데이터 셋(cifar10, cifar100, gtsrb)')
        self.parser.add_argument('--save_dir', type=str, default='./models/VGG16/', help='학습이 완료된 모델이 저장되는 경로')
        self.parser.add_argument('--num_classes', type=int, default=10, help='데이터 셋의 클래스 수')

        # logger 기록 이후 사용될 코드
        # self.parser.add_argument('--use_logger', type=bool, default=True, help='로그 기록을 할 것인지 아닌지')
