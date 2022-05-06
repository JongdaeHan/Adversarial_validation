import argparse
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

        self.parser.add_argument('--is_pretrained', type=bool, default=False,
                                 help='torch.utils.model_zoo를 통해 학습이 이루어진 모델을 사용하고 싶을 경우')
        self.parser.add_argument(
            '--is_transfered', type=bool, default=False, help='학습된 모델을 가져와 전이학습을 진행하고 싶은 경우')
        self.parser.add_argument('--epochs', type=int,
                                 default=30, help='에폭 크기')
        self.parser.add_argument(
            '--batch_size', type=int, default=64, help='배치 사이즈')
        self.parser.add_argument(
            '--learning_rate', type=float, default=0.001, help='학습에 사용할 학습율 값')
        self.parser.add_argument(
            '--momentum', type=float, default=0.9, help='학습에 사용할 모멘텀 값')
        self.parser.add_argument('--criterion', type=str, default='crossentropy',
                                 help='학습에 사용할 손실 함수(crossentropy, mseloss)')  # 추후 추가 가능
        self.parser.add_argument(
            '--optimizer', type=str, default='sgd', help='학습에 사용할 최적화 함수(sgd, adam)')
