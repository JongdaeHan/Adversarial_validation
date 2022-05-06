import argparse
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

        self.parser.add_argument('--batch_size', type=int, default=4, help='테스트를 위한 배치 사이즈')
        self.parser.add_argument('--saved_model', type=str, default='30', help='불러올 모델의 숫자, 매 테스트마다 모델을 저장함')
