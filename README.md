# experimental codes for adversalial validation

if you need modified dataset(~17GB), please contact to elvenwhite@smu.ac.kr.

## train

`python train.py --model lenet --save_dir ./models/LeNet/`

## test

`python test.py --model lenet --save_dir ./models/LeNet/`

---

## options

### 학습 & 평가 공통(base_options)

* --device : GPU 사용 여부

* --model : 사용할 모델(vgg16, lenet, resnet18)

* --save_dir : 학습이 완료된 모델이 저장되는 경로

* --num_classes : 데이터 셋의 클래스 수

### 학습

* --is_pretrained : torch.utils.model_zoo를 통해 학습이 이루어진 모델을 사용할 것인지

* --is_transfered : 학습된 모델을 가져와 전이학습을 진행할 것인지

* --epochs : 학습할 에폭 횟수

* --batch_size : 학습에 대한 배치 사이즈(평가 따로 사용)

* --learning_rate

* --momentum

* --criterion : 학습에 사용할 손실 함수(crossentropy, mseloss)

* --optimizer : 학습에 사용할 최적화 함수(sgd, adam)

### 평가

* --epochs : 평가를 진행할 에폭 횟수

* --batch_size : 평가에 대한 배치 사이즈

* --saved_model : 불러올 모델의 숫자, 매 에폭마다 모델을 저장함

