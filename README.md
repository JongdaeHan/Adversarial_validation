# experimental codes for adversalial validation

if you need modified dataset(~1.6GB), please contact to elvenwhite@smu.ac.kr or aggsae@gmail.com .

## train

`python train.py --model lenet --save_dir ./models/LeNet/`

## test

`python test.py --model lenet --save_dir ./models/LeNet/`

---

## options

### training & evaluation(base_options)

* --device : enables GPU 

* --model : select model(vgg16, lenet, resnet18)

* --save_dir : path to saved model

* --num_classes : number of classes belongs to training dataset

### for training only

* --is_pretrained : enable torch.utils.model_zoo pretrained model

* --is_transfered : enable transfer learning

* --epochs : maximum number of epochs for training(default 30)

* --batch_size 

* --learning_rate

* --momentum

* --criterion : loss function for training(crossentropy, mseloss)

* --optimizer : optimizer for training(sgd, adam)

### for evaluation only

* --epochs : number of epochs for evaluation

* --batch_size 

* --saved_model : ckpt number of a model to be used for evaluation

