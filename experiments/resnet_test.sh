python -u test.py \
       --device cuda \
       --model resnet18 \
       --dataset cifar100 \
       --save_dir ./models/ResNet18/ \
       --num_classes 1000 \
       --batch_size 4 \
       --saved_model 30