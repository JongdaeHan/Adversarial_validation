python -u test.py \
       --device cuda \
       --model vgg16 \
       --dataset cifar100 \
       --save_dir ./models/VGG16/ \
       --num_classes 1000 \
       --batch_size 4 \
       --saved_model 30