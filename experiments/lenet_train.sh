python -u train.py \
       --device cuda \
       --model lenet \
       --dataset cifar100 \
       --save_dir ./models/ \
       --num_classes 100 \
       --is_pretrained False \
       --is_transfered False \
       --epochs 30 \
       --batch_size 64 \
       --learning_rate 0.001 \
       --momentum 0.9 \
       --criterion crossentropy \
       --optimizer sgd