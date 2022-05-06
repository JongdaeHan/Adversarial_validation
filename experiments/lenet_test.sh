python -u test.py \
       --device cuda \
       --model lenet \
       --dataset cifar100 \
       --save_dir ./models/ \
       --num_classes 100 \
       --batch_size 4 \
       --saved_model 30