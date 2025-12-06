python main.py --model_name basemodel --phase main --data-dir D:\Datasets\cifar10

python main.py --model_name c2ae --phase main --data-dir D:\Datasets\cifar10 --epochs-stage1 20 --epochs-stage2 50 --alpha 0.9 --lr 0.0003 --batch-size 128 --num-known-classes 6

python main.py --model_name csgrl --phase main --data-dir D:\Datasets\cifar10 --epochs-stage1 20 --epochs-stage2 50 --learn_rate 0.4 --learn_rateG 0.4 --batch-size 128 --margin 5 --s_w 0.2 --gamma 0.1 --lr_decay 0.1 --num-known-classes 6

python main.py --model_name opengan --phase main --data-dir D:\Datasets\cifar10 --epochs-stage1 20 --epochs-stage2 50 --lr 0.0003 --batch-size 128 --num-known-classes 6