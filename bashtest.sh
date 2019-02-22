#!/bin/bash
echo "cifar10 cnn WC+BN #originalwgan"
python main.py --dataset cifar10 --arch cnn --norm_type WC+BN --cuda --opt rmsprop --lrD 0.00005 --lrG 0.00005
# echo "cifar10 cnn OR+BN"
# INIORTH_SCALE = 0.05
# ORTH_SCALE = 0.05
# ORTH_WEI = 10000
# python main.py --dataset cifar10 --arch cnn --norm_type OR+BN --cuda --show_sv_info --opt rmsprop --lrD 0.00005 --lrG 0.00005

# echo "cifar10 cnn OR+Mani+BN"
# echo "Mani+Proj"
# INIORTH_SCALE = 1
# ORTH_SCALE = 0.02
# python main.py --dataset cifar10 --arch cnn --norm_type OR+Mani+BN --cuda --show_sv_info --opt rmsprop --lrD 0.00005 --lrG 0.00005 --use_proj

# GP
# python main.py --dataset cifar10 --arch cnn --norm_type GP --cuda --show_sv_info --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.9

# UVR
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --arch cnn --norm_type UVR --cuda --show_sv_info --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.9 --orthwei 10000

# DC GAN
python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type WC+BN --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.9
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type WC+BN --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type none --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type WC --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type BN --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type WN --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type GP --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1 --gpwei 1
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type SN --show_sv_info --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1 
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type OR --show_sv_info --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1 --orthwei 10
CUDA_VISIBLE_DEVICES=0 python main.py --loss dcgan --dataset cifar10 --arch cnn --norm_type UVR --show_sv_info --cuda --opt adam --lrD 0.0001 --lrG 0.0001 --beta1 0.5 --beta2 0.999 --Diters 1 --orthwei 0.1
