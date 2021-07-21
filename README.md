# COSAL
Source code for paper "accurate and scalable graph neural networks for billion-scale graphs"

## Requirements

## Usage
'''
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file citeseer --topk 5 --cap_hidden_size 32 --rouit 1 --lr 0.011 --weight_decay 0.02 --dropout 0.6 --batch_size 64 --threshold 0.5 --reg_coef 0.039 --beta 0.32 --eps 0.001 --alpha 0.26
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file cora --topk 10 --alpha 0.33 --cap_hidden_size 32 --rouit 2 --lr 0.012  --weight_decay 0.005 --dropout 0.7 --batch_size 16 --threshold 0.87 --reg_coef 0.0004 --beta 0.76 --eps 0.001 --eval_step 10
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file pubmed --topk 256 --alpha 0.05 --cap_hidden_size 16 --rouit 4 --lr 0.002 --weight_decay 0.0001 --dropout 0.5 --batch_size 128 --threshold 0.3 --reg_coef 0.00015 --beta 0.68 --eps 1e-05
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file flickr.npz --eps 1e-4 --topk 128 --cap_hidden_size 32  --rouit 2 --lr 0.01 --weight_decay 0.0001 --dropout 0.0 --batch_size 512 --threshold 0.5 --reg_coef 0.0001 --beta 0.5  --alpha 0.26 --hidden_size 64
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file amazon2M.npz --eps 1e-4    --cap_hidden_size 16 --hidden_size 64 --lr 0.015 --weight_decay 0.00015 --rouit 2 --dropout 0.1 --topk 256 --alpha 0.26 --beta 0.83 --reg_coef 0.00012  --threshold 0.1
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file mag_coarse.npz --eps 1e-3    --cap_hidden_size 10 --hidden_size 40 --lr 5e-3 --weight_decay 1e-2 --rouit 1 --dropout 0.1 --topk 10 --alpha 0.25 --beta 0.45 --reg_coef 0.003  --threshold 0.8
$ CUDA_VISIBLE_DEVICES=1 python -u main.py --data_file ogbn-papers100M  --eps 1e-3    --cap_hidden_size 10 --hidden_size 40 --lr 5e-3 --weight_decay 0.0001 --rouit 2 --dropout 0.1 --topk 128 --alpha 0.28 --beta 0.5 --reg_coef 0.0001  --threshold 0.5  --init_mtd 0 --attention 1
'''

## Ackowledgement
This repo is modified from [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch), and [DisenGNNs](https://jianxinma.github.io/assets/DisenGCN-py3.zip)
