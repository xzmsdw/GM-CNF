import argparse
import json
import numpy as np
import torch
import os, time
import random
from exp.exp_gm_cnf import Exp_GM_CNF
import utils.config as config

def main():
    parser = argparse.ArgumentParser(description='GM-CNF for Semi-supervised Bearing Fault Diagnosis')

    parser.add_argument('--model_id', type=str, default='GM_CNF', help='model id')
    parser.add_argument('--dataset', type=str, default='mafaulda', help='dataset name')
    parser.add_argument('--c_in_x', type=int, default=6, help='vibration channels')
    parser.add_argument('--c_in_c', type=int, default=2, help='condition channels')
    parser.add_argument('--seq_len', type=int, default=1024, help='window length')
    parser.add_argument('--cond_dim', type=int, default=256, help='condition embedding dim')
    parser.add_argument('--n_blocks', type=int, default=8, help='number of flow blocks')
    parser.add_argument('--log_path', type=str, default=None, help='log path')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size for DualResNet1D backbone')
    
    parser.add_argument('--num_known_classes', type=int, default=5, help='number of known classes (labeled)')
    parser.add_argument('--num_classes', type=int, default=7, help='total GMM components allowed')
    
    parser.add_argument('--epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='save path')
    parser.add_argument('--window_size', type=int, default=1024, help='input window size')
    parser.add_argument('--stride', type=int, default=1024, help='input window stride')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=[''], 
                        help='Classes to exclude from training (Unknown classes)')
    parser.add_argument('--training_classes', type=str, nargs='+', default=[''])
    parser.add_argument('--test_classes', type=str, nargs='+', default=[''])
    parser.add_argument('--few_shot_num', type=int, default=10, help='number of labeled samples per known class for few-shot learning')
    parser.add_argument('--seed', type=int, default=46, help='random seed')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    
    parser.add_argument('--ablation_no_cond', action='store_true', help='w/o Cond')
    parser.add_argument('--ablation_pl', action='store_true', help='w/o unsup')
    parser.add_argument('--ablation_no_sn', action='store_true', help='w/o SN')
    parser.add_argument('--ablation_learnable_var', action='store_true', help='Learnable Sigma')

    args = parser.parse_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    path = os.path.join('./logs', time.strftime('%y%m%d%H%M%S_'+args.model_id+'_'+args.dataset, time.localtime(time.time())))
    if not os.path.exists(path):
        os.makedirs(path)
    args.log_path = path

    # unknown class configuration
    if args.dataset == 'mafaulda':
        ALL_CLASSES = config.MAFAULDA_CLASSES
    elif args.dataset == 'uottawa':
        ALL_CLASSES = config.UOTTAWA_CLASSES
    elif args.dataset == 'dirg':
        ALL_CLASSES = config.DIRG_CLASSES
    elif args.dataset == 'sq':
        ALL_CLASSES = config.SQ_CLASSES
    elif args.dataset == 'gbvc':
        ALL_CLASSES = config.GBVC_CLASSES
    elif args.dataset == 'vmcd':
        ALL_CLASSES = config.VMCD_CLASSES

    # train classes
    training_classes = [c for c in ALL_CLASSES if c not in args.exclude_classes]
    
    # test classes
    testing_classes = ALL_CLASSES 
    
    args.num_known_classes = len(training_classes)
    print(f"Open Set Config:")
    print(f"  - Training (Known): {training_classes} (Count: {len(training_classes)})")
    print(f"  - Testing (All): {testing_classes} (Count: {len(testing_classes)})")
    print(f"  - Excluded (Unknown): {args.exclude_classes}")

    args.training_classes = training_classes
    args.testing_classes = testing_classes
    args.window_size = args.seq_len * args.patch_size
    
    exp = Exp_GM_CNF(args)

    # hyperparameters save
    with open(os.path.join(path, 'hyperparameters.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('>>>>>>> Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()

if __name__ == "__main__":
    main()
