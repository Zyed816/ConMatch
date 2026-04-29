from __future__ import print_function, division
import os

import torch

from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader


def _strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict):
        return state_dict
    return {key[len('module.'):]: value for key, value in state_dict.items()}


def _select_state_dict(checkpoint, use_train_model):
    if use_train_model:
        for key in ['train_model', 'model']:
            if key in checkpoint:
                return checkpoint[key]
    else:
        for key in ['eval_model', 'ema_model', 'model']:
            if key in checkpoint:
                return checkpoint[key]
    raise KeyError('No compatible model state dict found in checkpoint.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/fixmatch/model_best.pth')
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model = _strip_module_prefix(_select_state_dict(checkpoint, args.use_train_model))
    
    _net_builder = net_builder(args.net, 
                               args.net_from_name,
                                {'first_stride': 2 if 'stl' in args.dataset else 1,
                                 'depth': args.depth,
                                 'widen_factor': args.widen_factor,
                                 'leaky_slope': args.leaky_slope,
                                 'bn_momentum': 1.0 - 0.999,
                                 'dropRate': args.dropout,
                                 'use_embed': False})

    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    net.to(device)
    net.eval()

    _eval_dset = SSL_Dataset(args, name=args.dataset, train=False,
                             num_classes=args.num_classes, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset()
    
    eval_loader = get_data_loader(eval_dset,
                                  args.batch_size, 
                                  num_workers=1)
 
    acc = 0.0
    with torch.no_grad():
        for _, image, target in eval_loader:
            image = image.float().to(device)
            logit = net(image)
            
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
    
    print(f"Test Accuracy: {acc/len(eval_dset)}")
