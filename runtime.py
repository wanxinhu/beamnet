import argparse
import random
import re
import numpy as np
import torch
from lib.networks import BEAMNet


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', choices=['DRIVE', 'CHASE_DB1'], default='DRIVE')
    parser.add_argument('--data-root', required=True, help='Dataset folder containing images/1st_manual subdirectories')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    for name in ['as-mscb', 'edge', 'rmsab', 'stage0', 'aaf']:
        parser.add_argument('--no-'+name, action='store_true')
    return parser


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(device)


def model_config(args):
    return dict(encoder='resnet34', activation='relu6', kernel_sizes=[1,3,5], expansion_factor=2,
                use_as_mscb=not args.no_as_mscb, use_edge=not args.no_edge,
                use_grab=not args.no_rmsab, use_stage0=not args.no_stage0, use_aaf=not args.no_aaf)


def build_model(args, device, pretrained=False):
    return BEAMNet(pretrain=pretrained, **model_config(args)).to(device)


def load_weights(model, path, device):
    weights = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=True)


def safe_run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('run ID may contain only letters, numbers, hyphens and underscores')
    return value
