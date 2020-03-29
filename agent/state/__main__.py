from argparse import ArgumentParser
from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import torch as T
import torch.backends.cudnn
from torch.utils.tensorboard import SummaryWriter

from agent.config import load_config, save_config
from agent.state.model import train_encoder
from agent.state.data import Dataset


parser = ArgumentParser(description='Train state model')
parser.add_argument('--config', type=Path, default='./train_conf.yaml',
                    help='Path to train config')
parser.add_argument('--data_dir', type=Path, default='./data',
                    help='Path to train config')
parser.add_argument('--write_freq', type=int, default=10,
                    help='Write logs at every nth step')
def_log_dir = Path('./runs') / datetime.now().isoformat().replace(':', '')[:-7]
parser.add_argument('--log_dir', type=Path, default=def_log_dir,
                    help='Path to TensorBoard dir')
parser.add_argument('--disable_tb', action='store_true',
                    help='Disable TensorBoard logging and write to stdout')
parser.add_argument('--debug', action='store_true',
                    help='Debug level logs')
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logging.info('Using config: {}'.format(args.config))

conf = load_config(args.config)
if conf.cuda:
    if T.cuda.is_available() and T.backends.cudnn.is_available():
        T.backends.cudnn.benchmark = True
        device = T.device('cuda')
    else:
        logging.error('Cuda or Cudnn is not available')
        sys.exit(1)
else:
    device = T.device('cpu')

save_config(conf, args.log_dir)
dataset = Dataset(args.data_dir, conf)
if args.disable_tb:
    writer = None
else:
    writer = SummaryWriter(log_dir=args.log_dir)

ckpt_dir = args.log_dir / 'encoder_ckpt'
os.makedirs(ckpt_dir, exist_ok=True)

logging.info('Starting encoder training. Logdir: {}'.format(args.log_dir))
train_encoder(dataset, conf, device, ckpt_dir, write_freq=args.write_freq,
              writer=writer)

