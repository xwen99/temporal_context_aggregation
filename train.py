import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import horovod.torch as hvd
import utils
from data import VCDBPairDataset
from model import NetVLAD, MoCo, NeXtVLAD, LSTMModule, GRUModule, TCA

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')


def train(args):
    # Horovod: initialize library.
    hvd.init()
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(1)
    kwargs = {'num_workers': args.num_workers,
              'pin_memory': True} if args.cuda else {}


    train_dataset = VCDBPairDataset(annotation_path=args.annotation_path, feature_path=args.feature_path,
                                    padding_size=args.padding_size, random_sampling=args.random_sampling, neg_num=args.neg_num)

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_sz,
                              sampler=train_sampler, drop_last=True, **kwargs)

    model = TCA(feature_size=args.pca_components, nlayers=args.num_layers, dropout=0.2)

    model = MoCo(model, dim=args.output_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = utils.CircleLoss(m=0.25, gamma=256).cuda()

    # Horovod: scale learning rate by lr_scaler.
    if False:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate * lr_scaler,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate * lr_scaler,
                                     weight_decay=args.weight_decay)
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start = datetime.now()
    model.train()
    for epoch in range(1, args.epochs + 1):
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (a, p, n, len_a, len_p, len_n) in enumerate(train_loader):
            if args.cuda:
                a, p, n = a.cuda(), p.cuda(), n.cuda()
                len_a, len_p, len_n = len_a.cuda(), len_p.cuda(), len_n.cuda()

            output, target = model(a, p, n, len_a, len_p, len_n)

            loss = criterion(output, target)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % args.print_freq == 0 and hvd.rank() == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(a), len(train_sampler),
                    100. * (batch_idx + 1) * len(a) / len(train_sampler), loss.item()))

        scheduler.step()

        if hvd.rank() == 0 and epoch % 10 == 0:
            print("Epoch complete in: " + str(datetime.now() - start))
            print("Saving model...")
            torch.save(model.encoder_q.state_dict(), 'models/model.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--annotation_path', type=str, required=True,
                        help='Path to the .pk file that contains the annotations of the train set')
    parser.add_argument('-fp', '--feature_path', type=str, required=True,
                        help='Path to the kv dataset that contains the features of the train set')
    parser.add_argument('-mp', '--model_path', type=str, required=True,
                        help='Directory where the generated files will be stored')

    parser.add_argument('-nc', '--num_clusters', type=int, default=256,
                        help='Number of clusters of the NetVLAD model')
    parser.add_argument('-od', '--output_dim', type=int, default=1024,
                        help='Dimention of the output embedding of the NetVLAD model')
    parser.add_argument('-nl', '--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('-ni', '--normalize_input', action='store_true',
                        help='If true, descriptor-wise L2 normalization is applied to input')
    parser.add_argument('-nn', '--neg_num', type=int, default=1,
                        help='Number of negative samples of each batch')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train the DML network. Default: 5')
    parser.add_argument('-bs', '--batch_sz', type=int, default=256,
                        help='Number of triplets fed every training iteration. '
                             'Default: 256')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate of the DML network. Default: 10^-4')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
                        help='Regularization parameter of the DML network. Default: 10^-4')

    parser.add_argument('-pc', '--pca_components', type=int, default=1024,
                        help='Number of components of the PCA module.')
    parser.add_argument('-ps', '--padding_size', type=int, default=300,
                        help='Padding size of the input data at temporal axis.')
    parser.add_argument('-rs', '--random_sampling', action='store_true',
                        help='Flag that indicates that the frames in a video are random sampled if max frame limit is exceeded')
    parser.add_argument('-nr', '--num_readers', type=int, default=16,
                        help='Number of readers for reading data')
    parser.add_argument('-nw', '--num_workers', type=int, default=4,
                        help='Number of workers of dataloader')

    # moco specific configs:
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    train(args)


if __name__ == '__main__':
    main()
