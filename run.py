import sys
import argparse
import torch
import os
from tqdm import tqdm
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.get_model import get_model, save_model
from dataset.argoverse_dataset import Argoverse_Dataset, batch_list_to_batch_tensors, __iter__
from utils.utils import fix_seed, setup, save_predictions
from utils.utils import eval_instance_argoverse, post_eval, multi_agent_metrics

from read_args import get_args


def validate(args, model, dataloader):
    file2pred = {}
    file2probs = {}
    file2labels = {}
    DEs = []
    iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')

    multi_outputs = []
    # reset inference time counter
    model.module.inference_time = 0.0

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(iter_bar):

            pred_trajectory, pred_probs, multi_out = model.module(batch, True)
            batch_size = pred_trajectory.shape[0]
            for i in range(batch_size):
                assert pred_trajectory[i].shape == (6, 30, 2)
                assert pred_probs[i].shape == (6, )

            # batch = [scene[0] for scene in batch]
            eval_instance_argoverse(
                batch_size, pred_trajectory, pred_probs, batch, file2pred, file2labels, file2probs, DEs, iter_bar, step == 0)
            if args.multi_agent:
                multi_agent_metrics(multi_out, step == 0, evaluate=False)
                multi_outputs.extend(multi_out)

    print(f"\nInference time: {model.module.inference_time / len(dataloader.dataset) * 1000:.2f} ms\n")
    post_eval(file2pred, file2labels, file2probs, DEs)
    if args.multi_agent:
        multi_agent_metrics(None, False, evaluate=True)

    predictions = {"file2pred": file2pred,
                   "file2probs": file2probs,
                   "file2labels": file2labels,
                   "multi_outputs": multi_outputs}
    return predictions


def train(model, iter_bar, optimizer, args, scheduler):
    main_device = (args.device == 0)
    total_loss = 0.0

    model.train()
    for step, batch in enumerate(iter_bar):

        traj_loss = model(batch)
        loss = traj_loss
        total_loss += loss.item()

        loss.backward()
        if main_device:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_desc = f"lr = {lr:.6f} loss = {total_loss/(step+1):.5f}"
            iter_bar.set_description(loss_desc)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


def main(rank, args):
    main_device = True if rank == 0 else False
    args.device = rank

    setup(rank, args.world_size)

    
    if not args.validate:
        # set train dataset and dataloader
        train_dataset = Argoverse_Dataset(args)
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size // args.world_size,
                                      pin_memory=False, drop_last=False, shuffle=False, sampler=train_sampler,
                                      collate_fn=batch_list_to_batch_tensors)

    # if main device, load validation dataset
    if main_device:
        val_dataset = Argoverse_Dataset(args, validation=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size // args.world_size,
                                    pin_memory=False, drop_last=False, shuffle=False,
                                    collate_fn=batch_list_to_batch_tensors)
        

    iter_num = None if args.validate else len(train_dataloader)
    model, optimizer, start_epoch, scheduler = get_model(args, iter_num)

    # Parameter number
    if main_device:
        model_parameters = filter(lambda p: p.requires_grad, model.module.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"\nParameter Number: {params / 1e6:.2f}M\n")


    # validation
    if args.validate:
        if main_device:
            predictions = validate(args, model, val_dataloader)
            save_predictions(args, predictions)

        dist.barrier()
        dist.destroy_process_group()
        return

    for i_epoch in range(start_epoch, args.epoch):
        if main_device: print(f"====== [Epoch {i_epoch + 1}/{args.epoch}] ======")
        train_sampler.set_epoch(i_epoch)

        if main_device: iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        else: iter_bar = train_dataloader

        train(model, iter_bar, optimizer, args, scheduler)

        if main_device:
            save_model(args, i_epoch, model, optimizer, scheduler)

            predictions = validate(args, model, val_dataloader)
            save_predictions(args, predictions)

            print("====== ====== ======\n")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    fix_seed(args.seed)
    mp.spawn(main, args=[args], nprocs=args.world_size)
