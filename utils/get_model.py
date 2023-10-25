import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def save_model(args, i_epoch, model, optimizer, scheduler):
    assert args.model_save_path is not None
    path = os.path.join(args.model_save_path, "checkpoint.pt")

    checkpoint = {"epoch": i_epoch,
                  "state_dict": model.module.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict()}
    torch.save(checkpoint, path)

    if (i_epoch + 1) % 6 == 0:
        path = os.path.join(args.model_save_path,
                            f"state_epoch{i_epoch + 1}.pt")
        torch.save(checkpoint, path)


def get_model(args, iter_num):
    from model.adapt import ADAPT
    rank = args.device

    # loading/creating model
    model = ADAPT(args)

    if args.use_checkpoint:
        assert os.path.exists(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if not args.validate:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
        total_cycle = args.epoch * iter_num
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_cycle * 0.7), int(total_cycle * 0.9)], gamma=0.15)

        start_epoch = 0

        if args.use_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1

    else:
        optimizer = None
        start_epoch = None
        scheduler = None

    return model, optimizer, start_epoch, scheduler
