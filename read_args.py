import sys
import argparse
import os
import torch
import shutil

class Arguments:
    def __init__(self, args):
        self.ex_file_path = args.ex_file_path
        self.val_ex_file_path = args.val_ex_file_path

        self.validate = args.validate

        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate

        self.model_save_path = args.model_save_path
        self.checkpoint_path = args.checkpoint_path
        self.use_checkpoint = args.use_checkpoint

        self.seed = args.seed

        self.layer_num = args.layer_num
        self.multi_agent = args.multi_agent

        self.static_agent_drop = args.static_agent_drop
        self.scaling = args.scaling

        self.assertions()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.world_size = torch.cuda.device_count()

        # self.print_args()

    def assertions(self):
        assert os.path.exists(self.ex_file_path)
        assert os.path.exists(self.val_ex_file_path)

        if self.use_checkpoint:
            assert self.checkpoint_path is not None
            assert os.path.exists(os.path.join(self.checkpoint_path))

        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)

    def print_args(self):
        print("====== Arguments ======")
        print(f"hidden_size: {self.hidden_size}")
        print(f"batch_size: {self.batch_size}")
        print(f"epoch: {self.epoch}")
        print(f"learning_rate: {self.learning_rate}\n")

        print(f"layer_num: {self.layer_num}")
        print(f"multi_agent: {self.multi_agent}\n")

        print(f"static_agent_drop: {self.static_agent_drop}")
        print(f"scaling: {self.scaling}")
        print("====== ======= ======\n")


def get_args():
    parser = argparse.ArgumentParser("ADAPT")

    # === Data Related Parameters ===
    parser.add_argument('--ex_file_path', type=str)
    parser.add_argument('--val_ex_file_path', type=str)

    # === Test Evaluation Related Parameters ===
    parser.add_argument('--validate', action="store_true")

    # === Common Hyperparameters ===
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=36)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # === Model Saving/Loading Parameters ===
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--use_checkpoint', action="store_true")

    # === Misc Training Parameters ===
    parser.add_argument('--seed', type=int, default=0)

    # ===  Architecture Parameters ===
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--multi_agent', action="store_true")

    # === Data Augmentations Applied ===
    parser.add_argument('--static_agent_drop', action="store_true")
    parser.add_argument('--scaling', action="store_true")

    args = parser.parse_args()

    arg_object = Arguments(args)
    return arg_object