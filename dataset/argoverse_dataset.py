import zlib
import pickle
import math
import random
import numpy as np
import torch

from torch.utils.data import Dataset


class Argoverse_Dataset(Dataset):
    def __init__(self, args, validation=False):
        self.validation = validation

        self.ex_file_path = args.ex_file_path
        self.val_ex_file_path = args.val_ex_file_path

        # === Data Augmentations ===
        self.static_agent_drop = args.static_agent_drop
        self.scaling = args.scaling

        if validation:
            self.static_agent_drop = False
            self.scaling = False

        if validation or args.validate:
            with open(self.val_ex_file_path, 'rb') as pickle_file:
                self.ex_list = pickle.load(pickle_file)
        else:
            with open(self.ex_file_path, 'rb') as pickle_file:
                self.ex_list = pickle.load(pickle_file)
            
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        data_compress = self.ex_list[idx]
        mapping = pickle.loads(zlib.decompress(data_compress))
        mapping = self.traditional_preprocess(mapping)
        return mapping

    def traditional_preprocess(self, mapping):
        # agent vector
        # [pre_x, pre_y, x, y, ...]
        #
        # lane vector
        # [..., y, x, pre_y, pre_x]

        matrix = mapping["matrix"]
        polyline_spans = mapping["polyline_spans"]
        map_start_polyline_idx = mapping["map_start_polyline_idx"]
        labels = torch.from_numpy(mapping["labels"])

        # deleted agents with 1 vector
        information_matrix = torch.from_numpy(mapping["information_matrix"])
        vector_available = torch.where((information_matrix[:20] == 1).sum(dim=0) > 1)[0]
        labels = labels[:, vector_available]
        information_matrix = information_matrix[:, vector_available]
        pos_matrix = torch.from_numpy(mapping["pos_matrix"])[:, vector_available]

        assert map_start_polyline_idx == labels.shape[1]
        assert map_start_polyline_idx == information_matrix.shape[1]

        random_scale = 1.0
        if self.scaling:
            random_scale = 0.75 + random.random()*0.5

        agent_list = []
        agent_indices = []
        lane_list = []
        meta_info = []
        for j, polyline_span in enumerate(polyline_spans):
            tensor = torch.tensor(matrix[polyline_span])

            # === Augmentation ===
            drop = False
            is_agent_polyline = j < map_start_polyline_idx

            if is_agent_polyline and (len(tensor) == 1):
                continue

            if is_agent_polyline:
                displacement = torch.norm(tensor[-1, 2:4] - tensor[0, :2])

            # === Static Agent Drop ===
            if (is_agent_polyline) and (j != 0) and (displacement < 1.0) and (self.static_agent_drop):
                drop = random.random() < 0.1

            if is_agent_polyline and not drop:
                tensor[:, :4] *= random_scale
                agent_list.append(tensor)
                agent_indices.append(j)

                dx, dy = tensor[-1, 2:4] - tensor[-1, :2]
                degree = math.atan2(dy, dx)
                x = tensor[-1, 2]
                y = tensor[-1, 3]
                pre_x = tensor[-1, 0]
                pre_y = tensor[-1, 1]
                info = torch.tensor(
                    [degree, x, y, pre_x, pre_y]).unsqueeze(dim=0)
                meta_info.append(info)

            elif not is_agent_polyline:
                tensor[:, -4:] *= random_scale
                tensor[:, -18:-16] *= random_scale
                lane_list.append(tensor)

        assert len(agent_indices) > 0

        meta_info = torch.cat(meta_info, dim=0)
        assert len(meta_info) == len(agent_indices)

        labels *= random_scale
        labels = labels[:, torch.tensor(agent_indices, dtype=torch.long)]

        pos_matrix = pos_matrix[:, torch.tensor(
            agent_indices, dtype=torch.long)]*random_scale

        label_is_valid = information_matrix[20:, torch.tensor(
            agent_indices, dtype=torch.long)]

        information_matrix = information_matrix[:, torch.tensor(
            agent_indices, dtype=torch.long)]

        full_traj = torch.mean(information_matrix, dim=0) == 1
        # moving.shape = (#agent_num)
        moving = torch.norm(pos_matrix[19] - pos_matrix[0], dim=-1) > 6.0
        moving[0] = True

        consider = torch.where(torch.logical_and(full_traj, moving))[0]
        assert 0 in consider


        new_mapping = {"agent_data": agent_list,
                       "lane_data": lane_list,
                       "city_name": mapping["city_name"],
                       "file_name": mapping["file_name"],
                       "origin_labels": mapping["origin_labels"],
                       "labels": labels,
                       "label_is_valid": label_is_valid,
                       "consider": consider,
                       "cent_x": mapping["cent_x"],
                       "cent_y": mapping["cent_y"],
                       "angle": mapping["angle"],
                       "meta_info": meta_info}

        return [new_mapping]


def batch_list_to_batch_tensors(batch):
    return [item for sublist in batch for item in sublist]


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = random.randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)
