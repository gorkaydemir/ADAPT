import random
import torch
import torch.distributed as dist
import os
import logging
import math
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
from argoverse.evaluation import eval_forecasting

import torch.backends.cudnn as cudnn


origin_point = None
origin_angle = None

multi_minFDE = []
multi_minADE = []


def multi_agent_metrics(multi_outs, first_time, evaluate):
    global multi_minFDE
    global multi_minADE
    if first_time:
        multi_minFDE = []
        multi_minADE = []

    if evaluate:
        assert len(multi_minFDE) == len(multi_minADE)

        avg_minFDE = sum(multi_minFDE)/len(multi_minFDE)
        avg_minADE = sum(multi_minADE)/len(multi_minADE)
        MR = np.sum(np.array(multi_minFDE) > 2.0)/len(multi_minFDE)

        print("\nMulti Agent Evaluation")
        print(f"minADE: {avg_minADE:.5f}")
        print(f"minFDE: {avg_minFDE:.5f}")
        print(f"MR: {MR:.5f}")
    else:
        batch_size = len(multi_outs)
        for i in range(batch_size):
            # label.shape = (M_c, T_f, 2)
            # output.shape = (M_c, 6, T_f, 2)

            output = multi_outs[i][0]
            label = multi_outs[i][1]
            assert output.shape[0] == label.shape[0]

            norms = torch.norm(
                output[:, :, -1] - label[:, -1].unsqueeze(dim=1), dim=-1)
            # best_ids.shape = (M_c)
            best_ids = torch.argmin(norms, dim=-1)

            # output.shape = (M_c, T_f, 2)
            output = output[torch.arange(len(best_ids)), best_ids]

            # minFDE.shape = (M_c)
            minFDE = torch.norm(output[:, -1] - label[:, -1], dim=-1)

            # minAde.shape = (M_c)
            minADE = torch.mean(torch.norm(output - label, dim=-1), dim=-1)

            assert minADE.shape == minFDE.shape

            multi_minFDE.extend(minFDE.tolist())
            multi_minADE.extend(minADE.tolist())


def get_meta_info(meta_info):
    batch_size = len(meta_info)
    device = meta_info[0].device

    agent_lengths = [len(scene) for scene in meta_info]
    max_agent_num = max(agent_lengths)

    meta_info_tensor = torch.zeros(
        batch_size, max_agent_num, 5, device=device)

    for i, agent_length in enumerate(agent_lengths):
        meta_info_tensor[i, :agent_length] = meta_info[i]

    return meta_info_tensor


def get_masks(agent_lengths, lane_lengths, device):
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    batch_size = len(agent_lengths)

    # === === Mask Generation Part === ===
    # === Agent - Agent Mask ===
    # query: agent, key-value: agent
    AA_mask = torch.zeros(
        batch_size, max_agent_num, max_agent_num, device=device)

    for i, agent_length in enumerate(agent_lengths):
        AA_mask[i, :agent_length, :agent_length] = 1
    # === === ===

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1
    # === === ===

    # === Lane - Lane Mask ===
    # query: lane, key-value: lane
    LL_mask = torch.zeros(
        batch_size, max_lane_num, max_lane_num, device=device)

    QL_mask = torch.zeros(
        batch_size, 6, max_lane_num, device=device)

    for i, lane_length in enumerate(lane_lengths):
        LL_mask[i, :lane_length, :lane_length] = 1

        QL_mask[i, :, :lane_length] = 1

    # === === ===

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros(
        batch_size, max_lane_num, max_agent_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_lengths, agent_lengths)):
        LA_mask[i, :lane_length, :agent_length] = 1

    # === === ===

    masks = [AA_mask, AL_mask, LL_mask, LA_mask]

    # === === === === ===

    return masks, QL_mask


def eval_instance_argoverse(batch_size, pred, pred_probs, mapping, file2pred, file2labels, file2probs, DEs, iter_bar, first_time):
    def get_dis_point_2_points(point, points):
        assert points.ndim == 2
        return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))
    global method2FDEs
    if first_time:
        method2FDEs = []

    for i in range(batch_size):
        a_pred = pred[i]
        a_prob = pred_probs[i]
        # a_endpoints = all_endpoints[i]
        assert a_pred.shape == (6, 30, 2)
        assert a_prob.shape == (6, )

        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2labels[file_name_int] = mapping[i]['origin_labels']
        file2probs[file_name_int] = a_prob

    DE = np.zeros([batch_size, 30])
    for i in range(batch_size):
        origin_labels = mapping[i]['origin_labels']
        FDE = np.min(get_dis_point_2_points(
                origin_labels[-1], pred[i, :, -1, :]))
        method2FDEs.append(FDE)
        for j in range(30):
            DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                    origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
    DEs.append(DE)
    miss_rate = 0.0
    miss_rate = np.sum(np.array(method2FDEs) > 2.0) / len(method2FDEs)

    iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def post_eval(file2pred, file2labels, file2probs, DEs):

    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, 30, 2.0, file2probs)

    for key in metric_results.keys():
        print(f"{key}_6: {metric_results[key]:.5f}")

    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 1, 30, 2.0, file2probs)

    for key in metric_results.keys():
        print(f"{key}_1: {metric_results[key]:.5f}")

    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        print(f" {'ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3'}: {score:.5f}")


def batch_init(mapping):
    global origin_point, origin_angle
    batch_size = len(mapping)

    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']


def to_origin_coordinate(points, idx_in_batch):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                    point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])


def merge_tensors(tensors, device, hidden_size):
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def save_predictions(args, predictions):
    pred_save_path = os.path.join(args.model_save_path, "predictions")
    pickle_file = open(pred_save_path, "wb")
    pickle.dump(predictions, pickle_file,
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle_file.close()


def setup(rank, world_size):
    now = datetime.now()
    s = int(now.second)
    m = int(now.minute)

    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = f"{12300 + s*2 + m*5}"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
