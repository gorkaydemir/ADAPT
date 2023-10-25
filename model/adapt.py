import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder modules
from model.modules import Trajectory_Loss, Trajectory_Decoder, Sub_Graph, Interaction_Module

from utils.utils import merge_tensors, get_masks
from utils.utils import to_origin_coordinate, batch_init


class ADAPT(nn.Module):
    def __init__(self, args):
        super(ADAPT, self).__init__()

        self.hidden_size = args.hidden_size
        self.device = args.device

        self.multi_agent = args.multi_agent
        self.validate = None

        self.lane_subgraph = Sub_Graph(self.hidden_size)
        self.agent_subgraph = Sub_Graph(self.hidden_size)

        self.interaction_module = Interaction_Module(
            self.hidden_size, depth=args.layer_num)
        self.trajectory_decoder = Trajectory_Decoder(args)

        self.trajectory_loss = Trajectory_Loss(args)

        self.inference_time = 0.0

    def forward(self, mapping, validate=False):
        # N: Number of prediction agents
        # M: Number of total agents in the scene
        # T_p: Past time step number
        # T_f: Future time step number
        # L: Lane number
        # D: Feature size

        batch_size = len(mapping)
        if validate:
            batch_init(mapping)

        outputs = torch.zeros(batch_size, 6, 30, 2, device=self.device)
        multi_outputs = []
        probs = torch.zeros(batch_size, 6, device=self.device)
        losses = torch.zeros(batch_size, device=self.device)

        agent_data = [mapping[i]["agent_data"] for i in range(batch_size)]
        lane_data = [mapping[i]["lane_data"] for i in range(batch_size)]
        labels = [mapping[i]["labels"] for i in range(batch_size)]
        label_is_valid = [mapping[i]["label_is_valid"]
                          for i in range(batch_size)]
        meta_info = [mapping[i]["meta_info"] for i in range(batch_size)]
        considers = [mapping[i]["consider"] for i in range(batch_size)]

        for i in range(batch_size):
            for j in range(len(agent_data[i])):
                agent_data[i][j] = agent_data[i][j].to(self.device)

            for j in range(len(lane_data[i])):
                lane_data[i][j] = lane_data[i][j].to(self.device)

            labels[i] = labels[i].to(self.device)
            label_is_valid[i] = label_is_valid[i].to(self.device)
            meta_info[i] = meta_info[i].to(self.device)
            considers[i] = considers[i].to(self.device)

        start = time.time()
        # agent_features.shape = (N, M, D)
        # lane_features.shape = (N, L, D)
        agent_features, lane_features = self.encode_polylines(
                agent_data, lane_data)

        # predictions.shape = (N, M, 6, 30, 2)
        predictions, logits = self.trajectory_decoder(
            agent_features, meta_info)

        end = time.time()

        total_agent_num = 0.0
        for scene_index in range(batch_size):
            # prediction.shape = (M, 6, T_f, 2)
            # logit.shape = (M, 6)
            # label.shape = (M, T_f, 2)
            # valid.shape = (M, T_f)
            # consider.shape = (M_c)
            prediction = predictions[scene_index]
            logit = logits[scene_index]
            label = labels[scene_index].permute(1, 0, 2)
            valid = label_is_valid[scene_index].permute(1, 0)

            if not self.multi_agent:
                consider = torch.tensor(
                    [0], device=self.device, dtype=torch.long)
            else:
                consider = considers[scene_index]

            pred_num = len(consider)
            total_agent_num += pred_num

            # prediction.shape = (M_c, 6, T_f, 2)
            # logit.shape = (M_c, 6)
            # label.shape = (M_c, T_f, 2)
            # valid.shape = (M_c, T_f, 1)
            prediction = prediction[consider]
            logit = logit[consider]
            label = label[consider]
            valid = valid[consider].unsqueeze(dim=-1)

            assert torch.sum(label == -666) == 0

            prob = F.softmax(logit / 0.3, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)

            outputs[scene_index] = prediction[0]
            probs[scene_index] = prob[0]

            if validate and self.multi_agent:
                multi_outputs.append(
                    (prediction.detach().cpu(), label.detach().cpu()))

            if not validate:
                loss_ = self.trajectory_loss(
                        prediction, log_prob, valid, label)

                losses[scene_index] = loss_*pred_num

        if not validate:
            return losses.sum()/total_agent_num

        else:
            outputs = np.array(outputs.tolist())
            metric_probs = np.array(probs.tolist(
            ), dtype=np.float32)

            for i in range(batch_size):
                for each in outputs[i]:
                    to_origin_coordinate(each, i)

            self.inference_time += (end - start)

            return outputs, metric_probs, multi_outputs

    def encode_polylines(self, agent_data, lane_data):
        batch_size = len(agent_data)

        # === === Polyline Subgraph Part === ===
        # len(agent_data) = batch size
        # len(agent_data[i]) = #agents in scene i

        lane_polylines = []
        agent_polylines = []

        for i in range(batch_size):
            # === Lane Encoding ===
            lane_polylines_i = self.lane_subgraph(lane_data[i])
            lane_polylines.append(lane_polylines_i)

            # === Agent Encoding ===
            agent_polylines_i = self.agent_subgraph(agent_data[i])
            agent_polylines.append(agent_polylines_i)

        # lane_states.shape = (batch_size, max_lane_num, hidden_size)
        lane_states, lane_lengths = merge_tensors(
            lane_polylines, self.device, self.hidden_size)
        # agent_states.shape = (batch_size, max_agent_num, hidden_size)
        agent_states, agent_lengths = merge_tensors(
            agent_polylines, self.device, self.hidden_size)

        # === === === === ===

        masks, _ = get_masks(agent_lengths, lane_lengths, self.device)

        agent_states, lane_states = self.interaction_module(agent_states, lane_states, masks)

        return agent_states, lane_states
