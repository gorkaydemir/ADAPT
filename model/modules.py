import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import Attention
from utils.utils import get_meta_info


def merge_tensors(tensors, device, hidden_size):
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


class Trajectory_Loss(nn.Module):
    def __init__(self, args):
        super(Trajectory_Loss, self).__init__()

    def forward(self, prediction, log_prob, valid, label):
        device = prediction.device
        loss_ = 0

        M = prediction.shape[0]
        # norms.shape = (M_c, 6)
        norms = torch.norm(prediction[:, :, -1] - label[:, -1].unsqueeze(dim=1), dim=-1)
        # best_ids.shape = (M_c)
        best_ids = torch.argmin(norms, dim=-1)

        # === L_reg ===
        # l_reg.shape = (M_c, T_f, 2)
        l_reg = F.smooth_l1_loss(prediction[torch.arange(M, device=device), best_ids], label, reduction="none")
        l_reg = (l_reg*valid).sum()/(valid.sum()*2)
        loss_ += l_reg
        # === === ===

        # === L_cls ===
        loss_ += F.nll_loss(log_prob, best_ids)
        # === === ===

        # === L_end ===
        loss_ += F.smooth_l1_loss(prediction[torch.arange(M, device=device), best_ids, -1], label[:, -1], reduction="mean")
        # === === ===

        return loss_


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out


class Dynamic_Trajectory_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Dynamic_Trajectory_Decoder, self).__init__()

        self.D_in = input_dim
        self.D_out = output_dim
        self.mlp = MLP(self.D_in, self.D_in, residual=True)

        self.weight_layer1 = nn.Linear(self.D_in, self.D_in*self.D_out)
        self.bias1 = nn.Linear(self.D_in, self.D_out)

        self.weight_layer2 = nn.Linear(self.D_in, self.D_out*self.D_out)
        self.bias2 = nn.Linear(self.D_in, self.D_out)

        self.norm1 = nn.LayerNorm(self.D_out)

    def forward(self, agent_features):
        # agent_features.shape = (N, M, D)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        assert D == self.D_in

        D_in = self.D_in
        D_out = self.D_out

        # agent_features_weights.shape = (N*M, D_in)
        w_source = self.mlp(agent_features).view(N*M, D_in)
        # agent_features.shape = (N*M, D_in, 1)
        agent_features = agent_features.view(N*M, D_in, 1)

        # === Weight Calculation ===
        # W_1.shape = (N*M, D_out, D_in)
        W_1 = self.weight_layer1(w_source).view(-1, D_out, D_in)
        # b_1.shape = (N, M, D_out)
        b_1 = self.bias1(w_source).view(N, M, D_out)

        # W_2.shape = (N*M, D_out, D_out)
        W_2 = self.weight_layer2(w_source).view(-1, D_out, D_out)
        # b_2.shape = (N, M, D_out)
        b_2 = self.bias2(w_source).view(N, M, D_out)
        # === === ===

        # agent_features.shape = (N, M, D_out)
        out = torch.bmm(W_1, agent_features).view(N, M, D_out)
        out += b_1
        out = self.norm1(out)
        out = F.relu(out)

        # out.shape = (N*M, D_out, 1)
        out = out.view(N*M, D_out, 1)

        # agent_features.shape = (N, M, D_out)
        out = torch.bmm(W_2, out).view(N, M, D_out)
        out += b_2

        return out






class Attention_Block(nn.Module):
    def __init__(self, hidden_size, num_heads=8, p_drop=0.1):
        """
            Transformer Encoder Layer
            This Attention_Block corresponds to MAB of autobot

            If key_value is given, Cross Attention Encoding
            Else Self Attention Encoding
        """
        super(Attention_Block, self).__init__()
        self.multiheadattention = Attention(hidden_size, num_heads, p_drop)

        self.ffn_layer = MLP(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value=None, attn_mask=None):
        if key_value is None:
            key_value = query

        attn_output = self.multiheadattention(
            query, key_value, attention_mask=attn_mask)

        query = self.norm1(attn_output + query)
        query_temp = self.ffn_layer(query)
        query = self.norm2(query_temp + query)

        return query


class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()
        """
            This Sub_Graph corresponds to Polyline Subgraph of VectorNet
        """

        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [MLP(hidden_size, hidden_size//2) for _ in range(depth)])

    def forward(self, lane_list):
        batch_size = len(lane_list)
        device = lane_list[0].device
        hidden_states, lengths = merge_tensors(
            lane_list, device, self.hidden_size)
        max_vector_num = hidden_states.shape[1]

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, self.hidden_size//2], device=device)
        attention_mask_final = torch.zeros(
            [batch_size, max_vector_num, self.hidden_size], device=device)

        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i][lengths[i]:max_vector_num].fill_(-10000.0)
            attention_mask_final[i][lengths[i]:max_vector_num].fill_(-10000.0)

        zeros = torch.zeros([self.hidden_size//2], device=device)

        for layer_index, layer in enumerate(self.layers):
            new_hidden_states = torch.zeros(
                [batch_size, max_vector_num, self.hidden_size], device=device)

            # encoded_hidden_states.shape = (lane_num, max_vector_num, hidden_size)
            #                            or (agent_num, max_vector_num, hidden_size)
            encoded_hidden_states = layer(hidden_states)

            # max_hidden.shape = (lane_num, hidden_size)
            #                 or (agent_num, hidden_size)
            max_hidden, _ = torch.max(
                encoded_hidden_states + attention_mask, dim=1)

            max_hidden = torch.max(
                max_hidden, zeros.unsqueeze(0).expand_as(max_hidden))

            new_hidden_states = torch.cat([encoded_hidden_states, max_hidden.unsqueeze(
                1).expand_as(encoded_hidden_states)], dim=-1)
            hidden_states = new_hidden_states

        # hidden_states.shape = (lane_num, hidden_size)
        #                    or (agent_num, hidden_size)
        hidden_states, _ = torch.max(
            hidden_states + attention_mask_final, dim=1)

        return hidden_states


class Trajectory_Decoder(nn.Module):
    def __init__(self, args):
        super(Trajectory_Decoder, self).__init__()

        self.multi_agent = args.multi_agent

        meta_size = 5
        hidden_size = args.hidden_size

        if self.multi_agent:
            self.endpoint_predictor = Dynamic_Trajectory_Decoder(hidden_size + meta_size, 6*2)
        else:
            self.endpoint_predictor = MLP(hidden_size + meta_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + meta_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + meta_size + 2, 2, residual=True)
        self.get_prob = MLP(hidden_size + meta_size + 2, 1, residual=True)

    def forward(self, agent_features, meta_info):
        # agent_features.shape = (N, M, 128)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # meta_info_tensor.shape = (N, M, 5)
        meta_info_tensor = get_meta_info(meta_info)
        # meta_info_input.shape = (N, M, 6, 5)
        meta_info_tensor_k = meta_info_tensor.unsqueeze(dim=2).expand(N, M, 6, 5)

        # agent_features.shape = (N, M, 128 + 5)
        agent_features_temp = torch.cat([agent_features, meta_info_tensor], dim=-1)

        # endpoints.shape = (N, M, 6, 2)
        endpoints = self.endpoint_predictor(agent_features_temp).view(N, M, 6, 2)

        # prediction_features.shape = (N, M, 6, 128)
        agent_features = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)
        # meta_info_input.shape = (N, M, 6, 128 + 5)
        agent_features = torch.cat([agent_features, meta_info_tensor_k], dim=-1)

        # offsets.shape = (N, M, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features, endpoints.detach()], dim=-1))
        endpoints += offsets

        # agent_features.shape = (N, M, 6, 128 + 5 + 2)
        agent_features = torch.cat([agent_features, endpoints.detach()], dim=-1)

        predictions = self.get_trajectory(agent_features).view(N, M, 6, 29, 2)
        logits = self.get_prob(agent_features).view(N, M, 6)

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        assert predictions.shape == (N, M, 6, 30, 2)

        return predictions, logits


class Interaction_Module(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Interaction_Module, self).__init__()

        self.depth = depth

        self.AA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])

    def forward(self, agent_features, lane_features, masks):

        for layer_index in range(self.depth):
            # === Lane to Agent ===
            lane_features = self.LA[layer_index](lane_features, agent_features, attn_mask=masks[-1])
            # === === ===

            # === Lane to Lane ===
            lane_features = self.LL[layer_index](lane_features, attn_mask=masks[-2])
            # === ==== ===

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[-3])
            # === ==== ===

            # === Agent to Agent ===
            agent_features = self.AA[layer_index](agent_features, attn_mask=masks[-4])
            # === ==== ===

        return agent_features, lane_features
