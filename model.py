import torch
import torch.nn as nn
import torch.nn.functional as F

from model_GT import SGT, EncoderLayer, FeedForwardNetwork, MultiHeadAttention
from utils import node_neighborhood_feature

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, args):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        # self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)
        self.SGT = SGT(n_layers=args.n_layers,
            input_dim=n_in,
            hidden_dim=args.hidden_dim,
            n_class=2,
            num_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout_rate=args.dropout,
            attention_dropout_rate=args.attention_dropout,
            args=args)
        

        # GT only
        self.GT_pre_MLP = nn.Linear(2 * 745, args.hidden_dim)
        encoders = [EncoderLayer(args.hidden_dim, args.ffn_dim, args.dropout, args.attention_dropout, args.n_heads)
                    for _ in range(args.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.hidden_dim)

        self.to(args.device)

    def forward(self, seq1, processed_seq1, adj, sample_abnormal_idx, normal_idx, train_flag, args, sparse=False):
        seq1 = seq1.to(args.device)
        adj = adj.to(args.device)
        # 用 GCN 或 SGT

        # GCN：
        #emb, con_loss = self.gcn2(self.gcn1(seq1, adj, sparse), adj, sparse), torch.tensor([0]).to(args.device)

        # GT only
        if True:
            emb = self.GT_pre_MLP(processed_seq1)
            for i, l in enumerate(self.layers):
                emb = self.layers[i](emb)
            # out = torch.split(emb, emb.shape[2] // 2, dim=2)
            # att_emb, hop_emb = out[0], out[1]
            # emb = att_emb * args.alpha + hop_emb * (1 - args.alpha)
            emb = self.final_ln(emb)
            con_loss = torch.tensor([0]).to(args.device)

        # SGT：
        # emb, con_loss = self.SGT(processed_feat)
        # emb = emb.unsqueeze(0).to(args.device)
        
        # print("shape of emb: ", emb.shape)

        emb_con = None
        emb_combine = None
        emb_abnormal = emb[:, sample_abnormal_idx, :]
        
        noise = torch.randn(emb_abnormal.size()) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise.to(args.device)
        # emb_abnormal = emb_abnormal + noise.cuda()
        if train_flag:
            # Add noise into the attribute of sampled abnormal nodes
            # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
            # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

            neigh_adj = adj[0, sample_abnormal_idx, :]
            # emb[0, sample_abnormal_idx, :] =self.act(torch.mm(neigh_adj, emb[0, :, :]))
            # emb[0, sample_abnormal_idx, :] = self.fc4(emb[0, sample_abnormal_idx, :])

            emb_con = torch.mm(neigh_adj, emb[0, :, :])
            emb_con = self.act(self.fc4(emb_con))
            # emb_con = self.act(self.fc6(emb_con))
            emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)

            # TODO ablation study add noise on the selected nodes

            # std = 0.01
            # mean = 0.02
            # noise = torch.randn(emb[:, sample_abnormal_idx, :].size()) * std + mean
            # emb_combine = torch.cat((emb[:, normal_idx, :], emb[:, sample_abnormal_idx, :] + noise), 1)

            # TODO ablation study generate outlier from random noise
            # std = 0.01
            # mean = 0.02
            # emb_con = torch.mm(neigh_adj, emb[0, :, :])
            # noise = torch.randn(emb_con.size()) * std + mean
            # emb_con = self.act(self.fc4(noise))
            # emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)

            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)
            emb[:, sample_abnormal_idx, :] = emb_con
        else:
            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)

        return emb, emb_combine, f_3, emb_con, emb_abnormal, con_loss

class GGADFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, args):
        super(GGADFormer, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        # self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)
        self.SGT = SGT(n_layers=args.n_layers,
            input_dim=n_in,
            hidden_dim=args.hidden_dim,
            n_class=2,
            num_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout_rate=args.dropout,
            attention_dropout_rate=args.attention_dropout,
            args=args)
        

        # GT only
        self.GT_pre_MLP = nn.Linear(2 * 745, args.hidden_dim)
        encoders = [EncoderLayer(args.hidden_dim, args.ffn_dim, args.dropout, args.attention_dropout, args.n_heads)
                    for _ in range(args.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.hidden_dim)
        
        # To generate outlier nodes
        self.generate_net = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim), # 输入是 h_p 和 h_CLS 的拼接
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
            # nn.Tanh()
        )

        self.to(args.device)

    def forward(self, seq1, processed_seq1, adj, sample_normal_idx, all_normal_idx, train_flag, args, sparse=False):
        seq1 = seq1.to(args.device)
        adj = adj.to(args.device)

        attention_weights = None # 初始化注意力权重
        emb = self.GT_pre_MLP(processed_seq1)
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                attention_weights = current_attention_weights
        emb = self.final_ln(emb)
        # emb: [1, num_nodes, hidden_dim]
        # attention_weights: [1, num_heads, num_nodes, num_nodes]

        # 聚合不同注意力头的注意力
        if attention_weights is not None:
            agg_attention_weights = torch.mean(attention_weights, dim=1)
        else:
            agg_attention_weights = None

        emb_con = None
        emb_combine = None
        emb_abnormal = emb[:, sample_normal_idx, :]
        
        noise = torch.randn(emb_abnormal.size()) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise.to(args.device)
        con_loss = torch.tensor([0]).to(args.device)
        if train_flag:
            # 获取全局上下文表示 h_CLS，[1, 1, hidden_dim]
            h_CLS = torch.mean(emb, dim=1, keepdim=True)
            # 采样出的用于生成离群点的正常节点的上下文表示 h_p，[1, len(sample_normal_idx), hidden_dim]
            h_p = emb[:, sample_normal_idx, :]
            h_CLS_expanded = h_CLS.expand_as(h_p)
            # 拼接 h_p 和 h_CLS_expanded 作为 GenerateNet 的输入，[1, len(sample_normal_idx), 2 * hidden_dim]
            generate_net_input = torch.cat((h_p, h_CLS_expanded), dim=2)
            perturbation = self.generate_net(generate_net_input)

            # Enable or disable attention-based local perturbation
            agg_attention_weights = None # 如果不需要注意力局部扰动，可以将其设置为 None

            # 计算基于注意力局部扰动
            if agg_attention_weights is not None:
                agg_perturbations = self.calculate_local_perturbation(
                    h_p=h_p,
                    full_embeddings=emb, # 传递完整的 emb 以便获取所有节点的特征
                    agg_attention_weights=agg_attention_weights,
                    sample_normal_idx=sample_normal_idx,
                    adj=adj,
                    args=args # 传递 args 以获取 topk_neighbors_attention
                )
            else:
                agg_perturbations = torch.zeros_like(h_p) # 如果没有注意力，局部扰动为0

            
            alpha = args.alpha_outlier_generation
            # emb_con: [1, len(sample_normal_idx), hidden_dim]
            emb_con = h_p + alpha * perturbation + 0.1 * agg_perturbations # emb_con 现在是生成的离群点表示

            # 构建 emb_combine, [1, num_normal_nodes + len(sample_normal_idx), hidden_dim]
            emb_combine = torch.cat((emb[:, all_normal_idx, :], emb_con), 1)


            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)
            # 替换采样节点的嵌入为新生成的离群点嵌入
            emb[:, sample_normal_idx, :] = emb_con

            # For loss_contrastive calculation

            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=-1)
            h_CLS_norm = torch.nn.functional.normalize(h_CLS, p=2, dim=-1)
            emb_con_norm = torch.nn.functional.normalize(emb_con, p=2, dim=-1)

            # 正常节点与 h_CLS 的相似度
            normal_nodes_emb_norm = emb_norm[:, all_normal_idx, :]
            sim_normal_to_cls = torch.sum(normal_nodes_emb_norm * h_CLS_norm.expand_as(normal_nodes_emb_norm), dim=-1) / args.temp
            
            mean_pos_sim = torch.mean(sim_normal_to_cls).item()
            std_pos_sim = torch.std(sim_normal_to_cls).item()


            # 正常节点与所有生成的离群点的相似度
            sim_normal_to_outliers = torch.bmm(normal_nodes_emb_norm, emb_con_norm.transpose(1, 2)) / args.temp
            sum_exp_neg_normal = torch.sum(torch.exp(sim_normal_to_outliers), dim=-1)

            mean_neg_sim = torch.mean(sim_normal_to_outliers).item()
            std_neg_sim = torch.std(sim_normal_to_outliers).item()

            # print(f"  Normal-CLS Sim (Positive): Mean={mean_pos_sim:.4f}, Std={std_pos_sim:.4f}")
            # print(f"  Normal-Outlier Sim (Negative): Mean={mean_neg_sim:.4f}, Std={std_neg_sim:.4f}")

            sim_gap_mean = mean_pos_sim - mean_neg_sim
            # print(f"  Mean Sim Gap (Pos - Neg): {sim_gap_mean:.4f}")

            logits_normal_alignment = torch.cat([sim_normal_to_cls.unsqueeze(-1), sim_normal_to_outliers], dim=-1) # [1, len(normal_idx), 1 + len(sample_normal_idx)]
            labels_normal_alignment = torch.zeros(logits_normal_alignment.shape[1], dtype=torch.long).to(args.device).unsqueeze(0) # [1, len(normal_idx)]
            loss_normal_alignment_per_node = -torch.log_softmax(logits_normal_alignment, dim=-1)[:, :, 0]
            L_normal_alignment = torch.mean(loss_normal_alignment_per_node)

            # 构建离群点的负样本集合
            # sim_outlier_to_cls_single 形状 [1, len(sample_normal_idx), 1]
            sim_outlier_to_cls_single = torch.sum(emb_con_norm * h_CLS_norm.expand_as(emb_con_norm), dim=-1, keepdim=True) / args.temp
            # sim_outlier_to_normals 形状 [1, len(sample_normal_idx), len(normal_idx)]
            sim_outlier_to_normals = torch.bmm(emb_con_norm, normal_nodes_emb_norm.transpose(1, 2)) / args.temp

            # 将 h_CLS_norm 和 normal_nodes_emb_norm 作为离群点的负样本集合
            # logits_outlier_separation 形状 [1, len(sample_normal_idx), 1 + len(normal_idx)]
            logits_outlier_separation = torch.cat([sim_outlier_to_cls_single, sim_outlier_to_normals], dim=-1)

            L_outlier_separation = torch.mean(torch.logsumexp(logits_outlier_separation, dim=-1))
            # 总的对比损失
            con_loss = 1 * L_normal_alignment + 1 * L_outlier_separation

            # For loss calculation in main function
            # emb_con [1, len(sample_normal_idx), hidden_dim] -> [len(sample_abnormal_idx), hidden_dim]
            emb_con = emb_con.squeeze(0)
            # con_loss = torch.tensor([0]).to(args.device)
            
        else:
            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)

        return emb, emb_combine, f_3, emb_con, emb_abnormal, con_loss
    
    def calculate_local_perturbation(self, h_p, full_embeddings, agg_attention_weights, sample_normal_idx, adj, args):
        list_agg_perturbations = []
        # 确保 full_embeddings, agg_attention_weights, adj 都是 [1, ...] 且在 device 上
        
        # 调整 adj 的形状，方便索引。假设 adj 是 [1, num_nodes, num_nodes]
        adj_matrix = adj.squeeze(0) # 形状 [num_nodes, num_nodes]

        # h_p 已经是从 full_embeddings 中取出的，这里用原始索引 p_idx 来从 full_embeddings 中取邻居
        # full_embeddings 形状 [1, num_nodes, hidden_dim]

        for i, p_idx in enumerate(sample_normal_idx): # 遍历每个被选为离群点生成的原始正常节点
            # 获取节点 p_idx 在聚合注意力矩阵中的行 [num_nodes]
            att_row_for_p = agg_attention_weights[0, p_idx, :] # 从 batch_size=1 的维度取

            # 获取节点 p_idx 的邻居索引 (来自原始邻接矩阵 adj)
            # adj_matrix 的 p_idx 行表示 p_idx 的邻居连接
            # neighbor_indices 是一个 Tensor，包含邻居的实际索引
            neighbor_indices = torch.nonzero(adj_matrix[p_idx]).squeeze(1)

            if neighbor_indices.numel() == 0: # 如果没有邻居
                agg_h = torch.zeros_like(h_p[0, i, :]) # 返回零向量
            else:
                # 从这些邻居中，根据注意力分数选择 Top-k
                # 获取这些邻居的注意力分数
                attention_scores_of_neighbors = att_row_for_p[neighbor_indices]

                # 确保 k 不超过实际邻居的数量
                k = min(args.topk_neighbors_attention, neighbor_indices.numel())
                if k == 0: # 再次检查 k 是否为 0
                    agg_h = torch.zeros_like(h_p[0, i, :])
                else:
                    # 找到这些邻居中 Top-k 的索引（相对于 attention_scores_of_neighbors 的索引）
                    topk_neighbor_attention_values, topk_relative_indices = torch.topk(attention_scores_of_neighbors, k=k, dim=-1)
                    # 获取实际的 Top-k 邻居的索引
                    topk_actual_neighbor_indices = neighbor_indices[topk_relative_indices]

                    # 获取这些 Top-k 邻居的嵌入
                    topk_neighbor_embeddings = full_embeddings[0, topk_actual_neighbor_indices, :] # 形状 [k, hidden_dim]

                    # 聚合：使用加权平均，权重就是 Top-k 注意力值
                    # 注意力权重需要归一化，因为 torch.softmax 已经做过，但 topk 筛选后可能需要重新归一化
                    # 如果 sum(topk_neighbor_attention_values) 为 0，避免除以 0
                    if topk_neighbor_attention_values.sum() > 0:
                        normalized_weights = topk_neighbor_attention_values / topk_neighbor_attention_values.sum()
                        agg_h = torch.sum(topk_neighbor_embeddings * normalized_weights.unsqueeze(-1), dim=0)
                    else:
                        agg_h = torch.zeros_like(h_p[0, i, :])

            list_agg_perturbations.append(agg_h)

        # 将列表转换为 Tensor，形状 [1, len(sample_abnormal_idx), hidden_dim]
        agg_perturbations = torch.stack(list_agg_perturbations, dim=0).unsqueeze(0)
        return agg_perturbations
