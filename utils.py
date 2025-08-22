import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter
import torch.nn as nn

import os
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler

from model import CommunityAutoencoder


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1):

    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
    all_labeled_normal_idx = all_normal_idx[: int(len(all_normal_idx) * rate)]
    print('Training rate', rate)

    # contamination
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.1 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

    # contamination 
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.05 * len(real_abnormal_id)  #0.05 0.1  0.15
    # remove_rate = 0.15 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # remove_abnormal_id = real_abnormal_id[:int(remove_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, remove_abnormal_id, False)

    # camouflage
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # normal_feat = np.mean(feat[normal_label_idx], 0)
    # replace_rate = 0.05 * normal_feat.shape[1]
    # feat[real_abnormal_id, :int(replace_rate)] = normal_feat[:, :int(replace_rate)]

    random.shuffle(all_labeled_normal_idx)
    # 0.05 for Amazon and 0.15 for other datasets
    if dataset in ['Amazon']:
        sample_normal_idx = all_labeled_normal_idx[: int(len(all_labeled_normal_idx) * 0.05)]  
    else:
        sample_normal_idx = all_labeled_normal_idx[: int(len(all_labeled_normal_idx) * 0.15)]  
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, all_labeled_normal_idx, sample_normal_idx


def adj_to_dgl_graph(adj, dataset_name: str = None):
    """
    Convert adjacency matrix to DGLGraph format, with caching mechanism.

    Args:
        adj: The adjacency matrix (scipy.sparse.csr_matrix).
        dataset_name (str, optional): The name of the dataset. If provided,
                                      the function will check for a cached DGLGraph
                                      and save it after conversion. Defaults to None.

    Returns:
        dgl.DGLGraph: The converted DGLGraph.
    """
    if dataset_name:
        cache_dir = "./dataset/dgl_cache"
        os.makedirs(cache_dir, exist_ok=True) # Ensure cache directory exists
        cache_path = os.path.join(cache_dir, f"{dataset_name}_dgl_graph.bin")

        if os.path.exists(cache_path):
            print(f"Loading DGLGraph for '{dataset_name}' from cache...")
            try:
                # dgl.load_graphs returns a list of graphs and a list of labels/meta_dicts
                glist, _ = dgl.load_graphs(cache_path)
                if glist:
                    return glist[0]
                else:
                    print("Cache file empty or corrupted, re-converting...")
            except Exception as e:
                print(f"Error loading from cache: {e}. Re-converting...")

    # If no dataset_name or cache not found/corrupted, perform conversion
    print(f"Converting adjacency matrix to DGLGraph for '{dataset_name if dataset_name else 'unnamed'}'...")
    nx_graph = nx.from_scipy_sparse_array(adj)
    # DGLGraph should be created from NetworkX graph directly for feature compatibility
    # Ensure all nodes are included even if they have no edges in nx_graph
    dgl_graph = dgl.from_networkx(nx_graph)

    if dataset_name:
        try:
            dgl.save_graphs(cache_path, [dgl_graph]) # dgl.save_graphs expects a list of graphs
            print(f"DGLGraph for '{dataset_name}' saved to cache at {cache_path}.")
        except Exception as e:
            print(f"Warning: Could not save DGLGraph to cache: {e}")

    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size * 3]
        subv[i].append(i)

    return subv


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
# plt.rcParams['figure.figsize'] = (10.5, 9.5)
from matplotlib.backends.backend_pdf import PdfPages


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))
    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 20)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # from matplotlib.pyplot import MultipleLocator
    # x_major_locator = MultipleLocator(0.02)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
    plt.close()


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))

    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 8)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
    plt.close()


def node_neighborhood_feature(adj, features, k, alpha=0.1):

    x_0 = features
    for i in range(k):
        # print(f"features.shape: {features.shape}, adj.shape: {adj.shape}")
        features = (1-alpha) * torch.mm(adj, features) + alpha * x_0

    return features

def preprocess_sample_features(args, features, adj):
    """
    基于节点序列采样方法，准备预处理特征矩阵
    Args:
        args: 输入的训练参数
        features: 特征矩阵, size = (N, d)
    Returns:
        features: 预处理后的特征矩阵, size = (N, args.sample_num_p+1 + args.sample_num_n+1, d)
    """
    data_file = './pre_sample/'+args.dataset +'_'+str(args.sample_num_p)+'_'+str(args.sample_num_n)+"_"+str(args.pp_k)+'.pt'
    if os.path.isfile(data_file):
        processed_features = torch.load(data_file)

    else:
        processed_features = node_seq_feature(features, args.sample_num_p, args.sample_num_n, args.sample_size)  # return (N, hops+1, d)

        if args.pp_k > 0:

            data_file_ppr = './pre_features'+args.dataset +'_'+str(args.pp_k)+'.pt'

            if os.path.isfile(data_file_ppr):
                ppr_features = torch.load(data_file_ppr)

            else:
                ppr_features = node_neighborhood_feature(adj, features, args.pp_k)  # return (N, d)
                # store the data 
                torch.save(ppr_features, data_file_ppr)

            ppr_processed_features = node_seq_feature(ppr_features, args.sample_num_p, args.sample_num_n, args.sample_size)

            processed_features = torch.concat((processed_features, ppr_processed_features), dim=1)

        # store the data
        # 检查父目录是否存在，如果不存在则创建
        if not os.path.exists(os.path.dirname(data_file)):
            os.makedirs(os.path.dirname(data_file))
        torch.save(processed_features, data_file)
    return processed_features

def node_seq_feature(features, pk, nk, sample_batch):
    """
    跟据特征矩阵采样正负样本

    Args:
        features: 特征矩阵, size = (N, d)
        pk: 采样正样本的个数
        nk: 采样负样本的个数
        sample_batch: 每次采样的batch大小
    Returns:
        nodes_features: 采样之后的特征矩阵, size = (N, 1, K+1, d)
    """

    nodes_features_p = torch.empty(features.shape[0], pk+1, features.shape[1])

    nodes_features_n = torch.empty(features.shape[0], nk+1, features.shape[1])

    x = features + torch.zeros_like(features)
    
    x = torch.nn.functional.normalize(x, dim=1)

    # 构建 batch 采样
    total_batch = int(features.shape[0]/sample_batch)

    rest_batch = int(features.shape[0]%sample_batch)

    for index_batch in tqdm(range(total_batch)):

        # x_batch, [b,d]
        # x1_batch, [b,d]
        # 切片操作左闭右开
        x_batch = x[(index_batch)*sample_batch:(index_batch+1)*sample_batch,:]

        
        s = torch.matmul(x_batch, x.transpose(1, 0))


        # Begin sampling positive samples
        # print(s.shape)
        for i in range(sample_batch):
            s[i][(index_batch)*(sample_batch) + i] = -1000        #将得到的相似度矩阵的对角线的值置为负的最大值

        topk_values, topk_indices = torch.topk(s, pk, dim=1)

        for index in range(sample_batch):

            nodes_features_p[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]
            for i in range(pk):
                nodes_features_p[(index_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]

        # Begin sampling positive samples
        if nk > 0:
            all_idx = [i for i in range(s.shape[1])]

            for index in tqdm(range(sample_batch)):

                nce_idx = list(set(all_idx) - set(topk_indices[index].tolist()))
                

                nce_indices = np.random.choice(nce_idx, nk, replace=True)

                nodes_features_n[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]
                for i in range(nk):
                    nodes_features_n[(index_batch)*sample_batch + index, i+1, :] = features[nce_indices[i]]

    if rest_batch > 0:

        x_batch = x[(total_batch)*sample_batch:(total_batch)*sample_batch + rest_batch,:]
        x = x
        # print(f"x_batch.shape: {x_batch.shape}, x.shape: {x.shape}")

        s = torch.matmul(x_batch, x.transpose(1, 0))

        print("------------begin sampling positive samples------------")

        #采正样本
        for i in range(rest_batch):
            s[i][(total_batch)*sample_batch + i] = -1000         #将得到的相似度矩阵的对角线的值置为负的最大值

        topk_values, topk_indices = torch.topk(s, pk, dim=1)
        # print(topk_indices.shape)

        for index in range(rest_batch):
            nodes_features_p[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]
            for i in range(pk):
                nodes_features_p[(total_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]


        print("------------begin sampling negative samples------------")

        #采负样本
        if nk > 0:
            all_idx = [i for i in range(s.shape[1])]

            for index in tqdm(range(rest_batch)):

                nce_idx = list(set(all_idx) - set(topk_indices[index].tolist()))
                

                nce_indices = np.random.choice(nce_idx, nk, replace=False)
                # print(nce_indices)

                # print((index_batch)*sample_batch + index)
                # print(nce_indices)


                nodes_features_n[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]
                for i in range(nk):
                    nodes_features_n[(total_batch)*sample_batch + index, i+1, :] = features[nce_indices[i]]


    nodes_features = torch.concat((nodes_features_p, nodes_features_n), dim=1)
    

    # print(nodes_features_p.shape)
    # print(nodes_features_n.shape)
    # print(nodes_features.shape)

    return nodes_features


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def calculate_modularity_matrix(adj):
    """
    Calculates the Modularity Matrix B for a given adjacency matrix.
    B_ij = A_ij - (k_i * k_j) / (2m)

    Args:
        adj (scipy.sparse.csr_matrix or np.ndarray): The adjacency matrix of the graph.
                                                     Assumes it's symmetric (undirected graph).

    Returns:
        torch.Tensor: The Modularity Matrix B, as a dense PyTorch tensor.
    """
    if isinstance(adj, np.ndarray):
        adj_sparse = sp.csr_matrix(adj)
    elif isinstance(adj, torch.Tensor):
        adj_sparse = sp.csr_matrix(adj.squeeze(0).cpu().numpy()) # Convert to numpy sparse if needed
    else:
        adj_sparse = adj # Assume it's already scipy sparse

    num_nodes = adj_sparse.shape[0]
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    total_edges_2m = degrees.sum() # Sum of degrees is 2m for undirected graph

    # Outer product k_i * k_j
    outer_product = np.outer(degrees, degrees)

    # Expected number of edges
    expected_edges = outer_product / total_edges_2m

    # Modularity Matrix B
    B_matrix = adj_sparse - sp.csr_matrix(expected_edges)

    return torch.FloatTensor(B_matrix.toarray()) # Convert to dense tensor for autoencoder


def train_community_detection_module(
    adj_original: sp.csr_matrix,
    device: torch.device,
    dataset_name: str,
    pretrain_dir: str = 'pretrain',
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dims: list = [128, 64],
    community_embedding_dim: int = 32
):
    """
    Trains the Community Detection Autoencoder or loads a pre-trained one.

    If a pre-trained model for the given dataset_name exists in pretrain_dir,
    it loads the model. Otherwise, it trains a new model and saves it.

    Args:
        adj_original (scipy.sparse.csr_matrix): The original adjacency matrix.
        device (torch.device): Device to run the model on.
        dataset_name (str): The name of the dataset (e.g., 'Cora', 'CiteSeer').
                            Used for naming the saved model file.
        pretrain_dir (str): Directory to save/load the pre-trained models.
        epochs (int): Number of training epochs if training from scratch.
        lr (float): Learning rate if training from scratch.
        hidden_dims (list): List of hidden layer dimensions for autoencoder.
        community_embedding_dim (int): Dimension of the community embedding H.

    Returns:
        torch.Tensor: The learned community embeddings H.
        CommunityAutoencoder: The trained or loaded autoencoder model.
    """
    # 准备路径并确保文件夹存在
    os.makedirs(pretrain_dir, exist_ok=True)
    # 在文件名中包含 community_embedding_dim 参数，防止不同维度时读取错误模型
    model_path = os.path.join(pretrain_dir, f'community_ae_{dataset_name}_dim{community_embedding_dim}.pth')

    # 计算模块度矩阵 B
    B_matrix_tensor = calculate_modularity_matrix(adj_original).to(device)
    input_dim = B_matrix_tensor.shape[1]

    # 初始化模型
    community_ae = CommunityAutoencoder(input_dim, hidden_dims, community_embedding_dim).to(device)

    # 检查预训练模型是否存在
    if os.path.exists(model_path):
        print(f"Found pre-trained community detection model. Loading from: {model_path}")
        # map_location=device 确保模型被加载到当前指定的设备上
        community_ae.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        # 如果模型不存在，则执行训练
        print(f"No pre-trained model found for '{dataset_name}'. Training from scratch...")
        optimizer = torch.optim.Adam(community_ae.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print("Training Community Detection Autoencoder...")
        for epoch in range(epochs):
            community_ae.train()
            optimizer.zero_grad()
            h_output, b_reconstructed = community_ae(B_matrix_tensor)
            loss = criterion(b_reconstructed, B_matrix_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"  Community AE Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        # 训练完成后，保存模型状态
        print(f"Saving trained model to: {model_path}")
        torch.save(community_ae.state_dict(), model_path)

    # 使用加载或新训练好的模型进行最终预测
    print("Generating final community embeddings...")
    community_ae.eval()
    with torch.no_grad():
        final_h, _ = community_ae(B_matrix_tensor)
    
    print("Community detection module is ready.")
    return final_h, community_ae


def get_dynamic_loss_weights(epoch, warmup_epoch, args):
    """
    根据当前epoch和warmup_epoch计算动态损失权重
    
    Args:
        epoch: 当前epoch
        warmup_epoch: warmup epoch数量
        args: 参数对象，包含各种损失权重的目标值
    
    Returns:
        dict: 包含各种损失权重的字典
    """
    if epoch < warmup_epoch:
        # warmup阶段：只开启community_loss和正常节点内部的对比损失
        return {
            'rec_weight': 0.0,
            'perturbation_weight': 0.0,
            'local_perturbation_weight': 0.0,
            'neighbor_perturbation_weight': 0.0,
            'normal_alignment_weight': 0.0,
            'outlier_separation_weight': 0.0,
            'community_loss_weight': args.community_loss_weight,
            'pull_weight': args.pull_weight,
            'push_weight': args.push_weight,
            'bce_weight': 0.0,
            'con_weight': args.con_weight,
            'gui_weight': args.gui_weight
        }
    else:
        # 超过warmup后，使用线性插值平滑地恢复到目标值
        progress = min(1.0, (epoch - warmup_epoch) / warmup_epoch)  # 在warmup_epoch个epoch内平滑过渡
        
        return {
            'rec_weight': progress * args.rec_weight,
            'perturbation_weight': progress * args.perturbation_weight,
            'local_perturbation_weight': progress * args.local_perturbation_weight,
            'neighbor_perturbation_weight': progress * args.neighbor_perturbation_weight,
            'normal_alignment_weight': progress * args.normal_alignment_weight,
            'outlier_separation_weight': progress * args.outlier_separation_weight,
            'community_loss_weight': args.community_loss_weight,
            'pull_weight': args.pull_weight,
            'push_weight': args.push_weight,
            'bce_weight': args.bce_weight,
            'con_weight': args.con_weight,
            'gui_weight': args.gui_weight
        }