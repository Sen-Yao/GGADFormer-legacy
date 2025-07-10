import torch.nn as nn

from model import GGADFormer
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='photo', choices=['Amazon', 't_finance', 'reddit', 'photo', 'elliptic'])
parser.add_argument('--lr', type=float)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0.0)
parser.add_argument('--var', type=float, default=0.0)

# GCFormer parameters
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--ffn_dim', type=int, default=64)
parser.add_argument('--dropout', type=int, default=0.5)
parser.add_argument('--attention_dropout', type=int, default=0.5)
parser.add_argument('--peak_lr', type=float, default=0.005,
                        help='learning rate')
parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--sample_num_p', type=int, default=7,
                        help='Number of node to be sampled')
parser.add_argument('--sample_num_n', type=int, default=7,
                        help='Number of node to be sampled')
parser.add_argument('--pp_k', type=int, default=3,
                        help='propagation steps')
parser.add_argument('--sample_size', type=int, default=50000,
                        help='Batch size')
parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
parser.add_argument('--tot_updates', type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
parser.add_argument('--alpha', type=float, default=1,
                        help='aggregation weight')
parser.add_argument('--temp', type=float, default=1,
                        help='temperature')

# GGADFormer parameters

parser.add_argument('--alpha_outlier_generation', type=float, default=2,
                        help='alpha used in outlier generation')
parser.add_argument('--topk_neighbors_attention', type=float, default=10,
                        help='topk neighbors attention')
parser.add_argument('--device', type=int, default=0, help='Chose the device to run the model on')

# community parameters
parser.add_argument('--community_embedding_dim', type=int, default=32, help='Dimension of the community embedding')


args = parser.parse_args()
if args.device >= 0 and torch.cuda.is_available():
        args.device = f'cuda:{args.device}'
elif args.device == -1:
    args.device = 'cpu'
    np.random.seed(args.seed)
    print(f"警告: CUDA 不可用或指定的设备索引无效 (device index: {args.device})，将使用 CPU。")
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
        

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['t_finance']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['photo']:
        args.num_epoch = 200
    if args.dataset in ['elliptic']:
        args.num_epoch = 150
    if args.dataset in ['reddit']:
        args.num_epoch = 300
    elif args.dataset in ['t_finance']:
        args.num_epoch = 500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
if args.dataset in ['reddit', 'photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0


print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, all_normal_idx, sample_normal_idx = load_mat(args.dataset)


# ano_label 为异常节点标签
# str_ano_label 为结构导致的异常
# attr_ano_label 为属性导致的异常
# all_normal_idx 为正常节点索引
# sample_normal_idx 为采样的用于生成离群点的正常节点索引

if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

print("Training on device:", args.device)
dgl_graph = adj_to_dgl_graph(adj, args.dataset)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
# adj = torch.FloatTensor(adj[np.newaxis])
features = torch.FloatTensor(features)
adj = torch.FloatTensor(adj)
# adj = adj.to_sparse_csr()
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# Community Training

COMMUNITY_AE_EPOCHS = 1000
COMMUNITY_AE_LR = 1e-3
COMMUNITY_AE_HIDDEN_DIMS = [128, 64] # Example: Two hidden layers for autoencoder
print("Starting Community Detection Module setup...")
community_H, community_ae_model = train_community_detection_module(
    adj_original=raw_adj,  # Pass your original adj matrix here
    device=args.device,
    epochs=COMMUNITY_AE_EPOCHS,
    lr=COMMUNITY_AE_LR,
    hidden_dims=COMMUNITY_AE_HIDDEN_DIMS,
    community_embedding_dim=args.community_embedding_dim,
)

# Initialize model and optimiser
model = GGADFormer(ft_size, args.hidden_dim, 'prelu', args.negsamp_ratio, args.readout, args)
processed_features = preprocess_sample_features(args, features.squeeze(0), adj.squeeze(0))
processed_features = processed_features.to(args.device)

# For GT only
# Only use the 0-hop and the last hop features
prop_seq1 = node_neighborhood_feature(adj.squeeze(0), features.squeeze(0), args.pp_k).to(args.device).unsqueeze(0)
processed_seq1 = torch.concat((features.to(args.device), prop_seq1), dim=2)
# Add community embedding to the processed features
community_H_reshaped = community_H.unsqueeze(0) # -> (1, num_nodes, community_embedding_dim)
processed_seq1 = torch.concat((processed_seq1, community_H.unsqueeze(0)), dim=2)

# Disable Matrix Multiplication for ablation study
# processed_seq1 = features.to(args.device)

optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
lr_scheduler = PolynomialDecayLR(
    optimizer,
    warmup_updates=args.warmup_updates,
    tot_updates=args.tot_updates,
    lr=args.peak_lr,
    end_lr=args.end_lr,
    power=1.0,
)


#
# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     raw_adj = raw_adj.cuda()

# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()
#
# if torch.cuda.is_available():
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
# else:
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio])).to(args.device)
xent = nn.CrossEntropyLoss().to(args.device)


# Train model
pbar = tqdm(range(args.num_epoch), desc='Training')
total_time = 0
last_auc = 0.0 # 初始化 last_auc，确保在第一次 test_gap 循环时能正常使用
last_AP = 0.0 # 初始化 last_AP

records = {
    'loss_margin': [],
    'loss_bce': [],
    'loss_rec': [],
    'con_loss': [],
    'total_loss': [],
    'test_AUC': [],
    'test_AP': [],
    'best_test_auc': 0.0,
    'best_test_AP': 0.0,
    'best_test_epoch': 0
}

test_gap = 5

for epoch in pbar:
    start_time = time.time()
    model.train()
    optimiser.zero_grad()

    # Train model
    train_flag = True
    emb, emb_combine, logits, emb_con, emb_abnormal, con_loss, gui_loss = model(features, processed_seq1, adj,
                                                            sample_normal_idx, all_normal_idx, community_H,
                                                            train_flag, args)
    # BCE loss
    lbl = torch.unsqueeze(torch.cat(
        (torch.zeros(len(all_normal_idx)), torch.ones(len(emb_con)))),
        1).unsqueeze(0).to(args.device)
    # print(f"lbl.shape: {lbl.shape}, logits.shape: {logits.shape}, emb_con.shape: {emb_con.shape}, emb_abnormal.shape: {emb_abnormal.shape}")
    # if torch.cuda.is_available():
    #     lbl = lbl.cuda()
    loss_bce = b_xent(logits, lbl)
    loss_bce = torch.mean(loss_bce)

    # Local affinity margin loss
    emb = torch.squeeze(emb)

    emb_inf = torch.norm(emb, dim=-1, keepdim=True)
    emb_inf = torch.pow(emb_inf, -1)
    emb_inf[torch.isinf(emb_inf)] = 0.
    emb_norm = emb * emb_inf

    sim_matrix = torch.mm(emb_norm, emb_norm.T)
    raw_adj = torch.squeeze(raw_adj).to(args.device)
    similar_matrix = sim_matrix * raw_adj

    r_inv = torch.pow(torch.sum(raw_adj, 0), -1)
    r_inv[torch.isinf(r_inv)] = 0.
    affinity = torch.sum(similar_matrix, 0) * r_inv

    affinity_normal_mean = torch.mean(affinity[all_normal_idx])
    affinity_abnormal_mean = torch.mean(affinity[sample_normal_idx])

    confidence_margin = 0.7
    # 论文里的 loss ala
    loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

    diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
    # 论文里的 EC loss
    loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

    # For ablation study, set con_loss to zero
    # con_loss = torch.zeros_like(con_loss).to(args.device)

    loss = 0 * loss_margin + 1 * loss_bce + 1 * loss_rec + 1 * con_loss + 0.02 * gui_loss

    loss.backward()
    optimiser.step()
    end_time = time.time()
    total_time += end_time - start_time
    
    # if epoch % 2 == 0:
    if False:
        logits = np.squeeze(logits.cpu().detach().numpy())
        lbl = np.squeeze(lbl.cpu().detach().numpy())
        auc = roc_auc_score(lbl, logits)
        # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
        # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
        # print('Traininig AP:', AP)

        auc_train = roc_auc_score(lbl, logits)
        # 可以在这里更新 tqdm 的后缀信息，或者单独打印
        # pbar.set_postfix({..., 'train_auc': f'{auc_train:.4f}'})
        # 暂时保留打印，以便你能看到完整的日志
        print(f"\nEpoch: {epoch:04d} Training AUC: {auc_train:.4f}")
        print("=====================================================================")
    if epoch % test_gap == 0:
        model.eval()
        train_flag = False
        emb, emb_combine, logits, emb_con, emb_abnormal, con_loss_eval, gui_loss_eval = model(features, processed_seq1, adj, sample_normal_idx, all_normal_idx, community_H, 
                                                                train_flag, args)
        # evaluation on the valid and test node
        logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
        last_auc = roc_auc_score(ano_label[idx_test], logits)
        # print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
        last_AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
        # print('Testing AP:', AP)
        records['test_AUC'].append(last_auc.item())
        records['test_AP'].append(last_AP.item())
        if last_auc > records['best_test_auc']:
            records['best_test_auc'] = last_auc
            records['best_test_AP'] = last_AP
            records['best_test_auc_epoch'] = epoch
    pbar.set_postfix({
        'time': f'{total_time:.2f}s',
        'AUC': f'{last_auc.item():.5f}',
        'AP': f'{last_AP.item():.5f}'
    })
    records['loss_margin'].append(loss_margin.item())
    records['loss_bce'].append(loss_bce.item())
    records['loss_rec'].append(loss_rec.item())
    records['con_loss'].append(con_loss.item()) # SGT 返回的 con_loss
    records['total_loss'].append(loss.item())


results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.figure(figsize=(12, 8))
epochs = range(1, args.num_epoch + 1)

# Plotting loss trends
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(epochs, records['loss_margin'], label='Loss Margin')
plt.plot(epochs, records['loss_bce'], label='Loss BCE')
plt.plot(epochs, records['loss_rec'], label='Loss Rec')
plt.plot(epochs, records['con_loss'], label='Loss SGT')
plt.plot(epochs, records['total_loss'], label='Total Loss', linewidth=2, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title(f'Training Loss Trends for {args.dataset}')
plt.legend()
plt.grid(True)

# Plotting AUC trend
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
epochs_auc = range(test_gap, args.num_epoch + 1, test_gap)  # AUC is recorded every 10 epochs
plt.plot(epochs_auc, records['test_AUC'], label='Test AUC', color='green')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title(f'Test AUC Trend for {args.dataset}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'{args.dataset}_loss_and_auc_trends.png'))
plt.show()

print(f"Loss and AUC trend plot saved to '{os.path.join(results_dir, f'{args.dataset}_loss_and_auc_trends.png')}'")

print(f"Best Test AUC: {records['best_test_auc']:.5f}, AP: {records['best_test_AP']:.5f} at Epoch: {records['best_test_auc_epoch']}")
