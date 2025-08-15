import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import json
import os
import argparse
from tqdm import tqdm
import time
import random
import dgl

from model import GGADFormer
from utils import *

def analyze_distribution_differences(model, features, adj, sample_normal_idx, all_labeled_normal_idx, 
                                   community_H, ano_label, args, device):
    """
    分析真实异常点和生成离群点的分布差异
    """
    print("=== 分布差异分析 ===")
    
    # 获取真实异常点索引
    real_abnormal_idx = np.where(ano_label == 1)[0]
    real_normal_idx = np.where(ano_label == 0)[0]
    
    print(f"真实异常点数量: {len(real_abnormal_idx)}")
    print(f"真实正常点数量: {len(real_normal_idx)}")
    print(f"生成的离群点数量: {len(sample_normal_idx)}")
    
    # 获取嵌入
    model.eval()
    with torch.no_grad():
        emb, emb_combine, logits, emb_con, emb_abnormal, con_loss, gui_loss = model(
            features, adj, sample_normal_idx, all_labeled_normal_idx, community_H, False, args
        )
        
        # 获取真实异常点和正常点的嵌入
        real_abnormal_emb = emb[0, real_abnormal_idx, :].cpu().numpy()
        real_normal_emb = emb[0, real_normal_idx, :].cpu().numpy()
        generated_abnormal_emb = emb_abnormal[0, :, :].cpu().numpy()
        
    print(f"嵌入维度: {real_abnormal_emb.shape[1]}")
    
    # 计算统计特性
    print("\n=== 基本统计量 ===")
    
    # L2范数
    real_abnormal_norms = np.linalg.norm(real_abnormal_emb, axis=1)
    real_normal_norms = np.linalg.norm(real_normal_emb, axis=1)
    generated_norms = np.linalg.norm(generated_abnormal_emb, axis=1)
    
    print(f"真实异常点L2范数 - 均值: {np.mean(real_abnormal_norms):.4f}, 标准差: {np.std(real_abnormal_norms):.4f}")
    print(f"真实正常点L2范数 - 均值: {np.mean(real_normal_norms):.4f}, 标准差: {np.std(real_normal_norms):.4f}")
    print(f"生成离群点L2范数 - 均值: {np.mean(generated_norms):.4f}, 标准差: {np.std(generated_norms):.4f}")
    
    # 与正常点的距离
    normal_center = np.mean(real_normal_emb, axis=0)
    real_abnormal_distances = np.linalg.norm(real_abnormal_emb - normal_center, axis=1)
    generated_distances = np.linalg.norm(generated_abnormal_emb - normal_center, axis=1)
    
    print(f"\n=== 与正常点的距离 ===")
    print(f"生成离群点到正常中心距离 - 均值: {np.mean(generated_distances):.4f}, 标准差: {np.std(generated_distances):.4f}")
    print(f"真实异常点到正常中心距离 - 均值: {np.mean(real_abnormal_distances):.4f}, 标准差: {np.std(real_abnormal_distances):.4f}")
    
    # 相似度分析
    print(f"\n=== 相似度分析 ===")
    
    # 计算与正常点的余弦相似度
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 随机采样一些正常点进行相似度计算
    sample_size = min(100, len(real_normal_emb))
    sampled_normal_emb = real_normal_emb[np.random.choice(len(real_normal_emb), sample_size, replace=False)]
    
    generated_similarities = []
    real_abnormal_similarities = []
    
    for i in range(len(generated_abnormal_emb)):
        similarities = [cosine_similarity(generated_abnormal_emb[i], normal_emb) for normal_emb in sampled_normal_emb]
        generated_similarities.append(np.mean(similarities))
    
    for i in range(len(real_abnormal_emb)):
        similarities = [cosine_similarity(real_abnormal_emb[i], normal_emb) for normal_emb in sampled_normal_emb]
        real_abnormal_similarities.append(np.mean(similarities))
    
    print(f"生成离群点与真实正常点相似度 - 均值: {np.mean(generated_similarities):.4f}, 标准差: {np.std(generated_similarities):.4f}")
    print(f"真实异常点与真实正常点相似度 - 均值: {np.mean(real_abnormal_similarities):.4f}, 标准差: {np.std(real_abnormal_similarities):.4f}")
    
    # 计算生成离群点与真实异常点的相似度
    generated_to_real_similarities = []
    for i in range(len(generated_abnormal_emb)):
        similarities = [cosine_similarity(generated_abnormal_emb[i], real_abnormal_emb[j]) for j in range(len(real_abnormal_emb))]
        generated_to_real_similarities.append(np.mean(similarities))
    
    print(f"生成离群点与真实异常点相似度 - 均值: {np.mean(generated_to_real_similarities):.4f}, 标准差: {np.std(generated_to_real_similarities):.4f}")
    
    # 统计检验
    print(f"\n=== 统计检验 ===")
    
    # L2范数t检验
    t_stat_norm, p_value_norm = stats.ttest_ind(real_abnormal_norms, generated_norms)
    print(f"L2范数t检验 - t={t_stat_norm:.4f}, p={p_value_norm:.4f}")
    
    # 距离t检验
    t_stat_dist, p_value_dist = stats.ttest_ind(real_abnormal_distances, generated_distances)
    print(f"距离t检验 - t={t_stat_dist:.4f}, p={p_value_dist:.4f}")
    
    # 相似度t检验
    t_stat_sim, p_value_sim = stats.ttest_ind(real_abnormal_similarities, generated_similarities)
    print(f"相似度t检验 - t={t_stat_sim:.4f}, p={p_value_sim:.4f}")
    
    # 判断是否存在显著差异
    significant_difference = p_value_norm < 0.05 or p_value_dist < 0.05 or p_value_sim < 0.05
    
    print(f"\n=== 分析总结 ===")
    print(f"L2范数是否存在显著差异: {'是' if p_value_norm < 0.05 else '否'}")
    print(f"距离是否存在显著差异: {'是' if p_value_dist < 0.05 else '否'}")
    print(f"相似度是否存在显著差异: {'是' if p_value_sim < 0.05 else '否'}")
    print(f"与正常点相似度差异: {abs(np.mean(real_abnormal_similarities) - np.mean(generated_similarities)):.4f}")
    
    if significant_difference:
        print("⚠️  发现显著差异！生成的离群点与真实异常点分布确实不同。")
        print("建议：")
        print("1. 调整噪声参数 (args.var, args.mean)")
        print("2. 改进离群点生成策略")
        print("3. 使用更复杂的生成方法")
    else:
        print("✅ 未发现显著差异！生成的离群点与真实异常点分布相似。")
    
    # 保存结果
    results = {
        'real_abnormal_norm_mean': float(np.mean(real_abnormal_norms)),
        'real_abnormal_norm_std': float(np.std(real_abnormal_norms)),
        'generated_norm_mean': float(np.mean(generated_norms)),
        'generated_norm_std': float(np.std(generated_norms)),
        'real_abnormal_distance_mean': float(np.mean(real_abnormal_distances)),
        'generated_distance_mean': float(np.mean(generated_distances)),
        'real_abnormal_similarity_mean': float(np.mean(real_abnormal_similarities)),
        'generated_similarity_mean': float(np.mean(generated_similarities)),
        'p_value_norm': float(p_value_norm),
        'p_value_dist': float(p_value_dist),
        'p_value_sim': float(p_value_sim),
        'significant_difference': bool(significant_difference)
    }
    
    with open('distribution_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n分析结果已保存到 distribution_analysis_results.json")
    
    return results, {
        'real_abnormal_emb': real_abnormal_emb,
        'real_normal_emb': real_normal_emb,
        'generated_abnormal_emb': generated_abnormal_emb,
        'real_abnormal_norms': real_abnormal_norms,
        'generated_norms': generated_norms,
        'real_abnormal_distances': real_abnormal_distances,
        'generated_distances': generated_distances
    }

def visualize_distributions(data_dict, save_dir='results'):
    """
    可视化分布差异
    """
    print("\n=== 生成可视化 ===")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. PCA可视化
    print("生成PCA可视化...")
    plt.figure(figsize=(12, 5))
    
    # 合并所有嵌入进行PCA
    all_embeddings = np.vstack([
        data_dict['real_abnormal_emb'][:100],  # 采样100个真实异常点
        data_dict['real_normal_emb'][:100],    # 采样100个真实正常点
        data_dict['generated_abnormal_emb']    # 所有生成的离群点
    ])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings)
    
    # 绘制散点图
    real_abnormal_count = min(100, len(data_dict['real_abnormal_emb']))
    real_normal_count = min(100, len(data_dict['real_normal_emb']))
    generated_count = len(data_dict['generated_abnormal_emb'])
    
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:real_abnormal_count, 0], pca_result[:real_abnormal_count, 1], 
               c='red', alpha=0.6, label='真实异常点', s=20)
    plt.scatter(pca_result[real_abnormal_count:real_abnormal_count+real_normal_count, 0], 
               pca_result[real_abnormal_count:real_abnormal_count+real_normal_count, 1], 
               c='blue', alpha=0.6, label='真实正常点', s=20)
    plt.scatter(pca_result[real_abnormal_count+real_normal_count:, 0], 
               pca_result[real_abnormal_count+real_normal_count:, 1], 
               c='green', alpha=0.6, label='生成离群点', s=20)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA降维可视化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(data_dict['real_abnormal_norms'], bins=30, alpha=0.7, label='真实异常点', color='red', density=True)
    plt.hist(data_dict['generated_norms'], bins=30, alpha=0.7, label='生成离群点', color='green', density=True)
    plt.xlabel('L2范数')
    plt.ylabel('密度')
    plt.title('L2范数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"PCA可视化已保存为 {os.path.join(save_dir, 'distribution_visualization.png')}")
    
    # 3. 详细分布对比图
    print("生成详细分布对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # L2范数分布
    axes[0, 0].hist(data_dict['real_abnormal_norms'], bins=30, alpha=0.7, label='真实异常点', color='red', density=True)
    axes[0, 0].hist(data_dict['generated_norms'], bins=30, alpha=0.7, label='生成离群点', color='green', density=True)
    axes[0, 0].set_xlabel('L2范数')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('L2范数分布对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 距离分布
    axes[0, 1].hist(data_dict['real_abnormal_distances'], bins=30, alpha=0.7, label='真实异常点', color='red', density=True)
    axes[0, 1].hist(data_dict['generated_distances'], bins=30, alpha=0.7, label='生成离群点', color='green', density=True)
    axes[0, 1].set_xlabel('到正常中心的距离')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('距离分布对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 箱线图对比
    axes[1, 0].boxplot([data_dict['real_abnormal_norms'], data_dict['generated_norms']], 
                      labels=['真实异常点', '生成离群点'])
    axes[1, 0].set_ylabel('L2范数')
    axes[1, 0].set_title('L2范数箱线图对比')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 距离箱线图
    axes[1, 1].boxplot([data_dict['real_abnormal_distances'], data_dict['generated_distances']], 
                      labels=['真实异常点', '生成离群点'])
    axes[1, 1].set_ylabel('到正常中心的距离')
    axes[1, 1].set_title('距离箱线图对比')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"详细分布对比图已保存为 {os.path.join(save_dir, 'detailed_distribution_comparison.png')}")

def main():
    parser = argparse.ArgumentParser(description='分布差异分析')
    parser.add_argument('--dataset', type=str, default='photo', 
                       choices=['Amazon', 't_finance', 'reddit', 'photo', 'elliptic'])
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化')
    
    # 添加模型所需的参数
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--ffn_dim', type=int, default=64)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--attention_dropout', type=int, default=0.5)
    parser.add_argument('--peak_lr', type=float, default=0.005)
    parser.add_argument('--end_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--sample_num_p', type=int, default=7)
    parser.add_argument('--sample_num_n', type=int, default=7)
    parser.add_argument('--pp_k', type=int, default=3)
    parser.add_argument('--sample_size', type=int, default=50000)
    parser.add_argument('--warmup_updates', type=int, default=400)
    parser.add_argument('--tot_updates', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--alpha_outlier_generation', type=float, default=2)
    parser.add_argument('--topk_neighbors_attention', type=float, default=10)
    parser.add_argument('--outlier_margin', type=float, default=0.5)
    parser.add_argument('--community_embedding_dim', type=int, default=32)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--mean', type=float, default=0.02)
    parser.add_argument('--var', type=float, default=0.01)
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.device}'
    else:
        device = 'cpu'
        print(f"使用CPU进行分析")
    
    print(f"使用设备: {device}")
    print(f"数据集: {args.dataset}")
    
    # 加载数据
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, all_labeled_normal_idx, sample_normal_idx = load_mat(args.dataset)
    
    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    
    # 数据预处理
    raw_adj = adj
    adj = normalize_adj(adj)
    raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
    adj = (adj + sp.eye(adj.shape[0])).todense()
    
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    
    # 社区检测模块
    print("设置社区检测模块...")
    community_H, _ = train_community_detection_module(
        adj_original=raw_adj,
        device=device,
        dataset_name=args.dataset,
        epochs=1000,
        lr=1e-3,
        hidden_dims=[128, 64],
        community_embedding_dim=32,
    )
    
    # 准备输入特征
    prop_seq1 = node_neighborhood_feature(adj.squeeze(0), features.squeeze(0), 3).to(device).unsqueeze(0)
    concated_input_features = torch.concat((features.to(device), prop_seq1), dim=2)
    concated_input_features = torch.concat((concated_input_features, community_H.unsqueeze(0)), dim=2)
    
    # 初始化模型
    ft_size = features.shape[2]
    model = GGADFormer(ft_size, 64, 'prelu', 1, args)
    
    # 加载预训练模型
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载预训练模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型来自第 {checkpoint['epoch']} 轮，最佳AUC: {checkpoint['best_auc']:.5f}")
    else:
        print("未找到预训练模型，使用随机初始化的模型进行分析")
    
    model.to(device)
    
    # 进行分布差异分析
    results, data_dict = analyze_distribution_differences(
        model, concated_input_features, adj.to(device), 
        sample_normal_idx, all_labeled_normal_idx, community_H, 
        ano_label, args, device
    )
    
    # 生成可视化
    if args.visualize:
        visualize_distributions(data_dict)
    
    print("\n=== 分析完成 ===")
    print("结果已保存到 distribution_analysis_results.json")
    if args.visualize:
        print("可视化图表已保存到 results/ 目录")

if __name__ == "__main__":
    main()
