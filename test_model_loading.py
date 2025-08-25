#!/usr/bin/env python3
"""
测试最佳模型加载功能的脚本
"""

import torch
import os
import argparse
from model import GGADFormer

def test_model_loading(dataset_name='photo'):
    """
    测试最佳模型加载功能
    
    Args:
        dataset_name: 数据集名称
    """
    print(f"测试加载数据集 {dataset_name} 的最佳模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    best_model_path = f'best_model_{dataset_name}.pth'
    if not os.path.exists(best_model_path):
        print(f"错误: 模型文件 {best_model_path} 不存在!")
        return False
    
    print(f"找到模型文件: {best_model_path}")
    
    # 创建参数对象（模拟args）
    class Args:
        def __init__(self):
            self.dataset = dataset_name
            self.device = device
            self.hidden_dim = 64
            self.n_layers = 3
            self.n_heads = 1
            self.ffn_dim = 64
            self.dropout = 0.5
            self.attention_dropout = 0.5
            self.community_embedding_dim = 32
            self.alpha_outlier_generation = 2
            self.topk_neighbors_attention = 10
            self.outlier_margin = 0.5
            self.perturbation_weight = 2.0
            self.local_perturbation_weight = 1e-2
            self.neighbor_perturbation_weight = 0.0
            self.normal_alignment_weight = 1e-3
            self.outlier_separation_weight = 2e-3
            self.push_weight = 1e-3
            self.pull_weight = 1.0
            self.bce_weight = 1.0
            self.rec_weight = 1.0
            self.con_weight = 1.0
            self.community_loss_weight = 0.1
            self.mean = 0.0
            self.var = 0.0
            self.negsamp_ratio = 1
    
    args = Args()
    
    # 初始化模型
    ft_size = 64  # 假设特征维度为64
    model = GGADFormer(ft_size, args.hidden_dim, 'prelu', args)
    model = model.to(device)
    
    print("模型初始化完成")
    
    # 加载最佳模型
    try:
        print(f"正在加载最佳模型...")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # 加载主模型状态字典
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载社区自编码器状态字典（如果存在）
        if 'community_autoencoder_state_dict' in checkpoint:
            model.community_autoencoder.load_state_dict(checkpoint['community_autoencoder_state_dict'])
            print("✓ 成功加载社区自编码器状态字典")
        
        # 确保模型在正确的设备上
        model = model.to(device)
        
        print(f"✓ 成功加载最佳模型")
        print(f"  - 训练轮次: {checkpoint['epoch']}")
        print(f"  - 最佳AUC: {checkpoint['best_auc']:.5f}")
        print(f"  - 最佳AP: {checkpoint['best_ap']:.5f}")
        print(f"  - 模型设备: {next(model.parameters()).device}")
        
        # 测试模型前向传播
        print("测试模型前向传播...")
        model.eval()
        with torch.no_grad():
            # 创建模拟输入数据
            batch_size = 1
            num_nodes = 100
            hidden_dim = args.hidden_dim
            
            # 模拟输入特征
            concated_input_features = torch.randn(batch_size, num_nodes, ft_size * 2).to(device)
            adj = torch.randn(batch_size, num_nodes, num_nodes).to(device)
            sample_normal_idx = list(range(50))  # 前50个节点作为采样正常节点
            all_labeled_normal_idx = list(range(80))  # 前80个节点作为标记正常节点
            
            # 测试前向传播
            emb, emb_combine, logits, emb_con, emb_abnormal, con_loss, community_loss = model(
                concated_input_features, adj, sample_normal_idx, all_labeled_normal_idx, 
                train_flag=False, args=args
            )
            
            print("✓ 模型前向传播测试成功")
            print(f"  - 嵌入维度: {emb.shape}")
            print(f"  - 组合嵌入维度: {emb_combine.shape}")
            print(f"  - 输出维度: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 加载模型时出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试最佳模型加载功能')
    parser.add_argument('--dataset', type=str, default='photo', 
                       choices=['Amazon', 't_finance', 'reddit', 'photo', 'elliptic'],
                       help='数据集名称')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("最佳模型加载功能测试")
    print("=" * 60)
    
    success = test_model_loading(args.dataset)
    
    print("=" * 60)
    if success:
        print("✓ 测试通过！最佳模型加载功能正常工作")
    else:
        print("✗ 测试失败！请检查模型文件和代码")
    print("=" * 60)

if __name__ == "__main__":
    main()
