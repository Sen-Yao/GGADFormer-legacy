import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def visualize_outlier_generation_quality(model, concated_input_features, adj, sample_normal_idx, 
                                       all_labeled_normal_idx, ano_label, idx_test, 
                                       args, save_dir='results'):
    """
    Visualize the quality of generated outliers compared to real anomalies in embedding space.
    
    Args:
        model: Trained GGADFormer model
        concated_input_features: Input features with community embeddings
        adj: Adjacency matrix
        sample_normal_idx: Indices of sampled normal nodes for outlier generation
        all_labeled_normal_idx: Indices of all labeled normal nodes
        community_H: Community embeddings
        ano_label: True anomaly labels
        idx_test: Test node indices
        args: Model arguments
        save_dir: Directory to save visualization results
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings and generated outliers in training mode to access emb_combine
        train_flag = True
        emb, emb_combine, logits, emb_con, emb_abnormal, con_loss, community_loss = model(
            concated_input_features, adj.to(args.device), sample_normal_idx, 
            all_labeled_normal_idx, train_flag, args
        )
        
        # Use emb_combine for visualization instead of emb
        # emb_combine: [1, num_normal_nodes + len(sample_normal_idx), hidden_dim]
        emb_combine_np = emb_combine.squeeze(0).cpu().numpy()
        
        # Split emb_combine into normal nodes and generated outliers
        num_normal_nodes = len(all_labeled_normal_idx)
        normal_embeddings_combine = emb_combine_np[:num_normal_nodes]  # 正常节点在emb_combine中的嵌入
        generated_outliers = emb_combine_np[num_normal_nodes:]  # 生成的离群点在emb_combine中的嵌入
        
        # Get real anomaly embeddings from the original emb space
        # We need to map test indices to the original node indices
        train_flag = False
        emb_orig, _, _, _, _, _, _ = model(
            concated_input_features, adj.to(args.device), sample_normal_idx, 
            all_labeled_normal_idx, train_flag, args
        )
        all_embeddings_orig = emb_orig.squeeze(0).cpu().numpy()
        
        # Get different types of nodes from test set
        real_anomaly_idx = [i for i in idx_test if ano_label[i] == 1]
        real_normal_idx = [i for i in idx_test if ano_label[i] == 0]
        
        # Get real anomaly embeddings from original space
        real_anomaly_embeddings = all_embeddings_orig[real_anomaly_idx]
        
        print(f"Normal nodes in emb_combine: {len(normal_embeddings_combine)}")
        print(f"Real anomalies: {len(real_anomaly_idx)}")
        print(f"Generated outliers in emb_combine: {len(generated_outliers)}")
        
        # Create comprehensive visualizations using emb_combine space
        create_embedding_visualizations_emb_combine(
            normal_embeddings_combine, real_anomaly_embeddings, generated_outliers,
            args.dataset, save_dir
        )
        
        # Analyze embedding quality in emb_combine space
        analyze_embedding_quality_emb_combine(
            normal_embeddings_combine, real_anomaly_embeddings, generated_outliers,
            args.dataset, save_dir
        )
        
        print(f"Visualization results saved to {save_dir}")


def create_embedding_visualizations(normal_embeddings, real_anomaly_embeddings, 
                                  generated_outliers, dataset_name, save_dir):
    """
    Create multiple types of embedding visualizations.
    """
    
    # Combine all embeddings for dimensionality reduction
    all_embeddings = np.vstack([normal_embeddings, real_anomaly_embeddings, generated_outliers])
    
    # Create labels for coloring
    labels = (['Normal'] * len(normal_embeddings) + 
             ['Real Anomaly'] * len(real_anomaly_embeddings) + 
             ['Generated Outlier'] * len(generated_outliers))
    
    # 1. t-SNE Visualization
    print("Creating t-SNE visualization...")
    create_tsne_visualization(all_embeddings, labels, dataset_name, save_dir)
    
    # 2. PCA Visualization
    print("Creating PCA visualization...")
    create_pca_visualization(all_embeddings, labels, dataset_name, save_dir)
    
    # 3. Distance Distribution Analysis
    print("Creating distance distribution analysis...")
    create_distance_analysis(normal_embeddings, real_anomaly_embeddings, 
                           generated_outliers, dataset_name, save_dir)
    
    # 4. Embedding Statistics Comparison
    print("Creating embedding statistics comparison...")
    create_statistics_comparison(normal_embeddings, real_anomaly_embeddings, 
                               generated_outliers, dataset_name, save_dir)


def create_tsne_visualization(all_embeddings, labels, dataset_name, save_dir):
    """
    Create t-SNE visualization of embeddings.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//4))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = {'Normal': '#1f77b4', 'Real Anomaly': '#ff7f0e', 'Generated Outlier': '#2ca02c'}
    markers = {'Normal': 'o', 'Real Anomaly': '^', 'Generated Outlier': 's'}
    
    # Plot each type
    for label_type in ['Normal', 'Real Anomaly', 'Generated Outlier']:
        mask = np.array(labels) == label_type
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[label_type], marker=markers[label_type], 
                       label=label_type, alpha=0.7, s=60)
    
    plt.title(f't-SNE Visualization of Embedding Space - {dataset_name}', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_tsne_embedding_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_pca_visualization(all_embeddings, labels, dataset_name, save_dir):
    """
    Create PCA visualization of embeddings.
    """
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = {'Normal': '#1f77b4', 'Real Anomaly': '#ff7f0e', 'Generated Outlier': '#2ca02c'}
    markers = {'Normal': 'o', 'Real Anomaly': '^', 'Generated Outlier': 's'}
    
    # Plot each type
    for label_type in ['Normal', 'Real Anomaly', 'Generated Outlier']:
        mask = np.array(labels) == label_type
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[label_type], marker=markers[label_type], 
                       label=label_type, alpha=0.7, s=60)
    
    plt.title(f'PCA Visualization of Embedding Space - {dataset_name}', fontsize=16)
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_pca_embedding_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_distance_analysis(normal_embeddings, real_anomaly_embeddings, 
                           generated_outliers, dataset_name, save_dir):
    """
    Analyze and visualize distance distributions between different node types.
    """
    # Calculate centroids
    normal_centroid = np.mean(normal_embeddings, axis=0)
    
    # Calculate distances to normal centroid
    normal_distances = np.linalg.norm(normal_embeddings - normal_centroid, axis=1)
    real_anomaly_distances = np.linalg.norm(real_anomaly_embeddings - normal_centroid, axis=1)
    generated_distances = np.linalg.norm(generated_outliers - normal_centroid, axis=1)
    
    # Create distance distribution plot
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    plt.hist(normal_distances, bins=30, alpha=0.7, label='Normal Nodes', 
             color='#1f77b4', density=True)
    plt.hist(real_anomaly_distances, bins=30, alpha=0.7, label='Real Anomalies', 
             color='#ff7f0e', density=True)
    plt.hist(generated_distances, bins=30, alpha=0.7, label='Generated Outliers', 
             color='#2ca02c', density=True)
    
    plt.title(f'Distance Distribution from Normal Centroid - {dataset_name}', fontsize=16)
    plt.xlabel('Distance from Normal Centroid', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_distance_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nDistance Statistics for {dataset_name}:")
    print(f"Normal nodes - Mean: {np.mean(normal_distances):.4f}, Std: {np.std(normal_distances):.4f}")
    print(f"Real anomalies - Mean: {np.mean(real_anomaly_distances):.4f}, Std: {np.std(real_anomaly_distances):.4f}")
    print(f"Generated outliers - Mean: {np.mean(generated_distances):.4f}, Std: {np.std(generated_distances):.4f}")


def create_statistics_comparison(normal_embeddings, real_anomaly_embeddings, 
                               generated_outliers, dataset_name, save_dir):
    """
    Create comprehensive statistics comparison between different node types.
    """
    # Calculate various statistics
    stats = {}
    
    for name, embeddings in [('Normal', normal_embeddings), 
                            ('Real Anomaly', real_anomaly_embeddings), 
                            ('Generated Outlier', generated_outliers)]:
        stats[name] = {
            'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
            'mean_embedding': np.mean(embeddings, axis=0),
            'std_embedding': np.std(embeddings, axis=0),
            'dimension_variance': np.var(embeddings, axis=0)
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Norm comparison
    norms = [stats[name]['mean_norm'] for name in ['Normal', 'Real Anomaly', 'Generated Outlier']]
    norm_stds = [stats[name]['std_norm'] for name in ['Normal', 'Real Anomaly', 'Generated Outlier']]
    
    axes[0, 0].bar(['Normal', 'Real Anomaly', 'Generated Outlier'], norms, 
                   yerr=norm_stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Mean Embedding Norm Comparison')
    axes[0, 0].set_ylabel('Mean L2 Norm')
    
    # 2. Dimension-wise variance comparison
    dim_vars_normal = stats['Normal']['dimension_variance']
    dim_vars_real = stats['Real Anomaly']['dimension_variance']
    dim_vars_generated = stats['Generated Outlier']['dimension_variance']
    
    x_dims = range(min(50, len(dim_vars_normal)))  # Show first 50 dimensions
    axes[0, 1].plot(x_dims, dim_vars_normal[:len(x_dims)], 'o-', label='Normal', alpha=0.7)
    axes[0, 1].plot(x_dims, dim_vars_real[:len(x_dims)], '^-', label='Real Anomaly', alpha=0.7)
    axes[0, 1].plot(x_dims, dim_vars_generated[:len(x_dims)], 's-', label='Generated Outlier', alpha=0.7)
    axes[0, 1].set_title('Dimension-wise Variance Comparison')
    axes[0, 1].set_xlabel('Embedding Dimension')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].legend()
    
    # 3. Pairwise distance analysis
    def calculate_intra_distances(embeddings):
        if len(embeddings) > 1:
            distances = pdist(embeddings)
            return np.mean(distances), np.std(distances)
        return 0, 0
    
    intra_distances = {}
    for name, embeddings in [('Normal', normal_embeddings), 
                            ('Real Anomaly', real_anomaly_embeddings), 
                            ('Generated Outlier', generated_outliers)]:
        mean_dist, std_dist = calculate_intra_distances(embeddings)
        intra_distances[name] = {'mean': mean_dist, 'std': std_dist}
    
    names = list(intra_distances.keys())
    means = [intra_distances[name]['mean'] for name in names]
    stds = [intra_distances[name]['std'] for name in names]
    
    axes[1, 0].bar(names, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_title('Intra-group Distance Comparison')
    axes[1, 0].set_ylabel('Mean Pairwise Distance')
    
    # 4. Inter-group distance analysis
    normal_centroid = np.mean(normal_embeddings, axis=0)
    real_centroid = np.mean(real_anomaly_embeddings, axis=0)
    generated_centroid = np.mean(generated_outliers, axis=0)
    
    inter_distances = {
        'Normal-Real': np.linalg.norm(normal_centroid - real_centroid),
        'Normal-Generated': np.linalg.norm(normal_centroid - generated_centroid),
        'Real-Generated': np.linalg.norm(real_centroid - generated_centroid)
    }
    
    axes[1, 1].bar(inter_distances.keys(), inter_distances.values(), 
                   color=['#d62728', '#9467bd', '#8c564b'])
    axes[1, 1].set_title('Inter-group Centroid Distance')
    axes[1, 1].set_ylabel('Distance between Centroids')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_statistics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def analyze_embedding_quality(normal_embeddings, real_anomaly_embeddings, 
                            generated_outliers, dataset_name, save_dir):
    """
    Perform quantitative analysis of embedding quality.
    """
    print(f"\n=== Embedding Quality Analysis for {dataset_name} ===")
    
    # 1. Silhouette Analysis
    if len(real_anomaly_embeddings) > 0 and len(generated_outliers) > 0:
        # Combine embeddings and create labels
        all_embeddings = np.vstack([normal_embeddings, real_anomaly_embeddings, generated_outliers])
        labels = ([0] * len(normal_embeddings) + 
                 [1] * len(real_anomaly_embeddings) + 
                 [2] * len(generated_outliers))
        
        try:
            silhouette_avg = silhouette_score(all_embeddings, labels)
            print(f"Overall Silhouette Score: {silhouette_avg:.4f}")
        except:
            print("Could not calculate silhouette score")
    
    # 2. Separation Analysis
    normal_centroid = np.mean(normal_embeddings, axis=0)
    
    # Calculate average distances from normal centroid
    normal_dist = np.mean(np.linalg.norm(normal_embeddings - normal_centroid, axis=1))
    real_dist = np.mean(np.linalg.norm(real_anomaly_embeddings - normal_centroid, axis=1))
    generated_dist = np.mean(np.linalg.norm(generated_outliers - normal_centroid, axis=1))
    
    print(f"Average distance from normal centroid:")
    print(f"  Normal nodes: {normal_dist:.4f}")
    print(f"  Real anomalies: {real_dist:.4f}")
    print(f"  Generated outliers: {generated_dist:.4f}")
    
    # 3. Quality Metrics
    separation_ratio_real = real_dist / normal_dist if normal_dist > 0 else 0
    separation_ratio_generated = generated_dist / normal_dist if normal_dist > 0 else 0
    
    print(f"Separation ratios (higher is better for anomalies):")
    print(f"  Real anomalies: {separation_ratio_real:.4f}")
    print(f"  Generated outliers: {separation_ratio_generated:.4f}")
    
    # 4. Similarity between real and generated anomalies
    if len(real_anomaly_embeddings) > 0 and len(generated_outliers) > 0:
        real_centroid = np.mean(real_anomaly_embeddings, axis=0)
        generated_centroid = np.mean(generated_outliers, axis=0)
        
        centroid_similarity = np.dot(real_centroid, generated_centroid) / (
            np.linalg.norm(real_centroid) * np.linalg.norm(generated_centroid))
        centroid_distance = np.linalg.norm(real_centroid - generated_centroid)
        
        print(f"Real vs Generated anomaly centroids:")
        print(f"  Cosine similarity: {centroid_similarity:.4f}")
        print(f"  Euclidean distance: {centroid_distance:.4f}")
    
    # Save analysis results
    analysis_results = {
        'dataset': dataset_name,
        'normal_centroid_distance': normal_dist,
        'real_anomaly_centroid_distance': real_dist,
        'generated_outlier_centroid_distance': generated_dist,
        'separation_ratio_real': separation_ratio_real,
        'separation_ratio_generated': separation_ratio_generated,
    }
    
    # Save to file
    import json
    with open(os.path.join(save_dir, f'{dataset_name}_quality_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Quality analysis saved to {save_dir}")


def load_best_model_and_visualize(args, model, concated_input_features, adj, sample_normal_idx, 
                                 all_labeled_normal_idx, ano_label, idx_test, 
                                 records, save_dir='results'):
    """
    Load the best performing model from training records and create visualizations.
    
    Args:
        args: Model arguments
        model: Model instance
        concated_input_features: Input features
        adj: Adjacency matrix
        sample_normal_idx: Sample normal indices
        all_labeled_normal_idx: All labeled normal indices
        community_H: Community embeddings
        ano_label: Anomaly labels
        idx_test: Test indices
        records: Training records containing best epoch info
        save_dir: Directory to save results
    """
    
    print(f"\n=== Outlier Generation Quality Visualization ===")
    print(f"Best model from epoch {records.get('best_test_auc_epoch', 'unknown')}")
    print(f"Best AUC: {records.get('best_test_auc', 'unknown'):.5f}")
    print(f"Best AP: {records.get('best_test_AP', 'unknown'):.5f}")
    
    # 尝试加载最佳模型
    best_model_path = f'best_model_{args.dataset}.pth'
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=args.device)
        
        # 加载主模型状态字典
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载社区自编码器状态字典（如果存在）
        if 'community_autoencoder_state_dict' in checkpoint:
            model.community_autoencoder.load_state_dict(checkpoint['community_autoencoder_state_dict'])
            print("Successfully loaded community autoencoder state dict")
        
        # 确保模型在正确的设备上
        model = model.to(args.device)
        
        print(f"Successfully loaded best model from epoch {checkpoint['epoch']}")
        print(f"Model AUC: {checkpoint['best_auc']:.5f}, AP: {checkpoint['best_ap']:.5f}")
        print(f"Model loaded on device: {args.device}")
    else:
        print(f"Warning: Best model file {best_model_path} not found. Using current model state.")
    
    # Create visualization
    visualize_outlier_generation_quality(
        model, concated_input_features, adj, sample_normal_idx,
        all_labeled_normal_idx, ano_label, idx_test,
        args, save_dir
    )
    
    print(f"Visualization completed and saved to {save_dir}")


def create_embedding_visualizations_emb_combine(normal_embeddings_combine, real_anomaly_embeddings, 
                                              generated_outliers, dataset_name, save_dir):
    """
    Create multiple types of embedding visualizations for emb_combine space.
    """
    
    # Combine all embeddings for dimensionality reduction
    all_embeddings = np.vstack([normal_embeddings_combine, real_anomaly_embeddings, generated_outliers])
    
    # Create labels for coloring
    labels = (['Normal (emb_combine)'] * len(normal_embeddings_combine) + 
             ['Real Anomaly'] * len(real_anomaly_embeddings) + 
             ['Generated Outlier (emb_combine)'] * len(generated_outliers))
    
    # 1. t-SNE Visualization
    print("Creating t-SNE visualization for emb_combine space...")
    create_tsne_visualization_emb_combine(all_embeddings, labels, dataset_name, save_dir)
    
    # 2. PCA Visualization
    print("Creating PCA visualization for emb_combine space...")
    create_pca_visualization_emb_combine(all_embeddings, labels, dataset_name, save_dir)
    
    # 3. Distance Distribution Analysis
    print("Creating distance distribution analysis for emb_combine space...")
    create_distance_analysis_emb_combine(normal_embeddings_combine, real_anomaly_embeddings, 
                                       generated_outliers, dataset_name, save_dir)
    
    # 4. Embedding Statistics Comparison
    print("Creating embedding statistics comparison for emb_combine space...")
    create_statistics_comparison_emb_combine(normal_embeddings_combine, real_anomaly_embeddings, 
                                           generated_outliers, dataset_name, save_dir)


def create_tsne_visualization_emb_combine(all_embeddings, labels, dataset_name, save_dir):
    """
    Create t-SNE visualization of emb_combine embeddings.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//4))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = {'Normal (emb_combine)': '#1f77b4', 'Real Anomaly': '#ff7f0e', 'Generated Outlier (emb_combine)': '#2ca02c'}
    markers = {'Normal (emb_combine)': 'o', 'Real Anomaly': '^', 'Generated Outlier (emb_combine)': 's'}
    
    # Plot each type
    for label_type in ['Normal (emb_combine)', 'Real Anomaly', 'Generated Outlier (emb_combine)']:
        mask = np.array(labels) == label_type
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[label_type], marker=markers[label_type], 
                       label=label_type, alpha=0.7, s=60)
    
    plt.title(f't-SNE Visualization of emb_combine Embedding Space - {dataset_name}', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_tsne_emb_combine_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_pca_visualization_emb_combine(all_embeddings, labels, dataset_name, save_dir):
    """
    Create PCA visualization of emb_combine embeddings.
    """
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers
    colors = {'Normal (emb_combine)': '#1f77b4', 'Real Anomaly': '#ff7f0e', 'Generated Outlier (emb_combine)': '#2ca02c'}
    markers = {'Normal (emb_combine)': 'o', 'Real Anomaly': '^', 'Generated Outlier (emb_combine)': 's'}
    
    # Plot each type
    for label_type in ['Normal (emb_combine)', 'Real Anomaly', 'Generated Outlier (emb_combine)']:
        mask = np.array(labels) == label_type
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[label_type], marker=markers[label_type], 
                       label=label_type, alpha=0.7, s=60)
    
    plt.title(f'PCA Visualization of emb_combine Embedding Space - {dataset_name}', fontsize=16)
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_pca_emb_combine_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_distance_analysis_emb_combine(normal_embeddings_combine, real_anomaly_embeddings, 
                                       generated_outliers, dataset_name, save_dir):
    """
    Analyze and visualize distance distributions between different node types in emb_combine space.
    """
    # Calculate centroids
    normal_centroid = np.mean(normal_embeddings_combine, axis=0)
    
    # Calculate distances to normal centroid
    normal_distances = np.linalg.norm(normal_embeddings_combine - normal_centroid, axis=1)
    real_anomaly_distances = np.linalg.norm(real_anomaly_embeddings - normal_centroid, axis=1)
    generated_distances = np.linalg.norm(generated_outliers - normal_centroid, axis=1)
    
    # Create distance distribution plot
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Distance distributions
    plt.subplot(1, 3, 1)
    plt.hist(normal_distances, bins=30, alpha=0.7, label='Normal (emb_combine)', color='#1f77b4')
    plt.hist(real_anomaly_distances, bins=30, alpha=0.7, label='Real Anomaly', color='#ff7f0e')
    plt.hist(generated_distances, bins=30, alpha=0.7, label='Generated Outlier (emb_combine)', color='#2ca02c')
    plt.xlabel('Distance to Normal Centroid')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution to Normal Centroid')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    plt.subplot(1, 3, 2)
    data_to_plot = [normal_distances, real_anomaly_distances, generated_distances]
    labels = ['Normal\n(emb_combine)', 'Real\nAnomaly', 'Generated\nOutlier\n(emb_combine)']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Distance to Normal Centroid')
    plt.title('Distance Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Violin plot
    plt.subplot(1, 3, 3)
    plt.violinplot(data_to_plot, positions=[1, 2, 3])
    plt.xticks([1, 2, 3], labels)
    plt.ylabel('Distance to Normal Centroid')
    plt.title('Distance Distribution (Violin Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_distance_analysis_emb_combine.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nDistance Analysis in emb_combine space:")
    print(f"Normal (emb_combine) - Mean: {np.mean(normal_distances):.4f}, Std: {np.std(normal_distances):.4f}")
    print(f"Real Anomaly - Mean: {np.mean(real_anomaly_distances):.4f}, Std: {np.std(real_anomaly_distances):.4f}")
    print(f"Generated Outlier (emb_combine) - Mean: {np.mean(generated_distances):.4f}, Std: {np.std(generated_distances):.4f}")


def create_statistics_comparison_emb_combine(normal_embeddings_combine, real_anomaly_embeddings, 
                                           generated_outliers, dataset_name, save_dir):
    """
    Compare statistical properties of different embedding types in emb_combine space.
    """
    # Calculate statistics
    stats = {}
    
    # Normal embeddings in emb_combine
    stats['Normal (emb_combine)'] = {
        'mean': np.mean(normal_embeddings_combine, axis=0),
        'std': np.std(normal_embeddings_combine, axis=0),
        'min': np.min(normal_embeddings_combine, axis=0),
        'max': np.max(normal_embeddings_combine, axis=0)
    }
    
    # Real anomaly embeddings
    stats['Real Anomaly'] = {
        'mean': np.mean(real_anomaly_embeddings, axis=0),
        'std': np.std(real_anomaly_embeddings, axis=0),
        'min': np.min(real_anomaly_embeddings, axis=0),
        'max': np.max(real_anomaly_embeddings, axis=0)
    }
    
    # Generated outlier embeddings in emb_combine
    stats['Generated Outlier (emb_combine)'] = {
        'mean': np.mean(generated_outliers, axis=0),
        'std': np.std(generated_outliers, axis=0),
        'min': np.min(generated_outliers, axis=0),
        'max': np.max(generated_outliers, axis=0)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean comparison
    axes[0, 0].plot(stats['Normal (emb_combine)']['mean'], label='Normal (emb_combine)', alpha=0.8)
    axes[0, 0].plot(stats['Real Anomaly']['mean'], label='Real Anomaly', alpha=0.8)
    axes[0, 0].plot(stats['Generated Outlier (emb_combine)']['mean'], label='Generated Outlier (emb_combine)', alpha=0.8)
    axes[0, 0].set_title('Mean Values Across Dimensions')
    axes[0, 0].set_xlabel('Embedding Dimension')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation comparison
    axes[0, 1].plot(stats['Normal (emb_combine)']['std'], label='Normal (emb_combine)', alpha=0.8)
    axes[0, 1].plot(stats['Real Anomaly']['std'], label='Real Anomaly', alpha=0.8)
    axes[0, 1].plot(stats['Generated Outlier (emb_combine)']['std'], label='Generated Outlier (emb_combine)', alpha=0.8)
    axes[0, 1].set_title('Standard Deviation Across Dimensions')
    axes[0, 1].set_xlabel('Embedding Dimension')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Min values comparison
    axes[1, 0].plot(stats['Normal (emb_combine)']['min'], label='Normal (emb_combine)', alpha=0.8)
    axes[1, 0].plot(stats['Real Anomaly']['min'], label='Real Anomaly', alpha=0.8)
    axes[1, 0].plot(stats['Generated Outlier (emb_combine)']['min'], label='Generated Outlier (emb_combine)', alpha=0.8)
    axes[1, 0].set_title('Minimum Values Across Dimensions')
    axes[1, 0].set_xlabel('Embedding Dimension')
    axes[1, 0].set_ylabel('Minimum Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Max values comparison
    axes[1, 1].plot(stats['Normal (emb_combine)']['max'], label='Normal (emb_combine)', alpha=0.8)
    axes[1, 1].plot(stats['Real Anomaly']['max'], label='Real Anomaly', alpha=0.8)
    axes[1, 1].plot(stats['Generated Outlier (emb_combine)']['max'], label='Generated Outlier (emb_combine)', alpha=0.8)
    axes[1, 1].set_title('Maximum Values Across Dimensions')
    axes[1, 1].set_xlabel('Embedding Dimension')
    axes[1, 1].set_ylabel('Maximum Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_statistics_comparison_emb_combine.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def analyze_embedding_quality_emb_combine(normal_embeddings_combine, real_anomaly_embeddings, 
                                        generated_outliers, dataset_name, save_dir):
    """
    Analyze the quality of embeddings in emb_combine space.
    """
    print(f"\n=== Embedding Quality Analysis in emb_combine Space ===")
    
    # 1. Silhouette Analysis
    if len(real_anomaly_embeddings) > 0 and len(generated_outliers) > 0:
        # Combine embeddings and create labels
        all_embeddings = np.vstack([normal_embeddings_combine, real_anomaly_embeddings, generated_outliers])
        labels = ([0] * len(normal_embeddings_combine) + 
                 [1] * len(real_anomaly_embeddings) + 
                 [2] * len(generated_outliers))
        
        try:
            silhouette_avg = silhouette_score(all_embeddings, labels)
            print(f"Overall Silhouette Score in emb_combine space: {silhouette_avg:.4f}")
        except:
            print("Could not calculate silhouette score")
    
    # 2. Separation Analysis
    normal_centroid = np.mean(normal_embeddings_combine, axis=0)
    
    # Calculate average distances from normal centroid
    normal_dist = np.mean(np.linalg.norm(normal_embeddings_combine - normal_centroid, axis=1))
    real_dist = np.mean(np.linalg.norm(real_anomaly_embeddings - normal_centroid, axis=1))
    generated_dist = np.mean(np.linalg.norm(generated_outliers - normal_centroid, axis=1))
    
    print(f"Average distance from normal centroid in emb_combine space:")
    print(f"  Normal nodes (emb_combine): {normal_dist:.4f}")
    print(f"  Real anomalies: {real_dist:.4f}")
    print(f"  Generated outliers (emb_combine): {generated_dist:.4f}")
    
    # 3. Quality Metrics
    separation_ratio_real = real_dist / normal_dist if normal_dist > 0 else 0
    separation_ratio_generated = generated_dist / normal_dist if normal_dist > 0 else 0
    
    print(f"Separation ratios in emb_combine space (higher is better for anomalies):")
    print(f"  Real anomalies: {separation_ratio_real:.4f}")
    print(f"  Generated outliers: {separation_ratio_generated:.4f}")
    
    # 4. Similarity between real and generated anomalies
    if len(real_anomaly_embeddings) > 0 and len(generated_outliers) > 0:
        real_centroid = np.mean(real_anomaly_embeddings, axis=0)
        generated_centroid = np.mean(generated_outliers, axis=0)
        
        centroid_similarity = np.dot(real_centroid, generated_centroid) / (
            np.linalg.norm(real_centroid) * np.linalg.norm(generated_centroid))
        centroid_distance = np.linalg.norm(real_centroid - generated_centroid)
        
        print(f"Real vs Generated anomaly centroids in emb_combine space:")
        print(f"  Cosine similarity: {centroid_similarity:.4f}")
        print(f"  Euclidean distance: {centroid_distance:.4f}")
    
    # Save analysis results
    analysis_results = {
        'dataset': dataset_name,
        'space': 'emb_combine',
        'normal_centroid_distance': float(normal_dist),
        'real_anomaly_centroid_distance': float(real_dist),
        'generated_outlier_centroid_distance': float(generated_dist),
        'separation_ratio_real': float(separation_ratio_real),
        'separation_ratio_generated': float(separation_ratio_generated),
    }
    
    # Save to file
    import json
    with open(os.path.join(save_dir, f'{dataset_name}_quality_analysis_emb_combine.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Quality analysis for emb_combine space saved to {save_dir}")
