import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
import multiprocessing as mp
from functools import partial
import logging
from global_dataset import get_data_loaders
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import sys
# Set up logging directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"process_ssim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def compute_ssim_matrix(images, idx, return_dict, progress_dict):
    """Compute SSIM between one image and all others"""
    n = len(images)
    ssim_row = np.zeros(n)
    base_img = images[idx]
    
    # Log start
    logging.info(f"Worker {idx}: Starting SSIM computation")
    progress_dict[f"worker_{idx}"] = 0
    
    for j in range(n):
        if j != idx:
            ssim_row[j] = ssim(base_img, images[j], 
                              data_range=images[j].max() - images[j].min())
            
            # Log progress at 50%
            if j == n//2:
                logging.info(f"Worker {idx}: 50% complete")
                progress_dict[f"worker_{idx}"] = 50
    
    # Log completion
    logging.info(f"Worker {idx}: 100% complete")
    progress_dict[f"worker_{idx}"] = 100
    return_dict[idx] = ssim_row

def find_image_clusters(dataloader, n_clusters=5, num_workers=None):
    """
    Find clusters in image dataset based on SSIM similarity
    
    Args:
        dataloader: PyTorch dataloader containing images
        n_clusters: Number of clusters to find
        num_workers: Number of parallel workers (defaults to CPU count)
    
    Returns:
        cluster_labels: Cluster assignment for each image
        cluster_centers: Indices of cluster center images
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    # Extract all images into memory
    images = []
    labels = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            # print(batch[0].shape, batch[1])
            batch = batch[0]
            # sys.exit()# Assume first element is images
        images.extend([img.numpy().squeeze() for img in batch])
        labels.extend(batch[1])
    
    n_images = len(images)
    logging.info(f"Computing SSIM matrix for {n_images} images using {num_workers} workers")
    
    # Compute SSIM similarity matrix in parallel
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Add progress tracking dictionary
    progress_dict = manager.dict()
    
    pool = mp.Pool(processes=num_workers)
    compute_ssim_partial = partial(compute_ssim_matrix, images, 
                                 return_dict=return_dict,
                                 progress_dict=progress_dict)
    
    pool.map(compute_ssim_partial, range(n_images))
    pool.close()
    pool.join()
    
    # Convert results to similarity matrix
    similarity_matrix = np.zeros((n_images, n_images))
    for i in range(n_images):
        similarity_matrix[i] = return_dict[i]
    
    # Cluster using similarity matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    
    # Find images closest to cluster centers
    cluster_centers = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_similarities = similarity_matrix[cluster_mask].mean(axis=0)
        center_idx = np.argmax(cluster_similarities)
        cluster_centers.append(center_idx)
        
    logging.info(f"Found {n_clusters} clusters")
    
    return cluster_labels, cluster_centers, similarity_matrix
def visualize_clusters(images, cluster_labels, cluster_centers, save_path=None):
    """Visualize cluster results by showing center images and random samples from each cluster.
    
    Args:
        images: List of images
        cluster_labels: Cluster assignments for each image
        cluster_centers: Indices of cluster center images
        save_path: Optional path to save visualization plot
    """
    n_clusters = len(cluster_centers)
    n_samples = 5  # Number of random samples to show per cluster
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_clusters, n_samples + 1, figsize=(15, 3*n_clusters))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]
        
    for i in range(n_clusters):
        # Plot cluster center
        center_img = images[cluster_centers[i]]
        axes[i,0].imshow(center_img, cmap='gray')
        axes[i,0].set_title(f'Cluster {i}\nCenter')
        axes[i,0].axis('off')
        
        # Plot random samples
        cluster_indices = np.where(cluster_labels == i)[0]
        sample_indices = np.random.choice(cluster_indices, size=min(n_samples, len(cluster_indices)), replace=False)
        
        for j, idx in enumerate(sample_indices, 1):
            axes[i,j].imshow(images[idx], cmap='gray')
            axes[i,j].set_title(f'Sample {j}')
            axes[i,j].axis('off')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_cluster_representatives(similarity_matrix, cluster_labels, n_representatives=5, save_path=None):
    """
    Identify and save representative images for each cluster based on similarity scores.
    
    Args:
        similarity_matrix: Matrix of SSIM similarities between images
        cluster_labels: Cluster assignments for each image
        n_representatives: Number of representative images to find per cluster
        save_path: Path to save JSON output (optional)
    
    Returns:
        dict: Cluster representatives information
    """
    n_clusters = len(np.unique(cluster_labels))
    cluster_info = {}
    
    for i in range(n_clusters):
        # Get indices of images in this cluster
        cluster_mask = cluster_labels == i
        cluster_indices = np.where(cluster_mask)[0]
        
        # Calculate average similarity to all other images in the cluster
        cluster_similarities = similarity_matrix[cluster_mask][:, cluster_mask]
        avg_similarities = cluster_similarities.mean(axis=1)
        
        # Get top n_representatives images with highest average similarity
        top_indices = cluster_indices[np.argsort(avg_similarities)[-n_representatives:]]
        avg_scores = avg_similarities[np.argsort(avg_similarities)[-n_representatives:]]
        
        cluster_info[f"cluster_{i}"] = {
            "representative_indices": top_indices.tolist(),
            "similarity_scores": avg_scores.tolist(),
            "cluster_size": len(cluster_indices)
        }
        
        logging.info(f"Cluster {i}: Found {n_representatives} representative images "
                    f"from cluster of size {len(cluster_indices)}")
    
    # Save to JSON if path provided
    if save_path is None:
        save_path = os.path.join("outputs", 
                                f"cluster_representatives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(cluster_info, f, indent=4)
    
    logging.info(f"Saved cluster representatives to {save_path}")
    return cluster_info

# Example usage:
if __name__ == "__main__":
    json_path = "outputs/cluster_representatives.json"
    
    #  Define paths to datasets
    tb_data_path = "data/TB_Chest_Radiography_Database"
    indian_data_path = "data/indian_dataset"
    
    # Get data loaders
    loaders = get_data_loaders(tb_data_path, indian_data_path)
    dataloader = loaders['global_whole']
    
    # Extract images and labels
    images = []
    labels = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            images_batch, labels_batch = batch
        else:
            images_batch = batch
            labels_batch = None  # Handle case where labels are not provided
        images.extend([img.numpy().squeeze() for img in images_batch])
        if labels_batch is not None:
            labels.extend(labels_batch.numpy())
    
    if os.path.exists(json_path):
        # Load existing cluster representatives
        logging.info(f"Loading existing cluster representatives from {json_path}")
        with open(json_path, 'r') as f:
            cluster_info = json.load(f)
            
        # Create mapping from image index to cluster label
        cluster_labels = np.zeros(len(images))
        cluster_centers = []
        
        for cluster_id, info in cluster_info.items():
            cluster_num = int(cluster_id.split('_')[1])
            # Use the first representative as cluster center
            cluster_centers.append(info["representative_indices"][0])
            
            # Assign cluster labels based on representatives
            for idx in info["representative_indices"]:
                cluster_labels[idx] = cluster_num
                
        logging.info(f"Loaded {len(cluster_info)} clusters with their representatives")
        
        # Print labels of representative images
        for cluster_id, info in cluster_info.items():
            representative_labels = [labels[idx] for idx in info["representative_indices"]]
            logging.info(f"Cluster {cluster_id}: Representative labels: {representative_labels}")
        
        # Visualize using loaded cluster information
        visualize_clusters(images, cluster_labels, cluster_centers, 
                         save_path="outputs/cluster_visualization_loaded.png")
        
    else:
        # Perform clustering
        n_clusters = 10
        cluster_labels, cluster_centers, similarity_matrix = find_image_clusters(
            dataloader, 
            n_clusters=n_clusters,
            num_workers=30
        )
        
        # Visualize results
        visualize_clusters(images, cluster_labels, cluster_centers, 
                         save_path="outputs/cluster_visualization.png")
        
        # Save representative images
        cluster_info = save_cluster_representatives(
            similarity_matrix,
            cluster_labels,
            n_representatives=5,
            save_path=json_path
        )
