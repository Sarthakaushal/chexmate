import json
import matplotlib.pyplot as plt
import os
import numpy as np
from global_dataset import get_data_loaders

def load_images():
    """Load images from the dataset"""
    # Define paths to datasets
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
            labels_batch = None
        images.extend([img.numpy().squeeze() for img in images_batch])
        if labels_batch is not None:
            labels.extend(labels_batch.numpy())
    
    return images, labels

def visualize_cluster_representatives(json_path, images, labels):
    """Visualize cluster representatives with their labels"""
    # Load cluster representatives from JSON
    with open(json_path, 'r') as f:
        cluster_info = json.load(f)
    
    # Create figure for all clusters
    n_clusters = len(cluster_info)
    n_representatives = len(cluster_info["cluster_0"]["representative_indices"])
    
    fig = plt.figure(figsize=(20, 4*n_clusters))
    
    for cluster_idx, (cluster_id, info) in enumerate(cluster_info.items()):
        # Get cluster information
        rep_indices = info["representative_indices"]
        cluster_size = info["cluster_size"]
        
        # Create subplot for this cluster
        for rep_idx, img_idx in enumerate(rep_indices):
            ax = plt.subplot(n_clusters, n_representatives, 
                           cluster_idx * n_representatives + rep_idx + 1)
            
            # Display image
            ax.imshow(images[img_idx], cmap='gray')
            ax.axis('off')
            
            # Get label (0: Normal, 1: TB)
            label = labels[img_idx]
            label_text = "TB" if label == 1 else "Normal"
            
            # Set title with cluster and class information
            title = f"Cluster {cluster_id}\n{label_text}\nImage {img_idx}"
            if rep_idx == 0:
                title += f"\nCluster Size: {cluster_size}"
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig('outputs/cluster_visualization.png')
    plt.show()

def main():
    json_path = "outputs/cluster_representatives.json"
    
    # Load images and labels
    images, labels = load_images()
    
    # Visualize clusters
    visualize_cluster_representatives(json_path, images, labels)

if __name__ == "__main__":
    main()

