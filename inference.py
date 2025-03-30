#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2022/12/06
@Author  :   Xincan Lin
@Desc    :   Multi-view clustering inference script
'''

import argparse
import torch
import os
import numpy as np
from sklearn.manifold import TSNE

from Nets.SEL import DDCL
from Utils.clusteringPerformance1 import StatisticClustering, spectral_clustering1
from Utils.loadMatData import loadData, normalization

def get_args():
    """Configure inference parameters"""
    parser = argparse.ArgumentParser(description="Multi-view clustering inference")
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='MSRC_v2')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--cuda', type=int, default=torch.cuda.is_available())
    parser.add_argument('--cuda_device', type=str, default='0')
    return parser.parse_args()

def setup_device(args):
    """Setup inference device"""
    device = f"cuda:{args.cuda_device}" if args.cuda else "cpu"
    return device

def load_and_preprocess_data(args, device):
    """Load and preprocess test data"""
    dataset_path = os.path.join(args.dataset_dir, f"{args.dataset}.mat")
    features_ori, gnd, category = loadData(dataset_path)
    features = features_ori[0]
    views = features.shape[0]
    N = gnd.shape[0]
    
    # Preprocess features
    fea_dims = []
    processed_features = []
    for i in range(views):
        fea_dims.append(features[i].shape[1])
        feai = normalization(features[i])
        processed_features.append(torch.from_numpy(feai).to(device).double())
    
    return processed_features, gnd, category, views, N, fea_dims

def load_model(model_path, model_config, device):
    """Load trained model"""
    model = DDCL(**model_config).to(device).double()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def inference(model, features):
    """Perform inference"""
    with torch.no_grad():
        encodings, _, fusionEncoding, _, selfExp, _, _ = model(features)
        return fusionEncoding, selfExp

def cluster_data(fusion_features, shared_rep, gnd, category):
    """Perform clustering on features"""
    # K-means clustering
    fusion_features = fusion_features.detach().cpu().numpy()
    kmeans_perf = StatisticClustering(features=fusion_features, gnd=gnd, clusterNum=category)
    
    # Spectral clustering
    shared_rep = shared_rep.detach().cpu().numpy()
    Z = (abs(shared_rep) + abs(shared_rep.T)) / 2
    spectral_perf = spectral_clustering1(points=Z, gnd=gnd, k=category)
    
    return kmeans_perf, spectral_perf

def visualize_features(fusion_features):
    """Visualize features using t-SNE"""
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(fusion_features.detach().cpu().numpy())
    return features_2d

def main():
    # Initialize
    args = get_args()
    device = setup_device(args)
    
    # Load and preprocess data
    features, gnd, category, views, N, fea_dims = load_and_preprocess_data(args, device)
    
    # Model configuration
    model_config = {
        'N': N,
        'views': views,
        'fea_dims': fea_dims,
        'category': category,
        'device': device
    }
    
    # Load model
    model = load_model(args.model_path, model_config, device)
    
    # Perform inference
    fusion_features, shared_rep = inference(model, features)
    
    # Perform clustering
    kmeans_perf, spectral_perf = cluster_data(fusion_features, shared_rep, gnd, category)
    
    # Print clustering results
    print("\nSpectral Clustering Results:")
    print(f"ACC: {spectral_perf[0]:.4f}, NMI: {spectral_perf[1]:.4f}")
    

if __name__ == '__main__':
    main()