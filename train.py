#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo_simplified.py
@Time    :   2022/12/06
@Author  :   Xincan Lin
@Desc    :   Multi-view clustering learning training script
'''

import argparse
import torch
import os
import numpy as np
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE

from Losses.losses import CMVCLoss
from Nets.SEL import DDCL, weight_init
from Utils.clusteringPerformance1 import StatisticClustering, spectral_clustering1
from Utils.loadMatData import loadData, normalization

def get_args():
    """Configure training parameters"""
    parser = argparse.ArgumentParser(description="Multi-view clustering training")
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='MSRC_v2')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--cuda', type=int, default=torch.cuda.is_available())
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--lambda_', type=float, default=1)
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment"""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = f"cuda:{args.cuda_device}" if args.cuda else "cpu"
    return device

def load_and_preprocess_data(args, device):
    """Load and preprocess data"""
    # Load data
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

def calculate_losses(features, model_outputs, args):
    """Calculate all losses"""
    encodings, selfEncodings, fusionEncoding, selfEncoding, selfExp, selfReps, decodings = model_outputs
    
    # Basic losses
    loss_ae = sum(F.mse_loss(f, d, reduction='mean') for f, d in zip(features, decodings))
    loss_se = sum(F.mse_loss(e, se, reduction='mean') for e, se in zip(encodings, selfEncodings))
    
    # Regularization loss
    loss_norm = sum(torch.mean(torch.pow(sr, 2)) for sr in selfReps)
    
    # Contrastive losses
    contLoss = CMVCLoss(batch_size=len(features[0]), temperature=1.0, device=features[0].device)
    loss_contrast_sf = sum(contLoss(sr, selfExp) for sr in selfReps)
    loss_contrast_h = sum(contLoss(se, fusionEncoding) for se in selfEncodings)
    
    # Total loss
    total_loss = loss_ae + args.beta * (loss_se + loss_norm) + args.gamma * loss_contrast_sf + args.lambda_ * loss_contrast_h
    
    return total_loss, loss_ae, loss_se, loss_contrast_sf, loss_contrast_h

def pre_train_model(model, features, args):
    """Pre-train the model to initialize self-expression coefficients"""
    preTrainPath = os.path.join("pretrain", f"{args.dataset}.pkl")
    
    # Check if pre-trained model exists
    if os.path.exists(preTrainPath):
        params = torch.load(preTrainPath)
        model.load_state_dict(params)
        print("Pre-trained model loaded successfully")
    else:
        print("Starting model pre-training...")
        if not os.path.exists("pretrain"):
            os.makedirs("pretrain")
            
        # Create optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.9, patience=10, min_lr=1e-10)
        
        # Pre-training loop
        pre_epochs = 500  # Number of pre-training epochs
        for epoch in range(1, pre_epochs + 1):
            optimizer.zero_grad()
            
            # Only use encoder-decoder for pre-training
            encodings, decodings = model.preTrain(features)
            
            # Calculate reconstruction loss
            loss_ae = 0
            for i in range(len(features)):
                loss_ae += F.mse_loss(features[i], decodings[i], reduction='mean')
            
            # Backpropagation
            loss_ae.backward()
            optimizer.step()
            scheduler.step(loss_ae)
            
            # Print training progress
            if epoch % 50 == 0:
                print(f"Pre-training epoch {epoch}, reconstruction loss: {loss_ae.item():.4f}")
        
        # Save pre-trained model
        torch.save(model.state_dict(), preTrainPath)
        print("Pre-training completed, model saved")
    
    return model


def train_model(model, features, args):
    """Train the model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10, min_lr=1e-8)
    
    for epoch in range(1, args.n_epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Calculate losses
        total_loss, loss_ae, loss_se, loss_csf, loss_ch = calculate_losses(features, outputs, args)
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch}:")
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"AE Loss: {loss_ae.item():.4f}")
            print(f"SE Loss: {loss_se.item():.4f}")
            print("-" * 50)
    
    return model

def evaluate_model(model, features, gnd, category):
    """Evaluate model performance"""
    with torch.no_grad():
        encodings, _, _, fusionEncoding, _, selfExp, selfReps, _ = model(features)
        
        # K-means clustering evaluation
        fusion_features = fusionEncoding.detach().cpu().numpy()
        kmeans_perf = StatisticClustering(features=fusion_features, gnd=gnd, clusterNum=category)
        
        # Spectral clustering evaluation
        shared_rep = selfExp.detach().cpu().numpy()
        Z = (abs(shared_rep) + abs(shared_rep.T)) / 2
        spectral_perf = spectral_clustering1(points=Z, gnd=gnd, k=category)
        
        return kmeans_perf, spectral_perf

def main():
    # Initialize
    args = get_args()
    device = setup_environment(args)
    
    # Load and preprocess data
    features, gnd, category, views, N, fea_dims = load_and_preprocess_data(args, device)
    
    # Create model
    model = DDCL(
        N=N,
        views=views,
        fea_dims=fea_dims,
        category=category,
        device=device
    ).to(device).double()
    model.apply(weight_init)
    
    # Pre-train model
    model = pre_train_model(model, features, args)

    # Train model
    model = train_model(model, features, args)
    
    # Evaluate model
    kmeans_perf, spectral_perf = evaluate_model(model, features, gnd, category)
    
    # Print results
    print("\nSpectral Clustering Performance:")
    print(f"ACC: {spectral_perf[0]:.4f}, NMI: {spectral_perf[1]:.4f}")

if __name__ == '__main__':
    main()