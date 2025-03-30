# Deep Dual Contrastive Clustering for Multiv-View Subspace Clustering

## Train

The train.py script is used for training multi-view clustering models. This script implements a fusion of autoencoder structure and self-expression learning, which can effectively handle clustering tasks for multi-view data.

### Usage

```bash
python train.py --dataset_dir <dataset_directory> --dataset <dataset_name> --cuda_device <GPU_device_id> --n_epochs <training_epochs> --lr <learning_rate>
```

### Main Parameters

- `--dataset_dir`: Directory containing datasets (default: './datasets')
- `--dataset`: Dataset name (default: 'MSRC_v2')
- `--seed`: Random seed (default: 2022)
- `--cuda`: Whether to use GPU (default: auto-detect)
- `--cuda_device`: GPU device ID (default: '0')
- `--lr`: Learning rate (default: 1e-4)
- `--n_epochs`: Number of training epochs (default: 500)
- `--beta`: Weight for autoencoder reconstruction loss (default: 1)
- `--gamma`: Weight for self-expression contrastive loss (default: 1)
- `--lambda_`: Weight for fusion feature contrastive loss (default: 1)

### Features

- Automatically loads and preprocesses multi-view data
- Pre-training phase to initialize self-expression coefficients
- Uses contrastive learning to optimize multi-view representations
- Automatically evaluates K-means and spectral clustering performance after training

## Inference

The inference.py script is used to perform clustering analysis on multi-view data using a trained model.

### Usage

```bash
python inference.py --model_path <model_path> --dataset_dir <dataset_directory> --dataset <dataset_name> --cuda_device <GPU_device_id>
```

### Main Parameters

- `--dataset_dir`: Directory containing datasets (default: './datasets')
- `--dataset`: Dataset name (default: 'MSRC_v2')
- `--model_path`: Path to the trained model weights file (required)
- `--cuda`: Whether to use GPU (default: auto-detect)
- `--cuda_device`: GPU device ID (default: '0')

### Features

- Loads the trained model
- Performs feature extraction and fusion on input multi-view data
- Executes K-means and spectral clustering analysis
- Outputs evaluation metrics such as clustering accuracy (ACC) and normalized mutual information (NMI)