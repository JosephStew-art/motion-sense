import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_utils import (
    load_and_preprocess_data,
    subject_wise_split,
    normalize_data,
    MotionSenseDataset
)
from models import CNNLSTM, TransformerModel
from train_utils import train_model, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Human Activity Recognition using MotionSense Dataset')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/A_DeviceMotion_data',
                        help='Path to the A_DeviceMotion_data folder')
    parser.add_argument('--window_size', type=int, default=128,
                        help='Size of the sliding window for segmentation')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap between consecutive windows (0-1)')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnnlstm', choices=['cnnlstm', 'transformer'],
                        help='Type of model to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for LSTM or Transformer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in LSTM or Transformer')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Class names
    class_names = ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging']

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, subject_ids, preprocessing_info = load_and_preprocess_data(args.data_dir, args.window_size, args.overlap)

    # Split data
    print("Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = subject_wise_split(X, y, subject_ids)

    # Normalize data
    print("Normalizing data...")
    X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)

    # Save preprocessing information
    preprocessing_summary = {
        'window_size': args.window_size,
        'overlap': args.overlap,
        'total_samples': len(X),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'input_shape': X_train.shape,
        'num_classes': len(np.unique(y)),
        'class_names': ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging'],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    # Save preprocessing summary to a file
    import json
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'preprocessing_summary.json'), 'w') as f:
        json.dump(preprocessing_summary, f, indent=4)

    print(f"\nPreprocessing summary saved to {os.path.join(args.output_dir, 'preprocessing_summary.json')}")

    # Create datasets and dataloaders
    train_dataset = MotionSenseDataset(X_train, y_train)
    val_dataset = MotionSenseDataset(X_val, y_val)
    test_dataset = MotionSenseDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    print(f"Initializing {args.model_type} model...")
    input_dim = X_train.shape[2]  # Number of features

    if args.model_type == 'cnnlstm':
        model = CNNLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=len(class_names),
            dropout=args.dropout
        )
    else:  # transformer
        model = TransformerModel(
            input_dim=input_dim,
            d_model=args.hidden_dim,
            nhead=8,  # Number of attention heads
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_dim * 4,
            num_classes=len(class_names),
            dropout=args.dropout
        )

    # Train model
    print("Training model...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=args.output_dir
    )

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, test_loader, class_names, save_dir=args.output_dir)

    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
