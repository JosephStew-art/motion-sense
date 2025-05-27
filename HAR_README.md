# Human Activity Recognition with MotionSense Dataset

This project implements a machine learning model for human activity recognition (HAR) using the MotionSense dataset. The model is trained to classify six different activities (walking, jogging, sitting, standing, upstairs, downstairs) based on sensor data from mobile devices.

## Features

- Comprehensive data preprocessing with detailed statistics
- Subject-wise data splitting to prevent information leakage
- Multiple model architectures (CNN-LSTM and Transformer)
- Real-time training visualization with progress plots
- Detailed evaluation metrics and visualizations
- Proper normalization to ensure model generalization

## Project Structure

```
motion-sense/
├── data/                  # Dataset directory (already exists)
├── src/                   # Source code directory
│   ├── __init__.py        # Make src a Python package
│   ├── data_utils.py      # Data loading and preprocessing functions
│   ├── models.py          # Model architectures
│   ├── train_utils.py     # Training and evaluation functions
│   └── main.py            # Main script to run the pipeline
├── results/               # Results directory (created during execution)
└── requirements.txt       # Dependencies
```

## Installation

1. Install the required dependencies:
```bash
cd motion-sense
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with default parameters:

```bash
python src/main.py
```

This will:
1. Load and preprocess the data
2. Split the data into training, validation, and test sets
3. Train a CNN-LSTM model
4. Evaluate the model on the test set
5. Save results to the `results` directory

### Advanced Usage

Customize the training process with command-line arguments:

```bash
python src/main.py --data_dir data/A_DeviceMotion_data \
                   --window_size 128 \
                   --overlap 0.5 \
                   --model_type cnnlstm \
                   --hidden_dim 128 \
                   --num_layers 2 \
                   --batch_size 64 \
                   --num_epochs 50 \
                   --learning_rate 0.001 \
                   --output_dir results/cnnlstm
```

### Command-Line Arguments

- **Data parameters**:
  - `--data_dir`: Path to the A_DeviceMotion_data folder (default: 'data/A_DeviceMotion_data')
  - `--window_size`: Size of the sliding window for segmentation (default: 128)
  - `--overlap`: Overlap between consecutive windows (default: 0.5)

- **Model parameters**:
  - `--model_type`: Type of model to use ('cnnlstm' or 'transformer', default: 'cnnlstm')
  - `--hidden_dim`: Hidden dimension for LSTM or Transformer (default: 128)
  - `--num_layers`: Number of layers in LSTM or Transformer (default: 2)
  - `--dropout`: Dropout rate (default: 0.5)

- **Training parameters**:
  - `--batch_size`: Batch size for training (default: 64)
  - `--num_epochs`: Number of epochs to train (default: 100)
  - `--learning_rate`: Learning rate (default: 0.001)
  - `--weight_decay`: Weight decay (L2 penalty, default: 1e-5)
  - `--patience`: Patience for early stopping (default: 10)

- **Output parameters**:
  - `--output_dir`: Directory to save results (default: 'results')
  - `--seed`: Random seed for reproducibility (default: 42)

## Monitoring Training Progress

During training, the script generates real-time visualizations that are saved to the output directory:

- `training_progress.png`: Updated after each epoch with loss, accuracy, learning rate, and confusion matrix
- `epoch_X_progress.png`: Snapshot of training progress at epoch X
- `training_log.txt`: Detailed log of training metrics for each epoch

You can monitor the training progress by opening these files in your preferred image viewer or text editor.

## Results

After training and evaluation, the following files are generated in the output directory:

- `best_model.pth`: The trained model weights
- `training_history.png`: Plot of training and validation loss/accuracy
- `training_history.csv`: CSV file with training metrics for each epoch
- `confusion_matrix.png`: Confusion matrix of the test results
- `roc_curves.png`: ROC curves for each class
- `precision_recall_curves.png`: Precision-recall curves for each class
- `evaluation_results.txt`: Detailed evaluation metrics
- `test_predictions.csv`: Predictions and probabilities for each test sample
- `preprocessing_summary.json`: Summary of preprocessing steps and parameters

## Example Models

### CNN-LSTM Model

The CNN-LSTM model combines convolutional layers for feature extraction with LSTM layers for temporal modeling:

```
CNNLSTM(
  input_dim=12,
  hidden_dim=128,
  num_layers=2,
  num_classes=6,
  dropout=0.5
)
```

### Transformer Model

The Transformer model uses self-attention mechanisms to capture temporal dependencies:

```
TransformerModel(
  input_dim=12,
  d_model=128,
  nhead=8,
  num_layers=4,
  dim_feedforward=512,
  num_classes=6,
  dropout=0.1
)
```

## Troubleshooting

If you encounter any issues:

1. **Memory errors**: Reduce the batch size or window size
2. **Slow training**: Reduce the number of layers or hidden dimensions
3. **Overfitting**: Increase dropout or weight decay
4. **Poor performance**: Try different hyperparameters or model architectures

## Acknowledgments

- The MotionSense dataset was created by researchers at Queen Mary University of London
- This implementation is based on PyTorch and scikit-learn
